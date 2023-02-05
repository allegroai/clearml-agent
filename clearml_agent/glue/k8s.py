from __future__ import print_function, division, unicode_literals

import base64
import functools
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from threading import Thread
from time import sleep, time
from typing import Text, List, Callable, Any, Collection, Optional, Union, Iterable, Dict, Tuple, Set

import yaml

from clearml_agent.backend_api.session import Request
from clearml_agent.commands.events import Events
from clearml_agent.commands.worker import Worker, get_task_container, set_task_container, get_next_task
from clearml_agent.definitions import (
    ENV_DOCKER_IMAGE,
    ENV_AGENT_GIT_USER,
    ENV_AGENT_GIT_PASS,
    ENV_FORCE_SYSTEM_SITE_PACKAGES,
)
from clearml_agent.errors import APIError
from clearml_agent.glue.definitions import ENV_START_AGENT_SCRIPT_PATH
from clearml_agent.helper.base import safe_remove_file
from clearml_agent.helper.dicts import merge_dicts
from clearml_agent.helper.process import get_bash_output, stringify_bash_output
from clearml_agent.helper.resource_monitor import ResourceMonitor
from clearml_agent.interface.base import ObjectID



class K8sIntegration(Worker):
    K8S_PENDING_QUEUE = "k8s_scheduler"

    K8S_DEFAULT_NAMESPACE = "clearml"
    AGENT_LABEL = "CLEARML=agent"

    KUBECTL_APPLY_CMD = "kubectl apply --namespace={namespace} -f"

    KUBECTL_DELETE_CMD = "kubectl delete pods " \
                         "-l={agent_label} " \
                         "--field-selector=status.phase!=Pending,status.phase!=Running " \
                         "--namespace={namespace} " \
                         "--output name"

    BASH_INSTALL_SSH_CMD = [
        "apt-get update",
        "apt-get install -y openssh-server",
        "mkdir -p /var/run/sshd",
        "echo 'root:training' | chpasswd",
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
        "sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config",
        r"sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd",
        "echo 'AcceptEnv TRAINS_API_ACCESS_KEY TRAINS_API_SECRET_KEY CLEARML_API_ACCESS_KEY CLEARML_API_SECRET_KEY' "
        ">> /etc/ssh/sshd_config",
        'echo "export VISIBLE=now" >> /etc/profile',
        'echo "export PATH=$PATH" >> /etc/profile',
        'echo "ldconfig" >> /etc/profile',
        "/usr/sbin/sshd -p {port}"]

    DEFAULT_EXECUTION_AGENT_ARGS = os.getenv("K8S_GLUE_DEF_EXEC_AGENT_ARGS", "--full-monitoring --require-queue")
    POD_AGENT_INSTALL_ARGS = os.getenv("K8S_GLUE_POD_AGENT_INSTALL_ARGS", "")

    CONTAINER_BASH_SCRIPT = [
        "export DEBIAN_FRONTEND='noninteractive'",
        "echo 'Binary::apt::APT::Keep-Downloaded-Packages \"true\";' > /etc/apt/apt.conf.d/docker-clean",
        "chown -R root /root/.cache/pip",
        "apt-get update",
        "apt-get install -y git libsm6 libxext6 libxrender-dev libglib2.0-0",
        "declare LOCAL_PYTHON",
        "[ ! -z $LOCAL_PYTHON ] || for i in {{15..5}}; do which python3.$i && python3.$i -m pip --version && "
        "export LOCAL_PYTHON=$(which python3.$i) && break ; done",
        "[ ! -z $LOCAL_PYTHON ] || apt-get install -y python3-pip",
        "[ ! -z $LOCAL_PYTHON ] || export LOCAL_PYTHON=python3",
        "{extra_bash_init_cmd}",
        "$LOCAL_PYTHON -m pip install clearml-agent{agent_install_args}",
        "{extra_docker_bash_script}",
        "$LOCAL_PYTHON -m clearml_agent execute {default_execution_agent_args} --id {task_id}"
    ]

    DEFAULT_POD_NAME_PREFIX = "clearml-id-"
    DEFAULT_LIMIT_POD_LABEL = "ai.allegro.agent.serial=pod-{pod_number}"

    _edit_hyperparams_version = "2.9"

    def __init__(
            self,
            k8s_pending_queue_name=None,
            container_bash_script=None,
            debug=False,
            ports_mode=False,
            num_of_services=20,
            base_pod_num=1,
            user_props_cb=None,
            overrides_yaml=None,
            template_yaml=None,
            clearml_conf_file=None,
            extra_bash_init_script=None,
            namespace=None,
            max_pods_limit=None,
            pod_name_prefix=None,
            limit_pod_label=None,
            **kwargs
    ):
        """
        Initialize the k8s integration glue layer daemon

        :param str k8s_pending_queue_name: queue name to use when task is pending in the k8s scheduler
        :param str container_bash_script: container bash script to be executed in k8s (default: CONTAINER_BASH_SCRIPT)
            Notice this string will use format() call, if you have curly brackets they should be doubled { -> {{
            Format arguments passed: {task_id} and {extra_bash_init_cmd}
        :param bool debug: Switch logging on
        :param bool ports_mode: Adds a label to each pod which can be used in services in order to expose ports.
            Requires the `num_of_services` parameter.
        :param int num_of_services: Number of k8s services configured in the cluster. Required if `port_mode` is True.
            (default: 20)
        :param int base_pod_num: Used when `ports_mode` is True, sets the base pod number to a given value (default: 1)
        :param callable user_props_cb: An Optional callable allowing additional user properties to be specified
            when scheduling a task to run in a pod. Callable can receive an optional pod number and should return
            a dictionary of user properties (name and value). Signature is [[Optional[int]], Dict[str,str]]
        :param str overrides_yaml: YAML file containing the overrides for the pod (optional)
        :param str template_yaml: YAML file containing the template for the pod (optional).
            If provided the pod is scheduled with kubectl apply and overrides are ignored, otherwise with kubectl run.
        :param str clearml_conf_file: clearml.conf file to be use by the pod itself (optional)
        :param str extra_bash_init_script: Additional bash script to run before starting the Task inside the container
        :param str namespace: K8S namespace to be used when creating the new pods (default: clearml)
        :param int max_pods_limit: Maximum number of pods that K8S glue can run at the same time
        """
        super(K8sIntegration, self).__init__()
        self.pod_name_prefix = pod_name_prefix or self.DEFAULT_POD_NAME_PREFIX
        self.limit_pod_label = limit_pod_label or self.DEFAULT_LIMIT_POD_LABEL
        self.k8s_pending_queue_name = k8s_pending_queue_name or self.K8S_PENDING_QUEUE
        self.k8s_pending_queue_id = None
        self.container_bash_script = container_bash_script or self.CONTAINER_BASH_SCRIPT
        force_system_packages = ENV_FORCE_SYSTEM_SITE_PACKAGES.get()
        self._force_system_site_packages = force_system_packages if force_system_packages is not None else True
        if self._force_system_site_packages:
            # Use system packages, because by we will be running inside a docker
            self._session.config.put("agent.package_manager.system_site_packages", True)
        # Add debug logging
        if debug:
            self.log.logger.disabled = False
            self.log.logger.setLevel(logging.DEBUG)
            self.log.logger.addHandler(logging.StreamHandler())
        self.ports_mode = ports_mode
        self.num_of_services = num_of_services
        self.base_pod_num = base_pod_num
        self._edit_hyperparams_support = None
        self._user_props_cb = user_props_cb
        self.conf_file_content = None
        self.overrides_json_string = None
        self.template_dict = None
        self.extra_bash_init_script = extra_bash_init_script or None
        if self.extra_bash_init_script and not isinstance(self.extra_bash_init_script, str):
            self.extra_bash_init_script = ' ; '.join(self.extra_bash_init_script)  # noqa
        self.namespace = namespace or self.K8S_DEFAULT_NAMESPACE
        self.pod_limits = []
        self.pod_requests = []
        self.max_pods_limit = max_pods_limit if not self.ports_mode else None

        self._load_overrides_yaml(overrides_yaml)

        if template_yaml:
            self.template_dict = self._load_template_file(template_yaml)

        clearml_conf_file = clearml_conf_file or kwargs.get('trains_conf_file')

        if clearml_conf_file:
            with open(os.path.expandvars(os.path.expanduser(str(clearml_conf_file))), 'rt') as f:
                self.conf_file_content = f.read()

        self._agent_label = None

        self._monitor_hanging_pods()

        self._min_cleanup_interval_per_ns_sec = 1.0
        self._last_pod_cleanup_per_ns = defaultdict(lambda: 0.)

    def _load_overrides_yaml(self, overrides_yaml):
        if not overrides_yaml:
            return
        overrides = self._load_template_file(overrides_yaml)
        if not overrides:
            return
        containers = overrides.get('spec', {}).get('containers', [])
        for c in containers:
            resources = {str(k).lower(): v for k, v in c.get('resources', {}).items()}
            if not resources:
                continue
            if resources.get('limits'):
                self.pod_limits += ['{}={}'.format(k, v) for k, v in resources['limits'].items()]
            if resources.get('requests'):
                self.pod_requests += ['{}={}'.format(k, v) for k, v in resources['requests'].items()]
        # remove double entries
        self.pod_limits = list(set(self.pod_limits))
        self.pod_requests = list(set(self.pod_requests))
        if self.pod_limits or self.pod_requests:
            self.log.warning('Found pod container requests={} limits={}'.format(
                self.pod_limits, self.pod_requests))
        if containers:
            self.log.warning('Removing containers section: {}'.format(overrides['spec'].pop('containers')))
        self.overrides_json_string = json.dumps(overrides)

    def _monitor_hanging_pods(self):
        _check_pod_thread = Thread(target=self._monitor_hanging_pods_daemon)
        _check_pod_thread.daemon = True
        _check_pod_thread.start()

    @staticmethod
    def _load_template_file(path):
        with open(os.path.expandvars(os.path.expanduser(str(path))), 'rt') as f:
            return yaml.load(f, Loader=getattr(yaml, 'FullLoader', None))

    @staticmethod
    def _get_path(d, *path, default=None):
        try:
            return functools.reduce(
                lambda a, b: a[b], path, d
            )
        except (IndexError, KeyError):
            return default

    def _get_kubectl_options(self, command, extra_labels=None, filters=None, output="json", labels=None):
        # type: (str, Iterable[str], Iterable[str], str, Iterable[str]) -> Dict
        if not labels:
            labels = [self._get_agent_label()]
        labels = list(labels) + (list(extra_labels) if extra_labels else [])
        d = {
            "-l": ",".join(labels),
            "-n": str(self.namespace),
            "-o": output,
        }
        if filters:
            d["--field-selector"] = ",".join(filters)
        return d

    def get_kubectl_command(self, command, output="json", **args):
        opts = self._get_kubectl_options(command, output=output, **args)
        return 'kubectl {command} {opts}'.format(
            command=command, opts=" ".join(x for item in opts.items() for x in item)
        )

    def _monitor_hanging_pods_daemon(self):
        last_tasks_msgs = {}  # last msg updated for every task

        while True:
            kubectl_cmd = self.get_kubectl_command("get pods", filters=["status.phase=Pending"])
            self.log.debug("Detecting hanging pods: {}".format(kubectl_cmd))
            output = stringify_bash_output(get_bash_output(kubectl_cmd))
            try:
                output_config = json.loads(output)
            except Exception as ex:
                self.log.warning('K8S Glue pods monitor: Failed parsing kubectl output:\n{}\nEx: {}'.format(output, ex))
                sleep(self._polling_interval)
                continue
            pods = output_config.get('items', [])
            task_ids = set()
            for pod in pods:
                pod_name = pod.get('metadata', {}).get('name', None)
                if not pod_name:
                    continue

                task_id = pod_name.rpartition('-')[-1]
                if not task_id:
                    continue

                namespace = pod.get('metadata', {}).get('namespace', None)
                if not namespace:
                    continue

                task_ids.add(task_id)

                msg = None

                waiting = self._get_path(pod, 'status', 'containerStatuses', 0, 'state', 'waiting')
                if not waiting:
                    condition = self._get_path(pod, 'status', 'conditions', 0)
                    if condition:
                        reason = condition.get('reason')
                        if reason == 'Unschedulable':
                            message = condition.get('message')
                            msg = reason + (" ({})".format(message) if message else "")
                else:
                    reason = waiting.get("reason", None)
                    message = waiting.get("message", None)

                    msg = reason + (" ({})".format(message) if message else "")

                    if reason == 'ImagePullBackOff':
                        delete_pod_cmd = 'kubectl delete pods {} -n {}'.format(pod_name, namespace)
                        self.log.debug(" - deleting pod due to ImagePullBackOff: {}".format(delete_pod_cmd))
                        get_bash_output(delete_pod_cmd)
                        try:
                            self.log.debug(" - Detecting hanging pods: {}".format(kubectl_cmd))
                            self._session.api_client.tasks.failed(
                                task=task_id,
                                status_reason="K8S glue error: {}".format(msg),
                                status_message="Changed by K8S glue",
                                force=True
                            )
                        except Exception as ex:
                            self.log.warning(
                                'K8S Glue pods monitor: Failed deleting task "{}"\nEX: {}'.format(task_id, ex)
                            )

                        # clean up any msg for this task
                        last_tasks_msgs.pop(task_id, None)
                        continue
                if msg and last_tasks_msgs.get(task_id, None) != msg:
                    try:
                        result = self._session.send_request(
                            service='tasks',
                            action='update',
                            json={"task": task_id, "status_message": "K8S glue status: {}".format(msg)},
                            method=Request.def_method,
                            async_enable=False,
                        )
                        if not result.ok:
                            result_msg = self._get_path(result.json(), 'meta', 'result_msg')
                            raise Exception(result_msg or result.text)

                        # update last msg for this task
                        last_tasks_msgs[task_id] = msg
                    except Exception as ex:
                        self.log.warning(
                            'K8S Glue pods monitor: Failed setting status message for task "{}"\nMSG: {}\nEX: {}'.format(
                                task_id, msg, ex
                            )
                        )

            # clean up any last message for a task that wasn't seen as a pod
            last_tasks_msgs = {k: v for k, v in last_tasks_msgs.items() if k in task_ids}

            sleep(self._polling_interval)

    def _set_task_user_properties(self, task_id: str, task_session=None, **properties: str):
        session = task_session or self._session
        if self._edit_hyperparams_support is not True:
            # either not supported or never tested
            if self._edit_hyperparams_support == self._session.api_version:
                # tested against latest api_version, not supported
                return
            if not self._session.check_min_api_version(self._edit_hyperparams_version):
                # not supported due to insufficient api_version
                self._edit_hyperparams_support = self._session.api_version
                return
        try:
            session.get(
                service="tasks",
                action="edit_hyper_params",
                task=task_id,
                hyperparams=[
                    {
                        "section": "properties",
                        "name": k,
                        "value": str(v),
                    }
                    for k, v in properties.items()
                ],
            )
            # definitely supported
            self._runtime_props_support = True
        except APIError as error:
            if error.code == 404:
                self._edit_hyperparams_support = self._session.api_version

    def _get_agent_label(self):
        if not self.worker_id:
            print('WARNING! no worker ID found!!!')
            return self.AGENT_LABEL

        if not self._agent_label:
            h = hashlib.md5()
            h.update(str(self.worker_id).encode('utf-8'))
            self._agent_label = '{}-{}'.format(self.AGENT_LABEL, h.hexdigest()[:8])

        return self._agent_label

    def _get_used_pods(self):
        # type: () -> Tuple[int, Set[str]]
        # noinspection PyBroadException
        try:
            kubectl_cmd = self.get_kubectl_command(
                "get pods",
                output="jsonpath=\"{range .items[*]}{.metadata.name}{' '}{.metadata.namespace}{'\\n'}{end}\""
            )
            self.log.debug("Getting used pods: {}".format(kubectl_cmd))
            output = stringify_bash_output(get_bash_output(kubectl_cmd, raise_error=True))

            if not output:
                # No such pod exist so we can use the pod_number we found
                return 0, set([])

            try:
                items = output.splitlines()
                current_pod_count = len(items)
                namespaces = {item.rpartition(" ")[-1] for item in items}
                self.log.debug(" - found {} pods in namespaces {}".format(current_pod_count, ", ".join(namespaces)))
            except (KeyError, ValueError, TypeError, AttributeError) as ex:
                print("Failed parsing used pods command response for cleanup: {}".format(ex))
                return -1, set([])

            return current_pod_count, namespaces
        except Exception as ex:
            print('Failed obtaining used pods information: {}'.format(ex))
            return -2, set([])

    def _is_same_tenant(self, task_session):
        if not task_session or task_session is self._session:
            return True
        # noinspection PyStatementEffect
        try:
            tenant = self._session.get_decoded_token(self._session.token, verify=False)["tenant"]
            task_tenant = task_session.get_decoded_token(task_session.token, verify=False)["tenant"]
            return tenant == task_tenant
        except Exception as ex:
            print("ERROR: Failed getting tenant for task session: {}".format(ex))

    def run_one_task(self, queue: Text, task_id: Text, worker_args=None, task_session=None, **_):
        print('Pulling task {} launching on kubernetes cluster'.format(task_id))
        session = task_session or self._session
        task_data = session.api_client.tasks.get_all(id=[task_id])[0]

        # push task into the k8s queue, so we have visibility on pending tasks in the k8s scheduler
        if self._is_same_tenant(task_session):
            try:
                print('Pushing task {} into temporary pending queue'.format(task_id))
                _ = session.api_client.tasks.stop(task_id, force=True)

                res = self._session.api_client.tasks.enqueue(
                    task_id,
                    queue=self.k8s_pending_queue_id,
                    status_reason='k8s pending scheduler',
                )
                if res.meta.result_code != 200:
                    raise Exception(res.meta.result_msg)
            except Exception as e:
                self.log.error("ERROR: Could not push back task [{}] to k8s pending queue {} [{}], error: {}".format(
                    task_id, self.k8s_pending_queue_name, self.k8s_pending_queue_id, e))
                return

        container = get_task_container(session, task_id)
        if not container.get('image'):
            container['image'] = str(
                ENV_DOCKER_IMAGE.get() or session.config.get("agent.default_docker.image", "nvidia/cuda")
            )
            container['arguments'] = session.config.get("agent.default_docker.arguments", None)
            set_task_container(
                session, task_id, docker_image=container['image'], docker_arguments=container['arguments']
            )

        # get the clearml.conf encoded file, make sure we use system packages!

        git_user = ENV_AGENT_GIT_USER.get() or self._session.config.get("agent.git_user", None)
        git_pass = ENV_AGENT_GIT_PASS.get() or self._session.config.get("agent.git_pass", None)
        extra_config_values = [
            'agent.package_manager.system_site_packages: true' if self._force_system_site_packages else '',
            'agent.git_user: "{}"'.format(git_user) if git_user else '',
            'agent.git_pass: "{}"'.format(git_pass) if git_pass else '',
        ]

        # noinspection PyProtectedMember
        config_content = (
            self.conf_file_content or Path(session._config_file).read_text() or ""
        ) + '\n{}\n'.format('\n'.join(x for x in extra_config_values if x))

        hocon_config_encoded = config_content.encode("ascii")

        create_clearml_conf = ["echo '{}' | base64 --decode >> ~/clearml.conf".format(
            base64.b64encode(
                hocon_config_encoded
            ).decode('ascii')
        )]

        if task_session:
            create_clearml_conf.append(
                "export CLEARML_AUTH_TOKEN=$(echo '{}' | base64 --decode)".format(
                    base64.b64encode(task_session.token.encode("ascii")).decode('ascii')
                )
            )

        if self.ports_mode:
            print("Kubernetes looking for available pod to use")

        # noinspection PyBroadException
        try:
            queue_name = self._session.api_client.queues.get_by_id(queue=queue).name
        except Exception:
            queue_name = 'k8s'

        # Search for a free pod number
        pod_count = 0
        pod_number = self.base_pod_num
        while self.ports_mode or self.max_pods_limit:
            pod_number = self.base_pod_num + pod_count

            kubectl_cmd_new = self.get_kubectl_command(
                "get pods",
                extra_labels=[self.limit_pod_label.format(pod_number=pod_number)] if self.ports_mode else None
            )
            self.log.debug("Looking for a free pod/port: {}".format(kubectl_cmd_new))
            process = subprocess.Popen(kubectl_cmd_new.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            output = stringify_bash_output(output)
            error = stringify_bash_output(error)

            try:
                items_count = len(json.loads(output).get("items", []))
            except (ValueError, TypeError) as ex:
                self.log.warning(
                    "K8S Glue pods monitor: Failed parsing kubectl output:\n{}\ntask '{}' "
                    "will be enqueued back to queue '{}'\nEx: {}".format(
                        output, task_id, queue, ex
                    )
                )
                session.api_client.tasks.stop(task_id, force=True)
                # noinspection PyBroadException
                try:
                    self._session.api_client.tasks.enqueue(task_id, queue=queue, status_reason='kubectl parsing error')
                except:
                    self.log.warning("Failed enqueuing task to queue '{}'".format(queue))
                return

            if not items_count:
                # No such pod exist so we can use the pod_number we found (result exists but with no items)
                break

            if self.max_pods_limit:
                current_pod_count = items_count
                max_count = self.max_pods_limit
            else:
                current_pod_count = pod_count
                max_count = self.num_of_services - 1

            if current_pod_count >= max_count:
                # All pods are taken, exit
                self.log.debug(
                    "kubectl last result: {}\n{}".format(error, output))
                self.log.warning(
                    "All k8s services are in use, task '{}' "
                    "will be enqueued back to queue '{}'".format(
                        task_id, queue
                    )
                )
                session.api_client.tasks.stop(task_id, force=True)
                # noinspection PyBroadException
                try:
                    self._session.api_client.tasks.enqueue(
                        task_id, queue=queue, status_reason='k8s max pod limit (no free k8s service)'
                    )
                except:
                    self.log.warning("Failed enqueuing task to queue '{}'".format(queue))
                return
            elif self.max_pods_limit:
                # max pods limit hasn't reached yet, so we can create the pod
                break
            pod_count += 1

        labels = self._get_pod_labels(queue, queue_name)
        if self.ports_mode:
            labels.append(self.limit_pod_label.format(pod_number=pod_number))

        if self.ports_mode:
            print("Kubernetes scheduling task id={} on pod={} (pod_count={})".format(task_id, pod_number, pod_count))
        else:
            print("Kubernetes scheduling task id={}".format(task_id))

        try:
            template = self._resolve_template(task_session, task_data, queue)
        except Exception as ex:
            print("ERROR: Failed resolving template (skipping): {}".format(ex))
            return

        try:
            namespace = template['metadata']['namespace'] or self.namespace
        except (KeyError, TypeError, AttributeError):
            namespace = self.namespace

        if template:
            output, error = self._kubectl_apply(
                template=template,
                pod_number=pod_number,
                create_clearml_conf=create_clearml_conf,
                labels=labels,
                docker_image=container['image'],
                docker_args=container['arguments'],
                docker_bash=container.get('setup_shell_script'),
                task_id=task_id,
                queue=queue,
                namespace=namespace,
            )

        print('kubectl output:\n{}\n{}'.format(error, output))
        if error:
            send_log = "Running kubectl encountered an error: {}".format(error)
            self.log.error(send_log)
            self.send_logs(task_id, send_log.splitlines())

        user_props = {"k8s-queue": str(queue_name)}
        if self.ports_mode:
            user_props.update(
                {
                    "k8s-pod-number": pod_number,
                    "k8s-pod-label": labels[0],
                    "k8s-internal-pod-count": pod_count,
                }
            )

        if self._user_props_cb:
            # noinspection PyBroadException
            try:
                custom_props = self._user_props_cb(pod_number) if self.ports_mode else self._user_props_cb()
                user_props.update(custom_props)
            except Exception:
                pass

        if user_props:
            self._set_task_user_properties(
                task_id=task_id,
                task_session=task_session,
                **user_props
            )

    def _get_pod_labels(self, queue, queue_name):
        return [
            self._get_agent_label(),
            "clearml-agent-queue={}".format(self._safe_k8s_label_value(queue)),
            "clearml-agent-queue-name={}".format(self._safe_k8s_label_value(queue_name))
        ]

    def _get_docker_args(self, docker_args, flags, target=None, convert=None):
        # type: (List[str], Collection[str], Optional[str], Callable[[str], Any]) -> Union[dict, List[str]]
        """
        Get docker args matching specific flags.

        :argument docker_args: List of docker argument strings (flags and values)
        :argument flags: List of flags/names to intercept (e.g. "--env" etc.)
        :argument target: Controls return format. If provided, returns a dict with a target field containing a list
         of result strings, otherwise returns a list of result strings
        :argument convert: Optional conversion function for each result string
        """
        args = docker_args[:] if docker_args else []
        results = []
        while args:
            cmd = args.pop(0).strip()
            if cmd in flags:
                env = args.pop(0).strip()
                if convert:
                    env = convert(env)
                results.append(env)
            else:
                self.log.warning('skipping docker argument {} (only -e --env supported)'.format(cmd))
        if target:
            return {target: results} if results else {}
        return results

    def _kubectl_apply(
        self,
        create_clearml_conf,
        docker_image,
        docker_args,
        docker_bash,
        labels,
        queue,
        task_id,
        namespace,
        template=None,
        pod_number=None
    ):
        template.setdefault('apiVersion', 'v1')
        template['kind'] = 'Pod'
        template.setdefault('metadata', {})
        name = self.pod_name_prefix + str(task_id)
        template['metadata']['name'] = name
        template.setdefault('spec', {})
        template['spec'].setdefault('containers', [])
        template['spec'].setdefault('restartPolicy', 'Never')
        if labels:
            labels_dict = dict(pair.split('=', 1) for pair in labels)
            template['metadata'].setdefault('labels', {})
            template['metadata']['labels'].update(labels_dict)

        container = self._get_docker_args(
            docker_args,
            target="env",
            flags={"-e", "--env"},
            convert=lambda env: {'name': env.partition("=")[0], 'value': env.partition("=")[2]},
        )

        container_bash_script = [self.container_bash_script] if isinstance(self.container_bash_script, str) \
            else self.container_bash_script

        extra_docker_bash_script = '\n'.join(self._session.config.get("agent.extra_docker_shell_script", None) or [])
        if docker_bash:
            extra_docker_bash_script += '\n' + str(docker_bash) + '\n'

        script_encoded = '\n'.join(
            ['#!/bin/bash', ] +
            [line.format(extra_bash_init_cmd=self.extra_bash_init_script or '',
                         task_id=task_id,
                         extra_docker_bash_script=extra_docker_bash_script,
                         default_execution_agent_args=self.DEFAULT_EXECUTION_AGENT_ARGS,
                         agent_install_args=self.POD_AGENT_INSTALL_ARGS)
             for line in container_bash_script])

        extra_bash_commands = list(create_clearml_conf or [])

        start_agent_script_path = ENV_START_AGENT_SCRIPT_PATH.get() or "~/__start_agent__.sh"

        extra_bash_commands.append(
            "echo '{content}' | base64 --decode >> {script_path} ; /bin/bash {script_path}".format(
                content=base64.b64encode(
                    script_encoded.encode('ascii')
                ).decode('ascii'),
                script_path=start_agent_script_path
            )
        )

        # Notice: we always leave with exit code 0, so pods are never restarted
        container = self._merge_containers(
            container,
            dict(name=name, image=docker_image,
                 command=['/bin/bash'],
                 args=['-c', '{} ; exit 0'.format(' ; '.join(extra_bash_commands))])
        )

        if template['spec']['containers']:
            template['spec']['containers'][0] = self._merge_containers(template['spec']['containers'][0], container)
        else:
            template['spec']['containers'].append(container)

        if self._docker_force_pull:
            for c in template['spec']['containers']:
                c.setdefault('imagePullPolicy', 'Always')

        fp, yaml_file = tempfile.mkstemp(prefix='clearml_k8stmpl_', suffix='.yml')
        os.close(fp)
        with open(yaml_file, 'wt') as f:
            yaml.dump(template, f)

        self.log.debug("Applying template:\n{}".format(pformat(template, indent=2)))

        kubectl_cmd = self.KUBECTL_APPLY_CMD.format(
            task_id=task_id,
            docker_image=docker_image,
            queue_id=queue,
            namespace=namespace
        )
        # make sure we provide a list
        if isinstance(kubectl_cmd, str):
            kubectl_cmd = kubectl_cmd.split()

        # add the template file at the end
        kubectl_cmd += [yaml_file]
        try:
            process = subprocess.Popen(kubectl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
        except Exception as ex:
            return None, str(ex)
        finally:
            safe_remove_file(yaml_file)

        return stringify_bash_output(output), stringify_bash_output(error)

    def _cleanup_old_pods(self, namespaces, extra_msg=None):
        # type: (Iterable[str], Optional[str]) -> Dict[str, List[str]]
        self.log.debug("Cleaning up pods")
        deleted_pods = defaultdict(list)
        for namespace in namespaces:
            if time() - self._last_pod_cleanup_per_ns[namespace] < self._min_cleanup_interval_per_ns_sec:
                # Do not try to cleanup the same namespace too quickly
                continue
            kubectl_cmd = self.KUBECTL_DELETE_CMD.format(namespace=namespace, agent_label=self._get_agent_label())
            self.log.debug("Deleting old/failed pods{} for ns {}: {}".format(
                extra_msg or "", namespace, kubectl_cmd
            ))
            try:
                res = get_bash_output(kubectl_cmd, raise_error=True)
                lines = [
                    line for line in
                    (r.strip().rpartition("/")[-1] for r in res.splitlines())
                    if line.startswith(self.pod_name_prefix)
                ]
                self.log.debug(" - deleted pod(s) %s", ", ".join(lines))
                deleted_pods[namespace].extend(lines)
            except Exception as ex:
                self.log.error("Failed deleting old/failed pods for ns %s: %s", namespace, str(ex))
            finally:
                self._last_pod_cleanup_per_ns[namespace] = time()
        return deleted_pods

    def run_tasks_loop(self, queues: List[Text], worker_params, **kwargs):
        """
        :summary: Pull and run tasks from queues.
        :description: 1. Go through ``queues`` by order.
                      2. Try getting the next task for each and run the first one that returns.
                      3. Go to step 1
        :param queues: IDs of queues to pull tasks from
        :type queues: list of ``Text``
        :param worker_params: Worker command line arguments
        :type worker_params: ``clearml_agent.helper.process.WorkerParams``
        """
        events_service = self.get_service(Events)

        # make sure we have a k8s pending queue
        if not self.k8s_pending_queue_id:
            resolved_ids = self._resolve_queue_names([self.k8s_pending_queue_name], create_if_missing=True)
            if not resolved_ids:
                raise ValueError(
                    "Failed resolving or creating k8s pending queue {}".format(self.k8s_pending_queue_name)
                )
            self.k8s_pending_queue_id = resolved_ids[0]

        _last_machine_update_ts = 0
        while True:
            # Get used pods and namespaces
            current_pods, namespaces = self._get_used_pods()

            # just in case there are no pods, make sure we look at our base namespace
            namespaces.add(self.namespace)

            # check if have pod limit, then check if we hit it.
            if self.max_pods_limit:
                if current_pods >= self.max_pods_limit:
                    print("Maximum pod limit reached {}/{}, sleeping for {:.1f} seconds".format(
                        current_pods, self.max_pods_limit, self._polling_interval))
                    # delete old completed / failed pods
                    self._cleanup_old_pods(namespaces, " due to pod limit")
                    # go to sleep
                    sleep(self._polling_interval)
                    continue

            # iterate over queues (priority style, queues[0] is highest)
            for queue in queues:
                # delete old completed / failed pods
                self._cleanup_old_pods(namespaces)

                # get next task in queue
                try:
                    response = self._get_next_task(queue=queue, get_task_info=self._impersonate_as_task_owner)
                except Exception as e:
                    print("Warning: Could not access task queue [{}], error: {}".format(queue, e))
                    continue
                else:
                    if not response:
                        continue
                    try:
                        task_id = response["entry"]["task"]
                    except (KeyError, TypeError, AttributeError):
                        print("No tasks in queue {}".format(queue))
                        continue

                    task_session = None
                    if self._impersonate_as_task_owner:
                        try:
                            task_user = response["task_info"]["user"]
                            task_company = response["task_info"]["company"]
                        except (KeyError, TypeError, AttributeError):
                            print("Error: cannot retrieve owner user for the task '{}', skipping".format(task_id))
                            continue

                        task_session = self.get_task_session(task_user, task_company)
                        if not task_session:
                            print(
                                "Error: Could not login as the user '{}' for the task '{}', skipping".format(
                                    task_user, task_id
                                )
                            )
                            continue

                    events_service.send_log_events(
                        self.worker_id,
                        task_id=task_id,
                        lines="task {} pulled from {} by worker {}".format(
                            task_id, queue, self.worker_id
                        ),
                        level="INFO",
                        session=task_session,
                    )

                    self.report_monitor(ResourceMonitor.StatusReport(queues=queues, queue=queue, task=task_id))
                    self.run_one_task(queue, task_id, worker_params, task_session)
                    self.report_monitor(ResourceMonitor.StatusReport(queues=self.queues))
                    break
            else:
                # sleep and retry polling
                print("No tasks in Queues, sleeping for {:.1f} seconds".format(self._polling_interval))
                sleep(self._polling_interval)

            if self._session.config["agent.reload_config"]:
                self.reload_config()

    def k8s_daemon(self, queue, **kwargs):
        """
        Start the k8s Glue service.
        This service will be pulling tasks from *queue* and scheduling them for execution using kubectl.
        Notice all scheduled tasks are pushed back into K8S_PENDING_QUEUE,
        and popped when execution actually starts. This creates full visibility into the k8s scheduler.
        Manually popping a task from the K8S_PENDING_QUEUE,
        will cause the k8s scheduler to skip the execution once the scheduled tasks needs to be executed

        :param list(str) queue: queue name to pull from
        """
        return self.daemon(
            queues=[ObjectID(name=queue)] if queue else None,
            log_level=logging.INFO, foreground=True, docker=False, **kwargs,
        )

    def _get_next_task(self, queue, get_task_info):
        return get_next_task(
            self._session, queue=queue, get_task_info=get_task_info
        )

    def _resolve_template(self, task_session, task_data, queue):
        if self.template_dict:
            return deepcopy(self.template_dict)

    @classmethod
    def get_ssh_server_bash(cls, ssh_port_number):
        return ' ; '.join(line.format(port=ssh_port_number) for line in cls.BASH_INSTALL_SSH_CMD)

    @staticmethod
    def _merge_containers(c1, c2):
        def merge_env(k, d1, d2, not_set):
            if k != "env":
                return not_set
            # Merge environment lists, second list overrides first
            return list({
                item['name']: item for envs in (d1, d2) for item in envs
            }.values())

        return merge_dicts(
            c1, c2, custom_merge_func=merge_env
        )

    @staticmethod
    def _safe_k8s_label_value(value):
        """ Conform string to k8s standards for a label value """
        value = value.lower().strip()
        value = re.sub(r'^[^A-Za-z0-9]+', '', value)  # strip leading non-alphanumeric chars
        value = re.sub(r'[^A-Za-z0-9]+$', '', value)  # strip trailing non-alphanumeric chars
        value = re.sub(r'\W+', '-', value)  # allow only word chars (this removed "." which is supported, but nvm)
        value = re.sub(r'_+', '-', value)  # "_" is not allowed as well
        value = re.sub(r'-+', '-', value)  # don't leave messy "--" after replacing previous chars
        return value[:63]
