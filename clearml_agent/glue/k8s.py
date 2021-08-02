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
from copy import deepcopy
from pathlib import Path
from threading import Thread
from time import sleep
from typing import Text, List, Callable, Any, Collection, Optional, Union

import yaml

from clearml_agent.commands.events import Events
from clearml_agent.commands.worker import Worker, get_task_container, set_task_container
from clearml_agent.definitions import ENV_DOCKER_IMAGE
from clearml_agent.errors import APIError
from clearml_agent.helper.base import safe_remove_file
from clearml_agent.helper.dicts import merge_dicts
from clearml_agent.helper.process import get_bash_output
from clearml_agent.helper.resource_monitor import ResourceMonitor
from clearml_agent.interface.base import ObjectID


class K8sIntegration(Worker):
    K8S_PENDING_QUEUE = "k8s_scheduler"

    K8S_DEFAULT_NAMESPACE = "clearml"
    AGENT_LABEL = "CLEARML=agent"
    LIMIT_POD_LABEL = "ai.allegro.agent.serial=pod-{pod_number}"

    KUBECTL_APPLY_CMD = "kubectl apply --namespace={namespace} -f"

    KUBECTL_RUN_CMD = "kubectl run clearml-id-{task_id} " \
                      "--image {docker_image} {docker_args} " \
                      "--restart=Never " \
                      "--namespace={namespace}"

    KUBECTL_DELETE_CMD = "kubectl delete pods " \
                         "--selector={selector} " \
                         "--field-selector=status.phase!=Pending,status.phase!=Running " \
                         "--namespace={namespace}"

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

    CONTAINER_BASH_SCRIPT = [
        "export DEBIAN_FRONTEND='noninteractive'",
        "echo 'Binary::apt::APT::Keep-Downloaded-Packages \"true\";' > /etc/apt/apt.conf.d/docker-clean",
        "chown -R root /root/.cache/pip",
        "apt-get update",
        "apt-get install -y git libsm6 libxext6 libxrender-dev libglib2.0-0",
        "declare LOCAL_PYTHON",
        "for i in {{10..5}}; do which python3.$i && python3.$i -m pip --version && "
        "export LOCAL_PYTHON=$(which python3.$i) && break ; done",
        "[ ! -z $LOCAL_PYTHON ] || apt-get install -y python3-pip",
        "[ ! -z $LOCAL_PYTHON ] || export LOCAL_PYTHON=python3",
        "$LOCAL_PYTHON -m pip install clearml-agent",
        "{extra_bash_init_cmd}",
        "{extra_docker_bash_script}",
        "$LOCAL_PYTHON -m clearml_agent execute --full-monitoring --require-queue --id {task_id}"
    ]

    _edit_hyperparams_version = "2.9"

    def __init__(
            self,
            k8s_pending_queue_name=None,
            kubectl_cmd=None,
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
            **kwargs
    ):
        """
        Initialize the k8s integration glue layer daemon

        :param str k8s_pending_queue_name: queue name to use when task is pending in the k8s scheduler
        :param str|callable kubectl_cmd: kubectl command line str, supports formatting (default: KUBECTL_RUN_CMD)
            example: "task={task_id} image={docker_image} queue_id={queue_id}"
            or a callable function: kubectl_cmd(task_id, docker_image, docker_args, queue_id, task_data)
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
        :param str template_yaml: YAML file containing the template  for the pod (optional).
            If provided the pod is scheduled with kubectl apply and overrides are ignored, otherwise with kubectl run.
        :param str clearml_conf_file: clearml.conf file to be use by the pod itself (optional)
        :param str extra_bash_init_script: Additional bash script to run before starting the Task inside the container
        :param str namespace: K8S namespace to be used when creating the new pods (default: clearml)
        :param int max_pods_limit: Maximum number of pods that K8S glue can run at the same time
        """
        super(K8sIntegration, self).__init__()
        self.k8s_pending_queue_name = k8s_pending_queue_name or self.K8S_PENDING_QUEUE
        self.kubectl_cmd = kubectl_cmd or self.KUBECTL_RUN_CMD
        self.container_bash_script = container_bash_script or self.CONTAINER_BASH_SCRIPT
        # Always do system packages, because by we will be running inside a docker
        self._session.config.put("agent.package_manager.system_site_packages", True)
        # Add debug logging
        if debug:
            self.log.logger.disabled = False
            self.log.logger.setLevel(logging.INFO)
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
        if overrides_yaml:
            with open(os.path.expandvars(os.path.expanduser(str(overrides_yaml))), 'rt') as f:
                overrides = yaml.load(f, Loader=getattr(yaml, 'FullLoader', None))
            if overrides:
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
        if template_yaml:
            with open(os.path.expandvars(os.path.expanduser(str(template_yaml))), 'rt') as f:
                self.template_dict = yaml.load(f, Loader=getattr(yaml, 'FullLoader', None))

        clearml_conf_file = clearml_conf_file or kwargs.get('trains_conf_file')

        if clearml_conf_file:
            with open(os.path.expandvars(os.path.expanduser(str(clearml_conf_file))), 'rt') as f:
                self.conf_file_content = f.read()
            # make sure we use system packages!
            self.conf_file_content += '\nagent.package_manager.system_site_packages=true\n'

        self._agent_label = None

        self._monitor_hanging_pods()

    def _monitor_hanging_pods(self):
        _check_pod_thread = Thread(target=self._monitor_hanging_pods_daemon)
        _check_pod_thread.daemon = True
        _check_pod_thread.start()

    @staticmethod
    def _get_path(d, *path, default=None):
        try:
            return functools.reduce(
                lambda a, b: a[b], path, d
            )
        except (IndexError, KeyError):
            return default

    def _monitor_hanging_pods_daemon(self):
        last_tasks_msgs = {}  # last msg updated for every task

        while True:
            output = get_bash_output('kubectl get pods -n {namespace} -o=JSON'.format(
                namespace=self.namespace
            ))
            output = '' if not output else output if isinstance(output, str) else output.decode('utf-8')
            try:
                output_config = json.loads(output)
            except Exception as ex:
                self.log.warning('K8S Glue pods monitor: Failed parsing kubectl output:\n{}\nEx: {}'.format(output, ex))
                sleep(self._polling_interval)
                continue
            pods = output_config.get('items', [])
            task_ids = set()
            for pod in pods:
                if self._get_path(pod, 'status', 'phase') != "Pending":
                    continue

                pod_name = pod.get('metadata', {}).get('name', None)
                if not pod_name:
                    continue

                task_id = pod_name.rpartition('-')[-1]
                if not task_id:
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
                        delete_pod_cmd = 'kubectl delete pods {} -n {}'.format(pod_name, self.namespace)
                        get_bash_output(delete_pod_cmd)
                        try:
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
                            method='get',
                            async_enable=False,
                        )
                        if not result.ok:
                            result_msg = self._get_path(result.json(), 'meta', 'result_msg')
                            raise Exception(result_msg or result.text)

                        # update last msg for this task
                        last_tasks_msgs[task_id] = msg
                    except Exception as ex:
                        self.log.warning(
                            'K8S Glue pods monitor: Failed setting status message for task "{}"\nEX: {}'.format(
                                task_id, ex
                            )
                        )

            # clean up any last message for a task that wasn't seen as a pod
            last_tasks_msgs = {k: v for k, v in last_tasks_msgs.items() if k in task_ids}

            sleep(self._polling_interval)

    def _set_task_user_properties(self, task_id: str, **properties: str):
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
            self._session.get(
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

    def _get_number_used_pods(self):
        # noinspection PyBroadException
        try:
            kubectl_cmd_new = "kubectl get pods -l {agent_label} -n {namespace} -o json".format(
                agent_label=self._get_agent_label(),
                namespace=self.namespace,
            )
            process = subprocess.Popen(kubectl_cmd_new.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            output = '' if not output else output if isinstance(output, str) else output.decode('utf-8')
            error = '' if not error else error if isinstance(error, str) else error.decode('utf-8')

            if not output:
                # No such pod exist so we can use the pod_number we found
                return 0

            try:
                current_pod_count = len(json.loads(output).get("items", []))
            except (ValueError, TypeError) as ex:
                return -1

            return current_pod_count
        except Exception as ex:
            print('Failed getting number of used pods: {}'.format(ex))
            return -2

    def run_one_task(self, queue: Text, task_id: Text, worker_args=None, **_):
        print('Pulling task {} launching on kubernetes cluster'.format(task_id))
        task_data = self._session.api_client.tasks.get_all(id=[task_id])[0]

        # push task into the k8s queue, so we have visibility on pending tasks in the k8s scheduler
        try:
            print('Pushing task {} into temporary pending queue'.format(task_id))
            res = self._session.api_client.tasks.stop(task_id, force=True)
            res = self._session.api_client.tasks.enqueue(
                task_id,
                queue=self.k8s_pending_queue_name,
                status_reason='k8s pending scheduler',
            )
            if res.meta.result_code != 200:
                raise Exception(res.meta.result_msg)
        except Exception as e:
            self.log.error("ERROR: Could not push back task [{}] to k8s pending queue [{}], error: {}".format(
                task_id, self.k8s_pending_queue_name, e))
            return

        container = get_task_container(self._session, task_id)
        if not container.get('image'):
            container['image'] = str(
                ENV_DOCKER_IMAGE.get() or self._session.config.get("agent.default_docker.image", "nvidia/cuda")
            )
            container['arguments'] = self._session.config.get("agent.default_docker.arguments", None)
            set_task_container(
                self._session, task_id, docker_image=container['image'], docker_arguments=container['arguments']
            )

        # get the clearml.conf encoded file
        # noinspection PyProtectedMember
        hocon_config_encoded = (
            self.conf_file_content
            or Path(self._session._config_file).read_text()
        ).encode("ascii")
        create_clearml_conf = "echo '{}' | base64 --decode >> ~/clearml.conf".format(
            base64.b64encode(
                hocon_config_encoded
            ).decode('ascii')
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
            if self.ports_mode:
                kubectl_cmd_new = "kubectl get pods -l {pod_label},{agent_label} -n {namespace}".format(
                    pod_label=self.LIMIT_POD_LABEL.format(pod_number=pod_number),
                    agent_label=self._get_agent_label(),
                    namespace=self.namespace,
                )
            else:
                kubectl_cmd_new = "kubectl get pods -l {agent_label} -n {namespace} -o json".format(
                    agent_label=self._get_agent_label(),
                    namespace=self.namespace,
                )
            process = subprocess.Popen(kubectl_cmd_new.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            output = '' if not output else output if isinstance(output, str) else output.decode('utf-8')
            error = '' if not error else error if isinstance(error, str) else error.decode('utf-8')

            if not output:
                # No such pod exist so we can use the pod_number we found
                break

            if self.max_pods_limit:
                try:
                    current_pod_count = len(json.loads(output).get("items", []))
                except (ValueError, TypeError) as ex:
                    self.log.warning(
                        "K8S Glue pods monitor: Failed parsing kubectl output:\n{}\ntask '{}' "
                        "will be enqueued back to queue '{}'\nEx: {}".format(
                            output, task_id, queue, ex
                        )
                    )
                    self._session.api_client.tasks.stop(task_id, force=True)
                    self._session.api_client.tasks.enqueue(task_id, queue=queue, status_reason='kubectl parsing error')
                    return
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
                self._session.api_client.tasks.stop(task_id, force=True)
                self._session.api_client.tasks.enqueue(
                    task_id, queue=queue, status_reason='k8s max pod limit (no free k8s service)')
                return
            elif self.max_pods_limit:
                # max pods limit hasn't reached yet, so we can create the pod
                break
            pod_count += 1

        labels = ([self.LIMIT_POD_LABEL.format(pod_number=pod_number)] if self.ports_mode else []) + \
                 [self._get_agent_label()]
        labels.append("clearml-agent-queue={}".format(self._safe_k8s_label_value(queue)))
        labels.append("clearml-agent-queue-name={}".format(self._safe_k8s_label_value(queue_name)))

        if self.ports_mode:
            print("Kubernetes scheduling task id={} on pod={} (pod_count={})".format(task_id, pod_number, pod_count))
        else:
            print("Kubernetes scheduling task id={}".format(task_id))

        kubectl_kwargs = dict(
            create_clearml_conf=create_clearml_conf,
            labels=labels,
            docker_image=container['image'],
            docker_args=container['arguments'],
            docker_bash=container.get('setup_shell_script'),
            task_id=task_id,
            queue=queue
        )

        if self.template_dict:
            output, error = self._kubectl_apply(**kubectl_kwargs)
        else:
            output, error = self._kubectl_run(task_data=task_data, **kubectl_kwargs)

        error = '' if not error else (error if isinstance(error, str) else error.decode('utf-8'))
        output = '' if not output else (output if isinstance(output, str) else output.decode('utf-8'))
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
                **user_props
            )

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

    def _kubectl_apply(self, create_clearml_conf, docker_image, docker_args, docker_bash, labels, queue, task_id):
        template = deepcopy(self.template_dict)
        template.setdefault('apiVersion', 'v1')
        template['kind'] = 'Pod'
        template.setdefault('metadata', {})
        name = 'clearml-id-{task_id}'.format(task_id=task_id)
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
                         extra_docker_bash_script=extra_docker_bash_script)
             for line in container_bash_script])

        create_init_script = \
            "echo '{}' | base64 --decode >> ~/__start_agent__.sh ; " \
            "/bin/bash ~/__start_agent__.sh".format(
                base64.b64encode(
                    script_encoded.encode('ascii')
                ).decode('ascii'))

        # Notice: we always leave with exit code 0, so pods are never restarted
        container = self._merge_containers(
            container,
            dict(name=name, image=docker_image,
                 command=['/bin/bash'],
                 args=['-c', '{} ; {} ; exit 0'.format(create_clearml_conf, create_init_script)])
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

        kubectl_cmd = self.KUBECTL_APPLY_CMD.format(
            task_id=task_id,
            docker_image=docker_image,
            queue_id=queue,
            namespace=self.namespace
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

        return output, error

    def _kubectl_run(
        self, create_clearml_conf, docker_image, docker_args, docker_bash, labels, queue, task_data, task_id
    ):
        if callable(self.kubectl_cmd):
            kubectl_cmd = self.kubectl_cmd(task_id, docker_image, docker_args, queue, task_data)
        else:
            kubectl_cmd = self.kubectl_cmd.format(
                task_id=task_id,
                docker_image=docker_image,
                docker_args=" ".join(self._get_docker_args(
                    docker_args, flags={"-e", "--env"}, convert=lambda env: '--env={}'.format(env))
                ),
                queue_id=queue,
                namespace=self.namespace,
            )
        # make sure we provide a list
        if isinstance(kubectl_cmd, str):
            kubectl_cmd = kubectl_cmd.split()

        if self.overrides_json_string:
            kubectl_cmd += ['--overrides=' + self.overrides_json_string]

        if self.pod_limits:
            kubectl_cmd += ['--limits', ",".join(self.pod_limits)]
        if self.pod_requests:
            kubectl_cmd += ['--requests', ",".join(self.pod_requests)]

        if self._docker_force_pull and not any(x.startswith("--image-pull-policy=") for x in kubectl_cmd):
            kubectl_cmd += ["--image-pull-policy='always'"]

        container_bash_script = [self.container_bash_script] if isinstance(self.container_bash_script, str) \
            else self.container_bash_script
        container_bash_script = ' ; '.join(container_bash_script)

        kubectl_cmd += [
            "--labels=" + ",".join(labels),
            "--command",
            "--",
            "/bin/sh",
            "-c",
            "{} ; {}".format(create_clearml_conf, container_bash_script.format(
                extra_bash_init_cmd=self.extra_bash_init_script or "",
                extra_docker_bash_script=docker_bash or "",
                task_id=task_id
            )),
        ]
        process = subprocess.Popen(kubectl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        return output, error

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
        # noinspection PyBroadException
        try:
            self._session.api_client.queues.create(self.k8s_pending_queue_name)
        except Exception:
            pass
        # get queue id
        self.k8s_pending_queue_name = self._resolve_name(self.k8s_pending_queue_name, "queues")

        _last_machine_update_ts = 0
        while True:
            # check if have pod limit, then check if we hit it.
            if self.max_pods_limit:
                current_pods = self._get_number_used_pods()
                if current_pods >= self.max_pods_limit:
                    print("Maximum pod limit reached {}/{}, sleeping for {:.1f} seconds".format(
                        current_pods, self.max_pods_limit, self._polling_interval))
                    # delete old completed / failed pods
                    get_bash_output(
                        self.KUBECTL_DELETE_CMD.format(namespace=self.namespace, selector=self._get_agent_label())
                    )
                    # go to sleep
                    sleep(self._polling_interval)
                    continue

            # iterate over queues (priority style, queues[0] is highest)
            for queue in queues:
                # delete old completed / failed pods
                get_bash_output(
                    self.KUBECTL_DELETE_CMD.format(namespace=self.namespace, selector=self._get_agent_label())
                )

                # get next task in queue
                try:
                    response = self._session.api_client.queues.get_next_task(queue=queue)
                except Exception as e:
                    print("Warning: Could not access task queue [{}], error: {}".format(queue, e))
                    continue
                else:
                    try:
                        task_id = response.entry.task
                    except AttributeError:
                        print("No tasks in queue {}".format(queue))
                        continue
                    events_service.send_log_events(
                        self.worker_id,
                        task_id=task_id,
                        lines="task {} pulled from {} by worker {}".format(
                            task_id, queue, self.worker_id
                        ),
                        level="INFO",
                    )

                    self.report_monitor(ResourceMonitor.StatusReport(queues=queues, queue=queue, task=task_id))
                    self.run_one_task(queue, task_id, worker_params)
                    self.report_monitor(ResourceMonitor.StatusReport(queues=self.queues))
                    break
            else:
                # sleep and retry polling
                print("No tasks in Queues, sleeping for {:.1f} seconds".format(self._polling_interval))
                sleep(self._polling_interval)

            if self._session.config["agent.reload_config"]:
                self.reload_config()

    def k8s_daemon(self, queue):
        """
        Start the k8s Glue service.
        This service will be pulling tasks from *queue* and scheduling them for execution using kubectl.
        Notice all scheduled tasks are pushed back into K8S_PENDING_QUEUE,
        and popped when execution actually starts. This creates full visibility into the k8s scheduler.
        Manually popping a task from the K8S_PENDING_QUEUE,
        will cause the k8s scheduler to skip the execution once the scheduled tasks needs to be executed

        :param list(str) queue: queue name to pull from
        """
        return self.daemon(queues=[ObjectID(name=queue)] if queue else None,
                           log_level=logging.INFO, foreground=True, docker=False)

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
        value = re.sub(r'-+', '-', value)  # don't leave messy "--" after replacing previous chars
        return value[:63]
