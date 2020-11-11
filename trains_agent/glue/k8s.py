from __future__ import print_function, division, unicode_literals

import base64
import logging
import os
import subprocess
import tempfile
from copy import deepcopy

import yaml
import json
from time import sleep
from typing import Text, List

from pyhocon import HOCONConverter

from trains_agent.commands.events import Events
from trains_agent.commands.worker import Worker
from trains_agent.errors import APIError
from trains_agent.helper.base import safe_remove_file
from trains_agent.helper.dicts import merge_dicts
from trains_agent.helper.process import get_bash_output
from trains_agent.helper.resource_monitor import ResourceMonitor
from trains_agent.interface.base import ObjectID


class K8sIntegration(Worker):
    K8S_PENDING_QUEUE = "k8s_scheduler"

    KUBECTL_APPLY_CMD = "kubectl apply -f"

    KUBECTL_RUN_CMD = "kubectl run trains-id-{task_id} " \
                      "--image {docker_image} " \
                      "--restart=Never --replicas=1 " \
                      "--generator=run-pod/v1 " \
                      "--namespace=trains"

    KUBECTL_DELETE_CMD = "kubectl delete pods " \
                         "--selector=TRAINS=agent " \
                         "--field-selector=status.phase!=Pending,status.phase!=Running " \
                         "--namespace=trains"

    BASH_INSTALL_SSH_CMD = \
        "apt-get install -y openssh-server ; " \
        "mkdir -p /var/run/sshd ; " \
        "echo 'root:training' | chpasswd ; " \
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config ; " \
        "sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config ; " \
        "sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd ; " \
        "echo 'AcceptEnv TRAINS_API_ACCESS_KEY TRAINS_API_SECRET_KEY' >> /etc/ssh/sshd_config ; " \
        'echo "export VISIBLE=now" >> /etc/profile ; ' \
        "/usr/sbin/sshd -p {port} & ; "

    CONTAINER_BASH_SCRIPT = \
        "export DEBIAN_FRONTEND='noninteractive'; " \
        "echo 'Binary::apt::APT::Keep-Downloaded-Packages \"true\";' > /etc/apt/apt.conf.d/docker-clean ; " \
        "chown -R root /root/.cache/pip ; " \
        "apt-get update ; " \
        "apt-get install -y git libsm6 libxext6 libxrender-dev libglib2.0-0 ; " \
        "declare LOCAL_PYTHON ; " \
        "for i in {{10..5}}; do which python3.$i && " \
        "python3.$i -m pip --version && " \
        "export LOCAL_PYTHON=$(which python3.$i) && break ; done ; " \
        "[ ! -z $LOCAL_PYTHON ] || apt-get install -y python3-pip ; " \
        "[ ! -z $LOCAL_PYTHON ] || export LOCAL_PYTHON=python3 ; " \
        "$LOCAL_PYTHON -m pip install trains-agent ; " \
        "{extra_bash_init_cmd} ; " \
        "$LOCAL_PYTHON -m trains_agent execute --full-monitoring --require-queue --id {task_id} ; "

    AGENT_LABEL = "TRAINS=agent"
    LIMIT_POD_LABEL = "ai.allegro.agent.serial=pod-{pod_number}"

    _edit_hyperparams_version = "2.9"

    def __init__(
            self,
            k8s_pending_queue_name=None,
            kubectl_cmd=None,
            container_bash_script=None,
            debug=False,
            ports_mode=False,
            num_of_services=20,
            user_props_cb=None,
            overrides_yaml=None,
            template_yaml=None,
            trains_conf_file=None,
            extra_bash_init_script=None,
    ):
        """
        Initialize the k8s integration glue layer daemon

        :param str k8s_pending_queue_name: queue name to use when task is pending in the k8s scheduler
        :param str|callable kubectl_cmd: kubectl command line str, supports formatting (default: KUBECTL_RUN_CMD)
            example: "task={task_id} image={docker_image} queue_id={queue_id}"
            or a callable function: kubectl_cmd(task_id, docker_image, queue_id, task_data)
        :param str container_bash_script: container bash script to be executed in k8s (default: CONTAINER_BASH_SCRIPT)
            Notice this string will use format() call, if you have curly brackets they should be doubled { -> {{
            Format arguments passed: {task_id} and {extra_bash_init_cmd}
        :param bool debug: Switch logging on
        :param bool ports_mode: Adds a label to each pod which can be used in services in order to expose ports.
            Requires the `num_of_services` parameter.
        :param int num_of_services: Number of k8s services configured in the cluster. Required if `port_mode` is True.
            (default: 20)
        :param callable user_props_cb: An Optional callable allowing additional user properties to be specified
            when scheduling a task to run in a pod. Callable can receive an optional pod number and should return
            a dictionary of user properties (name and value). Signature is [[Optional[int]], Dict[str,str]]
        :param str overrides_yaml: YAML file containing the overrides for the pod (optional)
        :param str template_yaml: YAML file containing the template  for the pod (optional).
            If provided the pod is scheduled with kubectl apply and overrides are ignored, otherwise with kubectl run.
        :param str trains_conf_file: trains.conf file to be use by the pod itself (optional)
        :param str extra_bash_init_script: Additional bash script to run before starting the Task inside the container
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
        self._edit_hyperparams_support = None
        self._user_props_cb = user_props_cb
        self.trains_conf_file = None
        self.overrides_json_string = None
        self.template_dict = None
        self.extra_bash_init_script = extra_bash_init_script or None
        if self.extra_bash_init_script and not isinstance(self.extra_bash_init_script, str):
            self.extra_bash_init_script = ' ; '.join(self.extra_bash_init_script)
        self.pod_limits = []
        self.pod_requests = []
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

        if trains_conf_file:
            with open(os.path.expandvars(os.path.expanduser(str(trains_conf_file))), 'rt') as f:
                self.trains_conf_file = f.read()

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

    def run_one_task(self, queue: Text, task_id: Text, worker_args=None, **_):
        print('Pulling task {} launching on kubernetes cluster'.format(task_id))
        task_data = self._session.api_client.tasks.get_all(id=[task_id])[0]

        # push task into the k8s queue, so we have visibility on pending tasks in the k8s scheduler
        try:
            print('Pushing task {} into temporary pending queue'.format(task_id))
            self._session.api_client.tasks.reset(task_id)
            self._session.api_client.tasks.enqueue(task_id, queue=self.k8s_pending_queue_name,
                                                   status_reason='k8s pending scheduler')
        except Exception as e:
            self.log.error("ERROR: Could not push back task [{}] to k8s pending queue [{}], error: {}".format(
                task_id, self.k8s_pending_queue_name, e))
            return

        if task_data.execution.docker_cmd:
            docker_parts = task_data.execution.docker_cmd
        else:
            docker_parts = str(os.environ.get("TRAINS_DOCKER_IMAGE") or
                               self._session.config.get("agent.default_docker.image", "nvidia/cuda"))

        # take the first part, this is the docker image name (not arguments)
        docker_parts = docker_parts.split()
        docker_image = docker_parts[0]
        docker_args = docker_parts[1:] if len(docker_parts) > 1 else []

        # get the trains.conf encoded file
        # noinspection PyProtectedMember
        hocon_config_encoded = (self.trains_conf_file or self._session._config_file).encode('ascii')
        create_trains_conf = "echo '{}' | base64 --decode >> ~/trains.conf && ".format(
            base64.b64encode(
                hocon_config_encoded
            ).decode('ascii')
        )

        if self.ports_mode:
            print("Kubernetes looking for available pod to use")

        # Search for a free pod number
        pod_number = 1
        while self.ports_mode:
            kubectl_cmd_new = "kubectl get pods -l {pod_label},{agent_label} -n trains".format(
                pod_label=self.LIMIT_POD_LABEL.format(pod_number=pod_number),
                agent_label=self.AGENT_LABEL
            )
            process = subprocess.Popen(kubectl_cmd_new.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            output = '' if not output else output if isinstance(output, str) else output.decode('utf-8')
            error = '' if not error else error if isinstance(error, str) else error.decode('utf-8')

            if not output:
                # No such pod exist so we can use the pod_number we found
                break
            if pod_number >= self.num_of_services:
                # All pod numbers are taken, exit
                self.log.warning(
                    "kubectl last result: {}\n{}\nAll k8s services are in use, task '{}' "
                    "will be enqueued back to queue '{}'".format(
                        error,output, task_id, queue
                    )
                )
                self._session.api_client.tasks.reset(task_id)
                self._session.api_client.tasks.enqueue(task_id, queue=queue)
                return
            pod_number += 1

        labels = ([self.LIMIT_POD_LABEL.format(pod_number=pod_number)] if self.ports_mode else []) + [self.AGENT_LABEL]

        if self.ports_mode:
            print("Kubernetes scheduling task id={} on pod={}".format(task_id, pod_number))
        else:
            print("Kubernetes scheduling task id={}".format(task_id))

        if self.template_dict:
            output, error = self._kubectl_apply(
                create_trains_conf=create_trains_conf,
                labels=labels, docker_image=docker_image, docker_args=docker_args,
                task_id=task_id, queue=queue)
        else:
            output, error = self._kubectl_run(
                create_trains_conf=create_trains_conf,
                labels=labels, docker_image=docker_image,
                task_data=task_data,
                task_id=task_id, queue=queue)

        error = '' if not error else (error if isinstance(error, str) else error.decode('utf-8'))
        output = '' if not output else (output if isinstance(output, str) else output.decode('utf-8'))
        print('kubectl output:\n{}\n{}'.format(error, output))

        if error:
            self.log.error("Running kubectl encountered an error: {}".format(error))
        elif self.ports_mode:
            user_props = {"k8s-pod-number": pod_number, "k8s-pod-label": labels[0]}
            if self._user_props_cb:
                # noinspection PyBroadException
                try:
                    custom_props = self._user_props_cb(pod_number) if self.ports_mode else self._user_props_cb()
                    user_props.update(custom_props)
                except Exception:
                    pass
            self._set_task_user_properties(
                task_id=task_id,
                **user_props
            )

    def _parse_docker_args(self, docker_args):
        # type: (list) -> dict
        kube_args = {'env': []}
        while docker_args:
            cmd = docker_args.pop().strip()
            if cmd in ('-e', '--env',):
                env = docker_args.pop().strip()
                key, value = env.split('=', 1)
                kube_args[key] += {key: value}
            else:
                self.log.warning('skipping docker argument {} (only -e --env supported)'.format(cmd))
        return kube_args

    def _kubectl_apply(self, create_trains_conf, docker_image, docker_args, labels, queue, task_id):
        template = deepcopy(self.template_dict)
        template.setdefault('apiVersion', 'v1')
        template['kind'] = 'Pod'
        template.setdefault('metadata', {})
        name = 'trains-id-{task_id}'.format(task_id=task_id)
        template['metadata']['name'] = name
        template.setdefault('spec', {})
        template['spec'].setdefault('containers', [])
        if labels:
            labels_dict = dict(pair.split('=', 1) for pair in labels)
            template['metadata'].setdefault('labels', {})
            template['metadata']['labels'].update(labels_dict)
        container = self._parse_docker_args(docker_args)
        container = merge_dicts(
            container,
            dict(name=name, image=docker_image,
                 command=['/bin/sh'],
                 args=['-c', create_trains_conf + self.container_bash_script.format(
                     extra_bash_init_cmd=self.extra_bash_init_script,
                     task_id=task_id)]))

        if template['spec']['containers']:
            template['spec']['containers'][0] = merge_dicts(template['spec']['containers'][0], container)
        else:
            template['spec']['containers'].append(container)

        fp, yaml_file = tempfile.mkstemp(prefix='trains_k8stmpl_', suffix='.yml')
        os.close(fp)
        with open(yaml_file, 'wt') as f:
            yaml.dump(template, f)

        kubectl_cmd = self.KUBECTL_APPLY_CMD.format(
            task_id=task_id,
            docker_image=docker_image,
            queue_id=queue,
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

    def _kubectl_run(self, create_trains_conf, docker_image, labels, queue, task_data, task_id):
        if callable(self.kubectl_cmd):
            kubectl_cmd = self.kubectl_cmd(task_id, docker_image, queue, task_data)
        else:
            kubectl_cmd = self.kubectl_cmd.format(
                task_id=task_id,
                docker_image=docker_image,
                queue_id=queue
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

        kubectl_cmd += [
            "--labels=" + ",".join(labels),
            "--command",
            "--",
            "/bin/sh",
            "-c",
            create_trains_conf + self.container_bash_script.format(
                extra_bash_init_cmd=self.extra_bash_init_script, task_id=task_id),
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
        :type worker_params: ``trains_agent.helper.process.WorkerParams``
        """
        events_service = self.get_service(Events)

        # make sure we have a k8s pending queue
        try:
            self._session.api_client.queues.create(self.k8s_pending_queue_name)
        except Exception:
            pass
        # get queue id
        self.k8s_pending_queue_name = self._resolve_name(self.k8s_pending_queue_name, "queues")

        _last_machine_update_ts = 0
        while True:
            # iterate over queues (priority style, queues[0] is highest)
            for queue in queues:
                # delete old completed / failed pods
                get_bash_output(self.KUBECTL_DELETE_CMD)

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
        return self.daemon(queues=[ObjectID(name=queue)] if queue else None, log_level=logging.INFO, foreground=True, docker=False)

    @classmethod
    def get_ssh_server_bash(cls, ssh_port_number):
        return cls.BASH_INSTALL_SSH_CMD.format(port=ssh_port_number)
