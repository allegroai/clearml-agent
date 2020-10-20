from __future__ import print_function, division, unicode_literals

import base64
import logging
import os
import subprocess
from time import sleep
from typing import Text, List

from pyhocon import HOCONConverter

from trains_agent.commands.events import Events
from trains_agent.commands.worker import Worker
from trains_agent.helper.process import get_bash_output
from trains_agent.helper.resource_monitor import ResourceMonitor
from trains_agent.interface.base import ObjectID


class K8sIntegration(Worker):
    K8S_PENDING_QUEUE = "k8s_scheduler"

    KUBECTL_RUN_CMD = "kubectl run trains-id-{task_id} " \
                      "--image {docker_image} " \
                      "--restart=Never --replicas=1 " \
                      "--generator=run-pod/v1 " \
                      "--namespace=trains"

    KUBECTL_DELETE_CMD = "kubectl delete pods " \
                         "--selector=TRAINS=agent " \
                         "--field-selector=status.phase!=Pending,status.phase!=Running " \
                         "--namespace=trains"

    CONTAINER_BASH_SCRIPT = \
        "export DEBIAN_FRONTEND='noninteractive'; " \
        "echo 'Binary::apt::APT::Keep-Downloaded-Packages \"true\";' > /etc/apt/apt.conf.d/docker-clean ; " \
        "chown -R root /root/.cache/pip ; " \
        "apt-get update ; " \
        "apt-get install -y git libsm6 libxext6 libxrender-dev libglib2.0-0 ; " \
        "(which python3 && python3 -m pip --version) || apt-get install -y python3-pip ; " \
        "python3 -m pip install trains-agent ; " \
        "python3 -m trains_agent execute --full-monitoring --require-queue --id {} ; "

    AGENT_LABEL = "TRAINS=agent"
    LIMIT_POD_LABEL = "ai.allegro.agent.serial=pod-{pod_number}"

    def __init__(
            self,
            k8s_pending_queue_name=None,
            kubectl_cmd=None,
            container_bash_script=None,
            debug=False,
            ports_mode=False,
            num_of_services=20,
    ):
        """
        Initialize the k8s integration glue layer daemon

        :param str k8s_pending_queue_name: queue name to use when task is pending in the k8s scheduler
        :param str|callable kubectl_cmd: kubectl command line str, supports formatting (default: KUBECTL_RUN_CMD)
            example: "task={task_id} image={docker_image} queue_id={queue_id}"
            or a callable function: kubectl_cmd(task_id, docker_image, queue_id, task_data)
        :param str container_bash_script: container bash script to be executed in k8s (default: CONTAINER_BASH_SCRIPT)
        :param bool debug: Switch logging on
        :param bool ports_mode: Adds a label to each pod which can be used in services in order to expose ports.
            Requires the `num_of_services` parameter.
        :param int num_of_services: Number of k8s services configured in the cluster. Required if `port_mode` is True.
            (default: 20)
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

    def run_one_task(self, queue: Text, task_id: Text, worker_args=None, **_):
        task_data = self._session.api_client.tasks.get_all(id=[task_id])[0]

        # push task into the k8s queue, so we have visibility on pending tasks in the k8s scheduler
        try:
            self._session.api_client.tasks.reset(task_id)
            self._session.api_client.tasks.enqueue(task_id, queue=self.k8s_pending_queue_name,
                                                   status_reason='k8s pending scheduler')
        except Exception as e:
            self.log.error("ERROR: Could not push back task [{}] to k8s pending queue [{}], error: {}".format(
                task_id, self.k8s_pending_queue_name, e))
            return

        if task_data.execution.docker_cmd:
            docker_image = task_data.execution.docker_cmd
        else:
            docker_image = str(os.environ.get("TRAINS_DOCKER_IMAGE") or
                               self._session.config.get("agent.default_docker.image", "nvidia/cuda"))

        # take the first part, this is the docker image name (not arguments)
        docker_image = docker_image.split()[0]

        hocon_config_encoded = HOCONConverter.to_hocon(
                self._session.config._config
            ).encode('ascii')
        create_trains_conf = "echo '{}' | base64 --decode >> ~/trains.conf && ".format(
            base64.b64encode(
                hocon_config_encoded
            ).decode('ascii')
        )

        if callable(self.kubectl_cmd):
            kubectl_cmd = self.kubectl_cmd(task_id, docker_image, queue, task_data)
        else:
            kubectl_cmd = self.kubectl_cmd.format(
                task_id=task_id,
                docker_image=docker_image,
                queue_id=queue
            )

        # Search for a free pod number
        pod_number = 1
        while self.ports_mode:
            kubectl_cmd_new = "kubectl get pods -l {pod_label},{agent_label} -n trains".format(
                pod_label=self.LIMIT_POD_LABEL.format(pod_number=pod_number),
                agent_label=self.AGENT_LABEL
            )
            process = subprocess.Popen(kubectl_cmd_new.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
            if not output:
                # No such pod exist so we can use the pod_number we found
                break
            if pod_number >= self.num_of_services:
                # All pod numbers are taken, exit
                self.log.info(
                    "All k8s services are in use, task '{}' will be enqueued back to queue '{}'".format(
                        task_id, queue
                    )
                )
                self._session.api_client.tasks.reset(task_id)
                self._session.api_client.tasks.enqueue(task_id, queue=queue)
                return
            pod_number += 1

        # make sure we provide a list
        if isinstance(kubectl_cmd, str):
            kubectl_cmd = kubectl_cmd.split()

        labels = [self.AGENT_LABEL]
        message = "K8s scheduling experiment task id={}".format(task_id)
        if self.ports_mode:
            labels.insert(0, self.LIMIT_POD_LABEL.format(pod_number=pod_number))
            message += " pod #{}".format(pod_number)

        kubectl_cmd += [
            "--labels=" + ",".join(labels),
            "--command",
            "--",
            "/bin/sh",
            "-c",
            create_trains_conf + self.container_bash_script.format(task_id),
        ]
        process = subprocess.Popen(kubectl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        self.log.info(message)
        if error:
            self.log.error("Running kubectl encountered an error: {}".format(
                error if isinstance(error, str) else error.decode()))

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
                # delete old completed /failed pods
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
        return self.daemon(queues=[ObjectID(name=queue)], log_level=logging.INFO, foreground=True, docker=False)
