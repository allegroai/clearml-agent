from __future__ import print_function, division, unicode_literals

import errno
import json
import logging
import os
import os.path
import re
import signal
import subprocess
import sys
import shutil
import traceback
from collections import defaultdict
from copy import copy, deepcopy
from datetime import datetime
from distutils.spawn import find_executable
from functools import partial
from itertools import chain
from tempfile import mkdtemp, NamedTemporaryFile
from time import sleep, time
from typing import Text, Optional, Any, Tuple

import attr
import psutil
import six
from trains_agent.backend_api.services import queues as queues_api
from trains_agent.backend_api.services import tasks as tasks_api
from pathlib2 import Path
from pyhocon import ConfigTree, ConfigFactory
from six.moves.urllib.parse import quote

from trains_agent.backend_api.services import queues
from trains_agent.backend_config.defs import UptimeConf
from trains_agent.helper.check_update import start_check_update_daemon
from trains_agent.commands.base import resolve_names, ServiceCommandSection
from trains_agent.definitions import (
    WORKER_ALREADY_REGISTERED,
    ENVIRONMENT_SDK_PARAMS,
    INVALID_WORKER_ID,
    PROGRAM_NAME,
    DEFAULT_VENV_UPDATE_URL,
    ENV_TASK_EXECUTE_AS_USER,
    ENV_DOCKER_HOST_MOUNT,
    ENV_TASK_EXTRA_PYTHON_PATH,
    ENV_AGENT_GIT_USER,
    ENV_AGENT_GIT_PASS)
from trains_agent.definitions import WORKING_REPOSITORY_DIR, PIP_EXTRA_INDICES
from trains_agent.errors import APIError, CommandFailedError, Sigterm
from trains_agent.helper.base import (
    return_list,
    print_parameters,
    dump_yaml,
    warning,
    normalize_path,
    check_directory_path,
    select_for_platform,
    mkstemp as safe_mkstemp,
    print_table,
    safe_remove_file,
    is_windows_platform,
    rm_tree,
    is_conda,
    named_temporary_file,
    ExecutionInfo,
    HOCONEncoder,
    error,
    get_python_path,
    is_linux_platform,
    rm_file,
    add_python_path, safe_remove_tree, )
from trains_agent.helper.console import ensure_text, print_text, decode_binary_lines
from trains_agent.helper.os.daemonize import daemonize_process
from trains_agent.helper.package.base import PackageManager
from trains_agent.helper.package.conda_api import CondaAPI
from trains_agent.helper.package.post_req import PostRequirement
from trains_agent.helper.package.external_req import ExternalRequirements
from trains_agent.helper.package.pip_api.system import SystemPip
from trains_agent.helper.package.pip_api.venv import VirtualenvPip
from trains_agent.helper.package.poetry_api import PoetryConfig, PoetryAPI
from trains_agent.helper.package.pytorch import PytorchRequirement
from trains_agent.helper.package.requirements import RequirementsManager
from trains_agent.helper.package.venv_update_api import VenvUpdateAPI
from trains_agent.helper.process import (
    kill_all_child_processes,
    WorkerParams,
    ExitStatus,
    Argv,
    COMMAND_SUCCESS,
    Executable,
    get_bash_output,
    shutdown_docker_process,
    get_docker_id,
    commit_docker, terminate_process,
)
from trains_agent.helper.package.priority_req import PriorityPackageRequirement
from trains_agent.helper.repo import clone_repository_cached, RepoInfo, VCS
from trains_agent.helper.resource_monitor import ResourceMonitor
from trains_agent.helper.runtime_verification import check_runtime, print_uptime_properties
from trains_agent.session import Session
from trains_agent.helper.singleton import Singleton

from .events import Events

DOCKER_ROOT_CONF_FILE = "/root/trains.conf"
DOCKER_DEFAULT_CONF_FILE = "/root/default_trains.conf"


@attr.s
class LiteralScriptManager(object):
    """
    Manage notebook tasks
    """

    venv_folder = attr.ib(type=str)

    @staticmethod
    def is_literal_script(task):
        # type: (tasks_api.Task) -> bool
        """
        Returns whether a task object represents a notebook task
        """
        script = task.script
        if not script:
            return False
        diff = script.diff
        if not diff:
            return False

        # test git diff prefix
        if diff.lstrip().lower().startswith("diff "):
            return False

        # test git submodule prefix
        # noinspection PyBroadException
        try:
            if diff.lstrip().lower().startswith("submodule ") and \
                    diff.splitlines()[1].lstrip().lower().startswith("diff "):
                return False
        except Exception:
            pass

        # none of the above
        return True

    @staticmethod
    def write(task, directory, entry_point=None):
        # type: (tasks_api.Task, str, Optional[str]) -> str
        """
        Create notebook file for ``task`` in ``directory``
        """
        if entry_point:
            full_path = normalize_path(Text(directory), entry_point)
            if os.path.exists(full_path):
                return entry_point

            with open(full_path, "wt") as f:
                f.write(task.script.diff)
                return full_path

        with named_temporary_file(
            delete=False, prefix="script_", suffix=".py", dir=Text(directory), mode="wt"
        ) as f:
            f.write(task.script.diff)
            return f.name

    def create_notebook_file(self, task, execution, repo_info):
        # type: (tasks_api.Task, ExecutionInfo, Optional[RepoInfo]) -> Tuple[str, str]
        """
        Create notebook file in appropriate location
        :return: directory and script path
        """
        log = logging.getLogger(__name__)
        if repo_info and repo_info.root:
            location = Path(repo_info.root, execution.working_dir)
        else:
            if execution.working_dir and execution.working_dir.strip() != '.':
                log.warning(
                    "found task with `script.working_dir` (`%s`) but without `script.repository`, ignoring",
                    execution.working_dir,
                )
            if not execution.entry_point:
                execution.entry_point = 'untitled.py'
            else:
                # ignore any folders in the entry point we only need the file name
                execution.entry_point = execution.entry_point.split(os.path.sep)[-1]
            location = None
        location = location or (repo_info and repo_info.root)
        if not location:
            location = Path(self.venv_folder, "code")
            location.mkdir(exist_ok=True)
        log.debug("selected execution directory: %s", location)
        return Text(location), self.write(task, location, execution.entry_point)


def get_repo_auth_string(user, password):
    # type: (Text, Text) -> Text
    """
    Return user:password only if user and password are valid
    :param user: username
    :param password:
    :return: URL authentication string
    """
    if not (user and password):
        return ""
    return ":".join(map(quote, (user, password)))


CONCAT_CMD = select_for_platform(linux=" && ", windows=" & ")


class TaskStopReason(object):
    no_stop = 0         # type: TaskStopReason
    stopped = 1         # type: TaskStopReason
    reset = 2           # type: TaskStopReason
    status_changed = 3  # type: TaskStopReason


def get_task(session, task_id, *args, **kwargs):
    return session.api_client.tasks.get_all(id=[task_id], *args, **kwargs)[0]


class TaskStopSignal(object):
    """
    Follow task status and signals when it should be stopped
    """

    _number_of_consecutive_reset_tests = 4
    statuses = tasks_api.TaskStatusEnum
    unexpected_statuses = [
        statuses.closed,
        statuses.stopped,
        statuses.failed,
        statuses.published,
        statuses.queued,
    ]
    default = TaskStopReason.no_stop
    stopping_message = "stopping"

    def __init__(self, command, session, events_service, task_id):
        # type: (Worker, Session, Events, Text) -> ()
        """
        :param command: workers command
        :param session: command session
        :param events_service: events service object
        :param task_id: followed task ID
        """
        self.command = command
        self.session = session
        self.events_service = events_service
        self.worker_id = command.worker_id
        self._task_reset_state_counter = 0
        self.task_id = task_id

    def test(self):
        # type: () -> TaskStopReason
        """
        Returns whether task should stop and for what reason,
        returns TaskStopReason.no_stop if task shouldn't stop.
        Catches and logs exceptions.
        """
        try:
            return self._test()
        except Exception as ex:
            self.command.log_traceback(ex)
            # make sure we break nothing
            return TaskStopSignal.default

    def _test(self):
        # type: () -> TaskStopReason
        """
        "Unsafe" version of test()
        """
        task_info = get_task(
            self.session, self.task_id, only_fields=["status", "status_message"]
        )
        status = task_info.status
        message = task_info.status_message

        if status == self.statuses.in_progress and self.stopping_message in message:
            self.command.log(
                "task status_message has '%s', task will terminate",
                self.stopping_message,
            )
            return TaskStopReason.stopped

        if status in self.unexpected_statuses:  ## and "worker" not in message:
            self.command.log("unexpected status change, task will terminate")
            return TaskStopReason.status_changed

        if status == self.statuses.created:
            if (
                self._task_reset_state_counter
                >= self._number_of_consecutive_reset_tests
            ):
                self.command.log("task was reset, task will terminate")
                return TaskStopReason.reset
            self._task_reset_state_counter += 1
            warning_msg = "Warning: Task {} was reset! if state is consistent we shall terminate ({}/{}).".format(
                self.task_id,
                self._task_reset_state_counter,
                self._number_of_consecutive_reset_tests,
            )
            if self.events_service:
                self.events_service.send_log_events(
                    self.worker_id,
                    task_id=self.task_id,
                    lines=[warning_msg],
                    level="WARNING",
                )
            print(warning_msg)
        else:
            self._task_reset_state_counter = 0

        return TaskStopReason.no_stop


class Worker(ServiceCommandSection):
    _pip_extra_index_url = PIP_EXTRA_INDICES

    _default_pip = tuple()

    _requirement_substitutions = (
        PytorchRequirement,
        PriorityPackageRequirement,
        PostRequirement,
        ExternalRequirements,
    )

    # poll queues every _polling_interval seconds
    _polling_interval = 5.0
    # machine status update intervals, seconds
    _machine_update_interval = 30.0

    # message printed before starting task logging,
    # it will be parsed by services_mode, to identify internal docker logging start
    _task_logging_start_message = "Running task '{}'"
    # last message before passing control to the actual task
    _task_logging_pass_control_message = "Running task id [{}]:"

    _run_as_user_home = '/trains_agent_home'
    _docker_fixed_user_cache = '/trains_agent_cache'
    _temp_cleanup_list = []

    @property
    def service(self):
        """ Worker command service endpoint """
        return "workers"

    @property
    def _task_status_change_message(self):
        return "Changed by {} {}".format(PROGRAM_NAME, self.worker_id)

    @staticmethod
    def register_signal_handler():
        def handler(*_):
            for f in Worker._temp_cleanup_list + [Singleton.get_pid_file()]:
                safe_remove_tree(f)
            raise Sigterm()

        signal.signal(signal.SIGTERM, handler)

    def __init__(self, *args, **kwargs):
        super(Worker, self).__init__(*args, **kwargs)
        self.monitor = None
        self.log = self._session.get_logger(__name__)
        self.register_signal_handler()
        self._worker_registered = False
        self.is_conda = is_conda(self._session.config)  # type: bool
        # Add extra index url - system wide
        extra_url = None
        try:
            if self._session.config.get("agent.package_manager.extra_index_url", None):
                extra_url = self._session.config.get("agent.package_manager.extra_index_url", [])
                if not isinstance(extra_url, (tuple, list)):
                    extra_url = [extra_url]
                # put external pip url before default ones, so we first look for packages there
                for e in reversed(extra_url):
                    self._pip_extra_index_url.insert(0, e)
        except Exception:
            self.log.warning('Failed adding extra-index-url to pip environment: {}'.format(extra_url))
        # update pip install command
        pip_install_cmd = ["pip", "install"]
        if self._pip_extra_index_url:
            pip_install_cmd.extend(
                chain.from_iterable(
                    ("--extra-index-url", x) for x in self._pip_extra_index_url
                )
            )
        self.pip_install_cmd = tuple(pip_install_cmd)
        self.worker_id = self._session.config["agent.worker_id"] or "{}:{}".format(
            self._session.config["agent.worker_name"], os.getpid()
        )
        self._last_stats = defaultdict(lambda: 0)
        self._last_report_timestamp = psutil.time.time()
        self.temp_config_path = None
        self.queues = ()
        self.venv_folder = None  # type: Optional[Text]
        self.package_api = None  # type: PackageManager
        self.global_package_api = None

        self.is_venv_update = self._session.config.agent.venv_update.enabled
        self.poetry = PoetryConfig(self._session)
        self.docker_image_func = None
        self._docker_image = None
        self._docker_arguments = None
        PackageManager.set_pip_version(self._session.config.get("agent.package_manager.pip_version", None))
        self._extra_docker_arguments = self._session.config.get("agent.extra_docker_arguments", None)
        self._extra_shell_script = self._session.config.get("agent.extra_docker_shell_script", None)
        self._docker_force_pull = self._session.config.get("agent.docker_force_pull", False)
        self._daemon_foreground = None
        self._standalone_mode = None
        self._services_mode = None
        self._force_current_version = None
        self._redirected_stdout_file_no = None
        self._uptime_config = self._session.config.get("agent.uptime", None)
        self._downtime_config = self._session.config.get("agent.downtime", None)
        self._suppress_cr = self._session.config.get("agent.suppress_carriage_return", True)

        # True - supported
        # None - not initialized
        # str - not supported, version string indicates last server version
        self._runtime_props_support = None

    @classmethod
    def _verify_command_states(cls, kwargs):
        """
        Conform and enforce command argument
        This is where you can automatically turn on/off switches based on different states.
        :param kwargs:
        :return: kwargs
        """
        if kwargs.get('services_mode'):
            kwargs['cpu_only'] = True
            kwargs['docker'] = kwargs.get('docker') or []
            kwargs['gpus'] = None

        return kwargs

    def _get_requirements_manager(self, os_override=None, base_interpreter=None):
        requirements_manager = RequirementsManager(
            self._session, base_interpreter=base_interpreter
        )
        for requirement_cls in self._requirement_substitutions:
            if os_override and issubclass(requirement_cls, PytorchRequirement):
                requirement_cls = partial(requirement_cls, os_override=os_override)
            requirements_manager.register(requirement_cls)
        if not requirements_manager.found_cuda:
            self.warning(
                "could not find installed CUDA/CuDNN version using original requirements file"
            )
        return requirements_manager

    def handle_user_abort(self, task_id):
        """
        Set task status to appropriate value on user abort.
        """
        try:
            task_status = self._session.send_api(
                tasks_api.GetByIdRequest(task_id)
            ).task.status
            if task_status == tasks_api.TaskStatusEnum.in_progress:
                print("\nUser abort - stopping task {}".format(task_id))
                self._session.send_api(tasks_api.StoppedRequest(task_id))
        except Exception:
            pass

    def run_one_task(self, queue, task_id, worker_args, docker=None):
        # type: (Text, Text, WorkerParams, Optional[Text]) -> ()
        """
        Run one task pulled from queue.
        :param queue: ID of queue that task was pulled from
        :param task_id: ID of task to run
        :param worker_args: Worker command line arguments
        :param docker: Docker image in which the execution task will run
        """
        # start new process and execute task id
        # "Running task '{}'".format(task_id)
        print(self._task_logging_start_message.format(task_id))

        # set task status to in_progress so we know it was popped from the queue
        try:
            self._session.send_api(tasks_api.StartedRequest(task=task_id, force=True))
        except Exception:
            print("Warning: Could not start task id '{}', skipping".format(task_id))
            return
        # setup console log
        temp_stdout_name = safe_mkstemp(
            suffix=".txt", prefix=".trains_agent_out.", name_only=True
        )
        # temp_stderr_name = safe_mkstemp(suffix=".txt", prefix=".trains_agent_err.", name_only=True)
        temp_stderr_name = None
        print(
            "Storing stdout and stderr log to '{}', '{}'".format(
                temp_stdout_name, temp_stderr_name or temp_stdout_name
            )
        )

        docker_image = None
        if self.docker_image_func:
            try:
                response = get_task(self._session, task_id, only_fields=["execution.docker_cmd"])
                task_docker_cmd = docker or response.execution.docker_cmd
                task_docker_cmd = task_docker_cmd.strip() if task_docker_cmd else None
            except Exception:
                task_docker_cmd = None

            if task_docker_cmd:
                self.send_logs(task_id=task_id,
                               lines=['Running Task {} inside docker: {}\n'.format(task_id, task_docker_cmd)],
                               level="INFO")
                task_docker_cmd = task_docker_cmd.split(' ')
                docker_image = task_docker_cmd[0]
                docker_arguments = task_docker_cmd[1:]
            else:
                self.send_logs(task_id=task_id,
                               lines=['running Task {} inside default docker image: {} {}\n'.format(
                                   task_id, self._docker_image, self._docker_arguments or '')],
                               level="INFO")
                docker_image = self._docker_image
                docker_arguments = self._docker_arguments

            # Update docker command
            if self._services_mode:
                # if this is services mode, give the docker a unique worker id, as it will register itself.
                full_docker_cmd = self.docker_image_func(
                    worker_id='{}:service:{}'.format(self.worker_id, task_id),
                    docker_image=docker_image, docker_arguments=docker_arguments)
            else:
                full_docker_cmd = self.docker_image_func(docker_image=docker_image, docker_arguments=docker_arguments)
            try:
                self._session.send_api(
                    tasks_api.EditRequest(task_id, force=True, execution=dict(
                        docker_cmd=' '.join([docker_image] + docker_arguments) if docker_arguments else docker_image)))
            except Exception:
                pass

            # if this is services_mode, change the worker_id to a unique name
            # abd use full-monitoring, ot it registers itself as a worker for this specific service.
            # notice, the internal agent will monitor itself once the docker is up and running
            full_docker_cmd[-1] = full_docker_cmd[-1] + 'execute {} {} --id {}'.format(
                '--full-monitoring' if self._services_mode else '--disable-monitoring',
                '--standalone-mode' if self._standalone_mode else '',
                task_id)
            cmd = Argv(*full_docker_cmd)
            print('Running Docker:\n{}\n'.format(str(cmd)))
        else:
            cmd = worker_args.get_argv_for_command("execute") + (
                "--disable-monitoring",
                "--id",
                task_id,
            )

        events_service = self.get_service(Events)
        stop_signal = TaskStopSignal(
            command=self,
            session=self._session,
            events_service=events_service,
            task_id=task_id,
        )
        stop_signal_status = TaskStopSignal.default
        status = ExitStatus.failure
        try:
            # set WORKER ID
            os.environ['TRAINS_WORKER_ID'] = self.worker_id

            if self._docker_force_pull and docker_image:
                full_pull_cmd = ['docker', 'pull', docker_image]
                pull_cmd = Argv(*full_pull_cmd)
                status, stop_signal_status = self._log_command_output(
                    task_id=task_id,
                    cmd=pull_cmd,
                    stdout_path=temp_stdout_name,
                    stderr_path=temp_stderr_name,
                    daemon=True,
                    stop_signal=stop_signal,
                )

            status, stop_signal_status = self._log_command_output(
                task_id=task_id,
                cmd=cmd,
                stdout_path=temp_stdout_name,
                stderr_path=temp_stderr_name,
                daemon=True,
                stop_signal=stop_signal,
            )
            errors = temp_stderr_name and Path(temp_stderr_name).read_text()
            if errors:
                print("\nEncountered errors:\n\n{}\n".format(errors))
            if status is None:
                print(
                    "DONE: Running task '{}' (user aborted)".format(task_id)
                )
            else:
                print("DONE: Running task '{}', exit status {}".format(task_id, status))
        except KeyboardInterrupt:
            self.handle_user_abort(task_id)
            status = ExitStatus.interrupted
        finally:
            if self._services_mode and stop_signal_status is None:
                print('Service started, docker running in the background')
            else:
                self.handle_task_termination(task_id, status, stop_signal_status)
                # remove temp files after we sent everything to the backend
                if self.docker_image_func:
                    shutdown_docker_process(docker_cmd_contains='--id {}\'\"'.format(task_id))
                safe_remove_file(temp_stdout_name)
                safe_remove_file(temp_stderr_name)

    def run_tasks_loop(self, queues, worker_params, priority_order=True):
        """
        :summary: Pull and run tasks from queues.
        :description: 1. Go through ``queues`` by order.
                      2. Try getting the next task for each and run the first one that returns.
                      3. Go to step 1
        :param queues: IDs of queues to pull tasks from
        :type queues: list of ``Text``
        :param worker_params: Worker command line arguments
        :type worker_params: ``trains_agent.helper.process.WorkerParams``
        :param priority_order: If True pull order in priority manner. always from the first
            If False, pull from each queue once in a round robin manner
        :type priority_order: bool
        """

        if not self._daemon_foreground:
            print('Starting infinite task polling loop...')

        _last_machine_update_ts = 0

        while True:
            queue_tags = None
            runtime_props = None
            # iterate over queues (priority style, queues[0] is highest)
            for queue in queues:

                if queue_tags is None or runtime_props is None:
                    queue_tags, runtime_props = self.get_worker_properties(queues)

                if not self.should_be_currently_active(queue_tags[queue], runtime_props):
                    continue

                # get next task in queue
                try:
                    response = self._session.send_api(
                        queues_api.GetNextTaskRequest(queue=queue)
                    )
                except Exception as e:
                    print(
                        "Warning: Could not access task queue [{}], error: {}".format(
                            queue, e
                        )
                    )
                    continue
                else:
                    try:
                        task_id = response.entry.task
                    except AttributeError:
                        if self._daemon_foreground or worker_params.debug:
                            print("No tasks in queue {}".format(queue))
                        continue

                    # clear output log if we start a new Task
                    if not worker_params.debug and self._redirected_stdout_file_no is not None and \
                            self._redirected_stdout_file_no > 2:
                        # noinspection PyBroadException
                        try:
                            os.lseek(self._redirected_stdout_file_no, 0, 0)
                            os.ftruncate(self._redirected_stdout_file_no, 0)
                        except:
                            pass

                    self.send_logs(
                        task_id=task_id,
                        lines=["task {} pulled from {} by worker {}\n".format(task_id, queue, self.worker_id)],
                        level="INFO",
                    )
                    self.report_monitor(ResourceMonitor.StatusReport(queues=queues, queue=queue, task=task_id))
                    self.run_one_task(queue, task_id, worker_params)
                    self.report_monitor(ResourceMonitor.StatusReport(queues=self.queues))

                    queue_tags = None
                    runtime_props = None

                    # if we are using priority start pulling from the first always,
                    # if we are doing round robin, pull from the next one
                    if priority_order:
                        break
            else:
                # sleep and retry polling
                if self._daemon_foreground or worker_params.debug:
                    print("No tasks in Queues, sleeping for {:.1f} seconds".format(self._polling_interval))
                sleep(self._polling_interval)

            if self._session.config["agent.reload_config"]:
                self.reload_config()

    def get_worker_properties(self, queue_ids):
        queue_tags = {
            q.id: {'name': q.name, 'tags': q.tags}
            for q in self._session.send_api(
                queues_api.GetAllRequest(id=queue_ids, only_fields=["id", "tags"])
            ).queues
        }
        runtime_props = self.get_runtime_properties()
        return queue_tags, runtime_props

    def get_runtime_properties(self):
        if self._runtime_props_support is not True:
            # either not supported or never tested
            if self._runtime_props_support == self._session.api_version:
                # tested against latest api_version, not supported
                return []
            if not self._session.check_min_api_version(UptimeConf.min_api_version):
                # not supported due to insufficient api_version
                self._runtime_props_support = self._session.api_version
                return []
        try:
            res = self.get("get_runtime_properties", worker=self.worker_id)["runtime_properties"]
            # definitely supported
            self._runtime_props_support = True
            return res
        except APIError:
            self._runtime_props_support = self._session.api_version
        return []

    def should_be_currently_active(self, current_queue, runtime_properties):
        """
        Checks if a worker is active according to queue tags, worker's runtime properties and uptime schedule.
        """
        if UptimeConf.queue_tag_off in current_queue['tags']:
            self.log.debug("Queue {} is tagged '{}', worker will not pull tasks".format(
                current_queue['name'], UptimeConf.queue_tag_off)
            )
            return False
        if UptimeConf.queue_tag_on in current_queue['tags']:
            self.log.debug("Queue {} is tagged '{}', worker will pull tasks".format(
                current_queue['name'], UptimeConf.queue_tag_on)
            )
            return True
        force_flag = next(
            (prop for prop in runtime_properties if prop["key"] == UptimeConf.worker_key), None
        )
        if force_flag:
            if force_flag["value"].lower() in UptimeConf.worker_value_off:
                self.log.debug("worker has the following runtime property: '{}'. worker will not pull tasks".format(
                    force_flag)
                )
                return False
            elif force_flag["value"].lower() in UptimeConf.worker_value_on:
                self.log.debug("worker has the following runtime property: '{}'. worker will pull tasks".format(
                    force_flag)
                )
                return True
            else:
                print(
                    "Warning: invalid runtime_property '{}: {}' supported values are: '{}/{}', ignoring".format(
                        force_flag["key"], force_flag["value"], UptimeConf.worker_value_on, UptimeConf.worker_value_off
                    )
                )
        if self._uptime_config:
            self.log.debug("following uptime configurations")
            return check_runtime(self._uptime_config)
        if self._downtime_config:
            self.log.debug("following downtime configurations")
            return check_runtime(self._downtime_config, is_uptime=False)
        return True

    def reload_config(self):
        try:
            reloaded = self._session.reload()
        except Exception as ex:
            self.log("Failed reloading config file")
            self.log_traceback(ex)
        else:
            if reloaded:
                self.log(
                    'Config file change detected, reloading and updating "{.temp_config_path}"'.format(
                        self
                    )
                )
                self.dump_config()

    def check(self, **_):
        try:
            check_directory_path(str(Path(".").resolve()), check_whitespace_in_path=False)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise CommandFailedError("current working directory does not exist")
            raise

        for key in "agent.venvs_dir", "sdk.storage.cache.default_base_dir":
            try:
                value = self._session.config.get(key, None)
                if value:
                    check_directory_path(value)
            except CommandFailedError as e:
                raise CommandFailedError(
                    'Invalid config key "{}": {.message}'.format(key, e)
                )

        if is_windows_platform():
            # if not self.is_conda:
            #     self.warning("Worker on Windows without Conda are not supported")
            if self._session.config.agent.venv_update:
                # self.warning("venv-update is not supported on Windows")
                self.is_venv_update = False

        self._session.print_configuration()

    def daemon(self, queues, log_level, foreground=False, docker=False, detached=False, order_fairness=False, **kwargs):
        # if we do not need to create queues, make sure they are valid
        # match previous behaviour when we validated queue names before everything else
        queues = self._resolve_queue_names(queues, create_if_missing=kwargs.get('create_queue', False))

        self._standalone_mode = kwargs.get('standalone_mode', False)
        self._services_mode = kwargs.get('services_mode', False)
        # must have docker in services_mode
        if self._services_mode:
            kwargs = self._verify_command_states(kwargs)
            docker = docker or kwargs.get('docker')
        self._uptime_config = kwargs.get('uptime', None) or self._uptime_config
        self._downtime_config = kwargs.get('downtime', None) or self._downtime_config
        if self._uptime_config and self._downtime_config:
            self.log.error(
                "Both uptime and downtime were specified when only one of them could be used. Both will be ignored."
            )
            self._uptime_config = None
            self._downtime_config = None

        # We are not running a daemon we are killing one.
        # find the pid send termination signal and leave
        if kwargs.get('stop', False):
            return 1 if not self._kill_daemon() else 0

        queues_info = [
            q.to_dict()
            for q in self._session.send_api(
                queues_api.GetAllRequest(id=queues)
            ).queues
        ]

        if kwargs.get('status', False):
            runtime_properties = self.get_runtime_properties()
            if self._downtime_config:
                print_uptime_properties(self._downtime_config, queues_info, runtime_properties, is_uptime=False)
            else:
                print_uptime_properties(self._uptime_config, queues_info, runtime_properties)
            return 1

        # make sure we only have a single instance,
        # also make sure we set worker_id properly and cache folders
        self._singleton()

        # check if we have the latest version
        start_check_update_daemon()

        self.check(**kwargs)
        self.log.debug("starting resource monitor thread")
        print("Worker \"{}\" - ".format(self.worker_id), end='')

        columns = ("id", "name", "tags")
        print("Listening to queues:")
        print_table(queues_info, columns=columns, titles=columns)

        # register worker
        self._register(queues)

        # create temp config file with current configuration
        self.temp_config_path = NamedTemporaryFile(
            suffix=".cfg", prefix=".trains_agent.", mode='w+t').name

        # print docker image
        if docker is not False and docker is not None:
            self._force_current_version = kwargs.get('force_current_version', False)
            self.set_docker_variables(docker)
        else:
            self.dump_config()
            # only in none docker we have to make sure we have CUDA setup

            # make sure we have CUDA set if we have --gpus
            if kwargs.get('gpus') and self._session.config.get('agent.cuda_version', None) in (None, 0, '0'):
                message = 'Running with GPUs but no CUDA version was detected!\n' \
                          '\tSet OS environemnt CUDA_VERSION & CUDNN_VERSION to the correct version\n' \
                          '\tExample: export CUDA_VERSION=10.1 or (Windows: set CUDA_VERSION=10.1)'
                if is_conda(self._session.config):
                    self._unregister(queues)
                    safe_remove_file(self.temp_config_path)
                    raise ValueError(message)
                else:
                    warning(message+'\n')

        if self._services_mode:
            print('Trains-Agent running in services mode')

        self._daemon_foreground = foreground
        if not foreground:
            out_file, name = safe_mkstemp(
                prefix=".trains_agent_daemon_out",
                suffix=".txt",
                open_kwargs={
                    "buffering": self._session.config.get("agent.log_files_buffering", 1)
                },
            )
            print(
                "Running TRAINS-AGENT daemon in background mode, writing stdout/stderr to {}".format(
                    name
                )
            )

            if not self._session.debug_mode:
                self._temp_cleanup_list.append(name)

            if not detached:
                # redirect std out/err to new file
                sys.stdout = sys.stderr = out_file
            else:
                # in detached mode
                # fully detach stdin.stdout/stderr and leave main process, running in the background
                daemonize_process(out_file.fileno())
                self._redirected_stdout_file_no = out_file.fileno()
                # make sure we update the singleton lock file to the new pid
                Singleton.update_pid_file()
                # reprint headers to std file (we are now inside the daemon process)
                print("Worker \"{}\" :".format(self.worker_id))
                self._session.print_configuration()
                print_table(queues_info, columns=columns, titles=columns)

        try:
            while True:
                try:
                    self.new_monitor(ResourceMonitor.StatusReport(queues=queues))
                    self.run_tasks_loop(
                        queues,
                        worker_params=WorkerParams(
                            log_level=log_level,
                            config_file=self.temp_config_path,
                            debug=self._session.debug_mode,
                            trace=self._session.trace,
                        ),
                        priority_order=not order_fairness,
                    )
                except Exception:
                    tb = six.text_type(traceback.format_exc())
                    print("FATAL ERROR:")
                    print(tb)
                    crash_file, name = safe_mkstemp(prefix=".trains_agent-crash", suffix=".log")
                    try:
                        with crash_file:
                            crash_file.write(tb)
                    except Exception:
                        print(
                            "Could not write crash log to {}\nException:\n{}".format(
                                name, tb
                            )
                        )
                    sleep(1)
        finally:
            self._unregister(queues)
            safe_remove_file(self.temp_config_path)

    def report_monitor(self, report):
        if not self.monitor:
            self.new_monitor(report=report)
        else:
            self.monitor.set_report(report)
        self.monitor.send_report()

    def stop_monitor(self):
        if self.monitor:
            self.monitor.stop()
            self.monitor = None

    def new_monitor(self, report=None):
        self.stop_monitor()
        self.monitor = ResourceMonitor(
            session=self._session, worker_id=self.worker_id,
            first_report_sec=3.0,
            report_frequency_sec=self._machine_update_interval)
        self.monitor.set_report(report)
        self.monitor.start()
        return self.monitor

    def dump_config(self, config=None):
        def to_json(config):
            return json.dumps(config.as_plain_ordered_dict(), cls=HOCONEncoder, indent=4)
        try:
            Path(self.temp_config_path).write_text(
                six.text_type(self._session.to_json() if config is None else to_json(config)))
        except Exception:
            return False
        return True

    def _log_command_output(
        self,
        task_id,  # type: Text
        cmd,  # type: Executable
        stdout_path=None,  # type: Text
        stderr_path=None,  # type: Optional[Text]
        daemon=False,  # type: bool
        cwd=None,  # type: Text
        stop_signal=None,  # type: Optional[TaskStopSignal]
        **kwargs  # type: Any
    ):
        # type: (...) -> Tuple[Optional[int], TaskStopReason]
        def _print_file(file_path, prev_pos=0):
            with open(file_path, "rb") as f:
                f.seek(prev_pos)
                binary_text = f.read()
                pos = f.tell()
            # skip the previously printed lines,
            blines = binary_text.split(b'\n') if binary_text else []
            if not blines:
                return blines, pos
            return (
                decode_binary_lines(blines if blines[-1] else blines[:-1],
                                    replace_cr=not self._suppress_cr,
                                    overwrite_cr=self._suppress_cr),
                pos
            )

        stdout = open(stdout_path, "wt")
        stderr = open(stderr_path, "wt") if stderr_path else stdout
        stdout_line_count, stdout_pos_count, stdout_last_lines = 0, 0, []
        stderr_line_count, stderr_pos_count, stderr_last_lines = 0, 0, []
        service_mode_internal_agent_started = None
        stopping = False
        status = None
        try:
            _last_machine_update_ts = time()
            stop_reason = None

            process = cmd.call_subprocess(
                subprocess.Popen,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd and str(cwd),
                **kwargs
            )

            while status is None and not stopping:

                stop_reason = stop_signal.test() if stop_signal else TaskStopSignal.default
                if stop_reason != TaskStopSignal.default:
                    # mark quit loop
                    stopping = True
                    if daemon:
                        self.send_logs(
                            task_id=task_id,
                            lines=["User aborted: stopping task ({})\n".format(str(stop_reason))],
                            level="ERROR",
                        )
                        kill_all_child_processes(process.pid)
                else:
                    sleep(self._polling_interval)
                    status = process.poll()
                # flush stdout and stderr buffers
                if stdout:
                    stdout.flush()
                if stderr:
                    stderr.flush()

                # get diff from previous poll
                printed_lines, stdout_pos_count = _print_file(stdout_path, stdout_pos_count)
                if self._services_mode and not stopping and not status:
                    # if the internal agent started, we stop logging, it will take over logging.
                    # if the internal agent started running the task itself, it will return status==0,
                    # then we can quit the monitoring loop of this process
                    printed_lines, service_mode_internal_agent_started, status = self._check_if_internal_agent_started(
                        printed_lines, service_mode_internal_agent_started, task_id)
                    if status is not None:
                        stop_reason = 'Service started'

                stdout_line_count += self.send_logs(task_id, printed_lines)
                if stderr_path:
                    printed_lines, stderr_pos_count = _print_file(stderr_path, stderr_pos_count)
                    stderr_line_count += self.send_logs(task_id, printed_lines)

        except subprocess.CalledProcessError as ex:
            # non zero return code
            stop_reason = 'Exception occurred'
            status = ex.returncode
        except KeyboardInterrupt:
            # so someone else will catch us
            raise
        except Exception:
            # we should not get here, but better safe than sorry
            printed_lines, stdout_pos_count = _print_file(stdout_path, stdout_pos_count)
            stdout_line_count += self.send_logs(task_id, printed_lines)
            if stderr_path:
                printed_lines, stderr_pos_count = _print_file(stderr_path, stderr_pos_count)
                stderr_line_count += self.send_logs(task_id, printed_lines)
            stop_reason = 'Exception occurred'
            status = -1

        # if running in services mode, keep the file open
        # in case the docker was so quick it started and finished, check the stop reason
        if self._services_mode and service_mode_internal_agent_started and stop_reason == 'Service started':
            return None, None

        stdout.close()
        if stderr_path:
            stderr.close()

        # Send last lines
        printed_lines, stdout_pos_count = _print_file(stdout_path, stdout_pos_count)
        stdout_line_count += self.send_logs(task_id, printed_lines)
        if stderr_path:
            printed_lines, stderr_pos_count = _print_file(stderr_path, stderr_pos_count)
            stderr_line_count += self.send_logs(task_id, printed_lines)

        return status, stop_reason

    def _check_if_internal_agent_started(self, printed_lines, service_mode_internal_agent_started, task_id):
        log_start_msg = self._task_logging_start_message.format(task_id)
        log_control_end_msg = self._task_logging_pass_control_message.format(task_id)
        filter_lines = printed_lines if not service_mode_internal_agent_started else []
        for i, line in enumerate(printed_lines):
            if not service_mode_internal_agent_started and line.startswith(log_start_msg):
                service_mode_internal_agent_started = True
                filter_lines = printed_lines[:i+1]
            elif line.startswith(log_control_end_msg):
                return filter_lines, service_mode_internal_agent_started, 0

        return filter_lines, service_mode_internal_agent_started, None

    def send_logs(self, task_id, lines, level="DEBUG"):
        """
        Send output lines as log events to backend
        :param task_id: ID of task to send logs for
        :type task_id: Text
        :param lines: lines to send
        :type lines: [Text]
        :param str level: log level, default DEBUG
        :return: number of lines sent
        :rtype: int
        """
        if not lines:
            return 0
        print_text("".join(lines), newline=False)

        # remove backspaces from the text log, they look bad.
        for i, l in enumerate(lines):
            lines[i] = l.replace('\x08', '')

        events_service = self.get_service(Events)
        try:
            events_service.send_log_events(
                self.worker_id, task_id=task_id, lines=lines, level=level
            )
            return len(lines)
        except Exception as e:
            print("\n### Error sending log: %s ###\n" % e)
            # revert number of sent lines (we will try next time)
            return 0

    def _update_commit_id(self, task_id, execution, repo_info):
        """
        If commit ID is not set, set it to the currently running version of the repository
        """
        if not repo_info.commit or execution.version_num:
            return
        self.log("Updating task commit ID: %s", repo_info.commit)
        try:
            self._session.send_api(
                tasks_api.EditRequest(
                    task_id, script=dict(version_num=repo_info.commit), force=True
                )
            )
        except Exception:
            pass

    def apply_diff(self, task, vcs, execution_info, repo_info):
        # type: (Any, VCS, ExecutionInfo, RepoInfo) -> None
        """
        Apply task diff if present
        """
        diff = task.script and task.script.diff
        if not diff:
            return
        print("Applying uncommitted changes")
        try:
            success = vcs.patch(normalize_path(repo_info.root), diff)
        except Exception as ex:
            self.log.warning("could not apply diff: %s", ex)
            success = False

        if not success:
            raise ValueError("Failed applying git diff:\n{}\n\nERROR! Failed applying git diff, see diff above.".format(diff))

    @resolve_names
    def build(
        self,
        task_id,
        target=None,
        python_version=None,
        docker=None,
        entry_point=None,
        install_globally=False,
        **_
    ):
        if not task_id:
            raise CommandFailedError("Worker build must have valid task id")

        self._session.print_configuration()

        if docker is not False and docker is not None:
            return self._build_docker(docker, target, task_id, entry_point)

        current_task = self._session.api_client.tasks.get_by_id(task_id)

        execution = self.get_execution_info(current_task)

        if self._session.config.get("agent.package_manager.force_repo_requirements_txt", False):
            requirements = None
            print("[package_manager.force_repo_requirements_txt=true] "
                  "Skipping requirements, using repository \"requirements.txt\" ")
        else:
            try:
                requirements = current_task.script.requirements
            except AttributeError:
                requirements = None

        if not python_version:
            try:
                python_version = current_task.script.binary
                python_version = python_version.split('/')[-1].replace('python', '')
                # if we can cast it, we are good
                python_version = '{:.1f}'.format(float(python_version))
            except:
                python_version = None

        venv_folder, requirements_manager = self.install_virtualenv(
            venv_dir=target, requested_python_version=python_version, execution_info=execution)

        if self._default_pip:
            if install_globally and self.global_package_api:
                self.global_package_api.install_packages(*self._default_pip)
            else:
                self.package_api.install_packages(*self._default_pip)

        directory, vcs, repo_info = self.get_repo_info(execution, current_task, venv_folder.as_posix())

        self.install_requirements(
            execution,
            repo_info,
            requirements_manager=requirements_manager,
            cached_requirements=requirements,
            cwd=vcs.location if vcs and vcs.location else directory,
            package_api=self.global_package_api if install_globally else None,
        )
        freeze = self.freeze_task_environment(requirements_manager=requirements_manager)
        script_dir = directory

        # Summary
        print("Restoring running environment of task id [%s]:" % task_id)
        if freeze:
            print("Summary - installed python packages:")
            print(dump_yaml(freeze))
        else:
            print("No freeze information available")

        print("Virtual environment: {}".format(venv_folder / 'bin'))
        print("Source code: {}".format(repo_info.root if repo_info else execution.entry_point))
        print("Entry point: {}".format(Path(script_dir) / execution.entry_point))

        return 0

    def _build_docker(self, docker, target, task_id, entry_point=None, standalone_mode=True):

        self.temp_config_path = safe_mkstemp(
            suffix=".cfg", prefix=".trains_agent.", text=True, name_only=True
        )
        if not target:
            target = "task_id_{}".format(task_id)

        temp_config, docker_image_func = self.get_docker_config_cmd(docker)
        self.dump_config(temp_config)
        self.docker_image_func = docker_image_func
        try:
            response = get_task(self._session, task_id, only_fields=["execution.docker_cmd"])
            task_docker_cmd = response.execution.docker_cmd
            task_docker_cmd = task_docker_cmd.strip() if task_docker_cmd else None
        except Exception:
            task_docker_cmd = None
        if task_docker_cmd:
            print('Building Task {} inside docker: {}\n'.format(task_id, task_docker_cmd))
            task_docker_cmd = task_docker_cmd.split(' ')
            full_docker_cmd = self.docker_image_func(docker_image=task_docker_cmd[0],
                                                     docker_arguments=task_docker_cmd[1:])
        else:
            print('Building Task {} inside default docker image: {} {}\n'.format(
                task_id, self._docker_image, self._docker_arguments or ''))
            full_docker_cmd = self.docker_image_func(docker_image=self._docker_image,
                                                     docker_arguments=self._docker_arguments)
        end_of_build_marker = "build.done=true"
        docker_cmd_suffix = ' build --id {task_id} --install-globally; ' \
                            'echo "" >> {conf_file} ; ' \
                            'echo {end_of_build_marker} >> {conf_file} ; ' \
                            'bash'.format(
                                task_id=task_id,
                                end_of_build_marker=end_of_build_marker,
                                conf_file=DOCKER_ROOT_CONF_FILE
                            )
        full_docker_cmd[-1] = full_docker_cmd[-1] + docker_cmd_suffix
        cmd = Argv(*full_docker_cmd)

        # we will be checking the configuration file for changes
        temp_config = Path(self.temp_config_path)
        base_time_stamp = temp_config.stat().st_mtime

        # start the docker
        print('Starting docker build')
        cmd.call_subprocess(subprocess.Popen)

        # now we need to wait until the line shows on our configuration file.
        while True:
            while temp_config.stat().st_mtime == base_time_stamp:
                sleep(5.0)
            with open(temp_config.as_posix()) as f:
                lines = [l.strip() for l in f.readlines()]
            if 'build.done=true' in lines:
                break
            base_time_stamp = temp_config.stat().st_mtime

        print('\nDocker build done')

        # get the docker id.
        docker_id = get_docker_id(docker_cmd_contains='--id {} '.format(task_id))
        if not docker_id:
            print("Error: cannot locate docker for storage")
            return

        if entry_point == "clone_task" or entry_point == "reuse_task":
            change = 'ENTRYPOINT if [ ! -s "{trains_conf}" ] ; then ' \
                     'cp {default_trains_conf} {trains_conf} ; ' \
                     ' fi ; trains-agent execute --id {task_id} --standalone-mode {clone}'.format(
                        default_trains_conf=DOCKER_DEFAULT_CONF_FILE,
                        trains_conf=DOCKER_ROOT_CONF_FILE,
                        task_id=task_id,
                        clone=("--clone" if entry_point == "clone_task" else ""),
                     )
        else:
            change = 'ENTRYPOINT []'

        print('Committing docker container to: {}'.format(target))
        print(commit_docker(container_name=target, docker_id=docker_id, apply_change=change))
        shutdown_docker_process(docker_id=docker_id)

        return

    @resolve_names
    def execute(
        self,
        task_id,
        log_level,
        optimization=0,
        disable_monitoring=False,
        full_monitoring=False,
        require_queue=False,
        log_file=None,
        standalone_mode=None,
        docker=False,
        clone=False,
        **_
    ):

        self._standalone_mode = standalone_mode

        if not task_id:
            raise CommandFailedError("Worker execute must have valid task id")

        try:
            current_task = self._session.api_client.tasks.get_by_id(task_id)
            if not current_task.id:
                pass
        except Exception as ex:
            raise ValueError(
                "Could not find task id={} (for host: {})\nException: {}".format(
                    task_id, self._session.config.get("api.host", ""), ex
                )
            )

        if clone:
            try:
                print("Cloning task id={}".format(task_id))
                current_task = self._session.api_client.tasks.get_by_id(
                    self._session.send_api(
                        tasks_api.CloneRequest(task=current_task.id, new_task_name='Clone of {}'.format(current_task.name))
                    ).id
                )
                print("Task cloned, new task id={}".format(current_task.id))
            except Exception:
                raise CommandFailedError("Cloning failed")
        else:
            # make sure this task is not stuck in an execution queue, it shouldn't have been, but just in case.
            try:
                res = self._session.api_client.tasks.dequeue(task=current_task.id)
                if require_queue and res.meta.result_code != 200:
                    raise ValueError("Execution required enqueued task, "
                                     "but task id={} is not queued.".format(current_task.id))
            except Exception:
                if require_queue:
                    raise

        if docker is not False and docker is not None:
            self.set_docker_variables(docker)

        # We expect the same behaviour in case full_monitoring was set, and in case docker mode is used
        if full_monitoring or docker is not False:
            worker_params = WorkerParams(
                log_level=log_level,
                config_file=self._session.config_file,
                debug=self._session.debug_mode,
                trace=self._session.trace,
            )
            self.report_monitor(ResourceMonitor.StatusReport(task=current_task.id))
            self.run_one_task(queue='', task_id=current_task.id, worker_args=worker_params, docker=docker)

            self.stop_monitor()
            self._unregister()
            return

        self._session.print_configuration()

        # now mark the task as started
        self._session.api_client.tasks.started(
            task=current_task.id,
            status_reason="worker started execution",
            status_message=self._task_status_change_message,
        )

        if not disable_monitoring:
            self.log.debug("starting resource monitor")
            self.report_monitor(ResourceMonitor.StatusReport(task=current_task.id))

        execution = self.get_execution_info(current_task)

        if self._session.config.get("agent.package_manager.force_repo_requirements_txt", False):
            requirements = None
            print("[package_manager.force_repo_requirements_txt=true] "
                  "Skipping requirements, using repository \"requirements.txt\" ")
        else:
            try:
                requirements = current_task.script.requirements
            except AttributeError:
                requirements = None

        try:
            python_ver = current_task.script.binary
            python_ver = python_ver.split('/')[-1].replace('python', '')
            # if we can cast it, we are good
            python_ver = '{:.1f}'.format(float(python_ver))
        except:
            python_ver = None

        venv_folder, requirements_manager = self.install_virtualenv(
            standalone_mode=standalone_mode, requested_python_version=python_ver, execution_info=execution)

        if not standalone_mode:
            if self._default_pip:
                self.package_api.install_packages(*self._default_pip)

            print("\n")

        directory, vcs, repo_info = self.get_repo_info(
            execution, current_task, venv_folder
        )

        print("\n")

        if not standalone_mode:
            self.install_requirements(
                execution,
                repo_info,
                requirements_manager=requirements_manager,
                cached_requirements=requirements,
                cwd=vcs.location if vcs and vcs.location else directory,
            )

        # do not update the task packages if we are using conda,
        # it will most likely make the task environment unreproducible
        skip_freeze_update = self.is_conda and not self._session.config.get(
            "agent.package_manager.conda_full_env_update", False)

        freeze = self.freeze_task_environment(current_task.id if not skip_freeze_update else None,
                                              requirements_manager=requirements_manager)
        script_dir = (directory if isinstance(directory, Path) else Path(directory)).absolute().as_posix()

        # run code
        # print("Running task id [%s]:" % current_task.id)
        print(self._task_logging_pass_control_message.format(current_task.id))
        extra = ['-u', ]
        if optimization:
            extra.append(
                WorkerParams(optimization=optimization).get_optimization_flag()
            )
        # check if this is a module load, then load it.
        try:
            if current_task.script.binary and current_task.script.binary.startswith('python') and \
                    execution.entry_point and execution.entry_point.split()[0].strip() == '-m':
                # we need to split it
                import shlex
                extra.extend(shlex.split(execution.entry_point))
            else:
                extra.append(execution.entry_point)
        except:
            extra.append(execution.entry_point)

        command = self.package_api.get_python_command(extra)
        print("[{}]$ {}".format(execution.working_dir, command.pretty()))

        if freeze:
            print("Summary - installed python packages:")
            print(dump_yaml(freeze))
        else:
            print("No freeze information available")

        print("Environment setup completed successfully\n")

        sdk_env = {
            # config_file updated in session.py
            "task_id": current_task.id,
            "log_level": log_level,
            "log_to_backend": "0",
            "config_file": self._session.config_file,  # The config file is the tmp file that trains_agent created
        }
        os.environ.update(
            {
                sdk_key: str(value)
                for key, value in sdk_env.items()
                for sdk_key in ENVIRONMENT_SDK_PARAMS[key]
            }
        )

        if repo_info:
            self._update_commit_id(current_task.id, execution, repo_info)

        # Add the script CWD to the python path
        python_path = get_python_path(script_dir, execution.entry_point, self.package_api, is_conda_env=self.is_conda)
        if os.environ.get(ENV_TASK_EXTRA_PYTHON_PATH):
            python_path = add_python_path(python_path, os.environ.get(ENV_TASK_EXTRA_PYTHON_PATH))
        if python_path:
            os.environ['PYTHONPATH'] = python_path

        # check if we want to run as another user, only supported on linux
        if os.environ.get(ENV_TASK_EXECUTE_AS_USER, None) and is_linux_platform():
            command, script_dir = self._run_as_user_patch(
                command, self._session.config_file,
                script_dir, venv_folder,
                self._session.config.get('sdk.storage.cache.default_base_dir'),
                os.environ.get(ENV_TASK_EXECUTE_AS_USER))
            use_execv = False
        else:
            use_execv = is_linux_platform() and not isinstance(self.package_api, (PoetryAPI, CondaAPI))

        print("Starting Task Execution:\n".format(current_task.id))
        exit_code = -1
        try:
            if disable_monitoring:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.chdir(script_dir)
                    if use_execv:
                        os.execv(command.argv[0].as_posix(), tuple([command.argv[0].as_posix()])+command.argv[1:])
                    else:
                        exit_code = command.check_call(cwd=script_dir)
                        exit(exit_code)
                except subprocess.CalledProcessError as ex:
                    # non zero return code
                    exit_code = ex.returncode
                    if not use_execv:
                        exit(exit_code)
                except Exception as ex:
                    if not use_execv:
                        exit(-1)
                    raise ex
            else:
                # store stdout/stderr into file, and send to backend
                temp_stdout_fname = log_file or safe_mkstemp(
                    suffix=".txt", prefix=".trains_agent_out.", name_only=True
                )
                print("Storing stdout and stderr log into [%s]" % temp_stdout_fname)
                exit_code, _ = self._log_command_output(
                    task_id=current_task.id,
                    cmd=command,
                    stdout_path=temp_stdout_fname,
                    cwd=script_dir,
                )
        except KeyboardInterrupt:
            self.handle_user_abort(current_task.id)
            raise
        except Exception as e:
            self.log.warning(str(e))
            self.log_traceback(e)
            exit_code = -1

        # kill leftover processes
        kill_all_child_processes()

        # if we return ExitStatus.interrupted==2,
        # it means user aborted, KeyboardInterrupt should have caught it,
        # that cannot happen when running with disable monitoring
        exit_code = exit_code if exit_code != ExitStatus.interrupted else -1

        if not disable_monitoring:
            # we need to change task status according to exit code
            self.handle_task_termination(current_task.id, exit_code, TaskStopReason.no_stop)
            self.stop_monitor()
            # unregister the worker
            self._unregister()

        return 1 if exit_code is None else exit_code

    def set_docker_variables(self, docker):
        temp_config, docker_image_func = self.get_docker_config_cmd(docker)
        self.dump_config(temp_config)
        self.docker_image_func = docker_image_func

    def get_execution_info(self, current_task):
        # type: (...) -> ExecutionInfo
        try:
            execution = ExecutionInfo.from_task(current_task)
        except Exception as e:
            self.error("Could not parse task execution info: {}".format(e.args[0]))
            current_task.failed(
                status_reason=e.args[0], status_message=self._task_status_change_message
            )
            self.exit(e.args[0])
        if "\\" in execution.working_dir:
            warning(
                'Working dir "{}" contains backslashes. '
                "All path separators must be forward slashes.".format(
                    execution.working_dir
                )
            )
        print("Executing task id [%s]:" % current_task.id)
        for pair in attr.asdict(execution).items():
            print("{} = {}".format(*pair))
        print()
        return execution

    def get_repo_info(self, execution, task, venv_folder):
        # type: (ExecutionInfo, tasks_api.Task, str) -> Tuple[str, Optional[VCS], Optional[RepoInfo]]
        literal_script = LiteralScriptManager(venv_folder)
        has_repository = bool(execution.repository)
        is_literal_script = literal_script.is_literal_script(task)
        if not has_repository and not is_literal_script:
            raise CommandFailedError(
                "Can not run task without repository or literal script in `script.diff`"
            )
        repo_info = None
        directory = None
        vcs = None
        if has_repository:
            vcs, repo_info = self._get_repo_info(execution, task, venv_folder)
            directory = Path(repo_info.root, execution.working_dir or ".")

            for pair in attr.asdict(repo_info).items():
                print("{}: {}".format(*pair))
            if not is_literal_script:
                self.apply_diff(
                    task=task, vcs=vcs, execution_info=execution, repo_info=repo_info
                )
        if is_literal_script:
            self.log.info("found literal script in `script.diff`")
            directory, script = literal_script.create_notebook_file(
                task, execution, repo_info
            )
            execution.entry_point = script
            if not has_repository:
                return directory, None, None
        else:
            # in case of no literal script, there is not difference between empty working dir and `.`
            execution.working_dir = execution.working_dir or "."
        if not directory:
            assert False, "unreachable code"
        return directory, vcs, repo_info

    def _get_repo_info(self, execution, task, venv_folder):
        try:
            self._session.config.put("agent.standalone_mode", self._standalone_mode)
            vcs, repo_info = clone_repository_cached(
                session=self._session,
                execution=execution,
                destination=Path(venv_folder) / WORKING_REPOSITORY_DIR,
            )
        except CommandFailedError:
            raise
        except Exception as ex:
            print('Repository cloning failed: {}'.format(ex))
            task.failed(
                status_reason="failed cloning repository",
                status_message=self._task_status_change_message,
            )
            if self._session.debug_mode:
                raise
            raise CommandFailedError(
                "Failed cloning repository. \n"
                "1) Make sure you pushed the requested commit:\n{}\n"
                "2) Check if remote-worker has valid credentials [see worker configuration file]".format(
                    str(execution).replace('ExecutionInfo', '', 1).replace('version_num', 'commit_id', 1))
            )
        return vcs, repo_info

    def handle_task_termination(self, task_id, exit_code, stop_reason):
        # type: (Text, int, TaskStopReason) -> None
        try:
            if stop_reason == TaskStopReason.stopped:
                self.log("Stopping - tasks.stop was called for task")
                self._session.api_client.tasks.stopped(
                    task=task_id,
                    status_reason="task was stopped by tasks.stop",
                    status_message=self._task_status_change_message,
                )

            elif stop_reason == TaskStopReason.status_changed:
                try:
                    task_status = get_task(
                        self._session, task_id, only_fields=["status"]
                    ).status
                    self.log(
                        "Task status changed unexpectedly (status: {}), "
                        "terminating task process.".format(task_status)
                    )
                except Exception as ex:
                    self.log(
                        "Task status changed unexpectedly. Was not able to obtain the current status: "
                        "{}: {}".format(type(ex), ex)
                    )
                    self.log_traceback(ex)

            elif stop_reason == TaskStopReason.reset:
                self.log("Task was reset unexpectedly")

            elif stop_reason == TaskStopReason.no_stop:
                self.handle_task_process_termination(task_id, exit_code)
            else:
                self.log(
                    "INTERNAL ERROR: unidentified task stop reason: {}".format(
                        stop_reason
                    )
                )

        except Exception as e:
            # task probably set its own status
            self.log(
                "Warning: could not update task id '{}' status. Task exit code {}".format(
                    task_id, exit_code
                )
            )
            self.log_traceback(e)

    def handle_task_process_termination(self, task_id, exit_code):
        # type: (Text, int) -> None
        self.log("Task process terminated")

        if exit_code == COMMAND_SUCCESS:
            self.log("Task success: completing")
            self._session.api_client.tasks.completed(
                task=task_id,
                status_reason="worker execution done",
                status_message=self._task_status_change_message,
            )
        elif exit_code == ExitStatus.interrupted:
            self.log("Task interrupted: stopping")
            self._session.api_client.tasks.stopped(
                task=task_id,
                status_reason="user abort",
                status_message=self._task_status_change_message,
            )
        else:
            self.log("Task failure: setting status to 'failed'")
            self._session.api_client.tasks.failed(
                task=task_id,
                status_reason="worker execution exit code {}".format(exit_code),
                status_message=self._task_status_change_message,
            )

    def freeze_task_environment(self, task_id=None, requirements_manager=None):
        try:
            freeze = self.package_api.freeze()
        except Exception as e:
            print("Could not freeze installed packages")
            self.log_traceback(e)
            return None

        if requirements_manager:
            freeze = requirements_manager.replace_back(freeze)

        if not task_id:
            return freeze

        # get original requirements and update with the new frozen requirements
        try:
            current_task = get_task(self._session, task_id, only_fields=["script.requirements"])
            requirements = current_task.script.requirements
            requirements.update(freeze)
        except Exception:
            requirements = freeze

        request = tasks_api.SetRequirementsRequest(task=task_id, requirements=requirements)
        try:
            self._session.send_api(request)
        except APIError as e:
            print("Could not set task requirements")
            self.log_traceback(e)
        return freeze

    def _install_poetry_requirements(self, repo_info):
        # type: (Optional[RepoInfo]) -> Optional[PoetryAPI]
        if not repo_info:
            return None
        try:
            if not self.poetry.enabled:
                return None
            self.poetry.initialize(cwd=repo_info.root)
            api = self.poetry.get_api(repo_info.root)
            if api.enabled:
                print('Poetry Enabled: Ignoring requested python packages, using repository poetry lock file!')
                api.install()
                return api
        except Exception:
            self.log.error("failed installing poetry requirements:")
        return None

    def install_requirements(
        self, execution, repo_info, requirements_manager, cached_requirements=None, cwd=None, package_api=None
    ):
        return self.install_requirements_for_package_api(execution, repo_info, requirements_manager,
                                                         cached_requirements=cached_requirements, cwd=cwd,
                                                         package_api=package_api if package_api else self.package_api)

    def install_requirements_for_package_api(
        self, execution, repo_info, requirements_manager, cached_requirements=None, cwd=None, package_api=None,
    ):
        # type: (ExecutionInfo, RepoInfo, RequirementsManager, Optional[dict], Optional[str], Optional[Any]) -> None
        """
        :summary: Install requirements for task script using pip.
        :description: A file named "requirements.txt" is looked for in each containing folder between the
                      repository root and the directory containing the script.
                      For each such file, CUDA based packages are looked for and replaced if necessary.
        :param execution: task execution information
        :param repo_info: repository information
        :param requirements_manager: requirements manager for task
        :param cached_requirements: cached requirements from previous run
        :param cwd: current folder
        :param package_api: package_api to be used when installing requirements
         """
        if package_api:
            package_api.cwd = cwd
        api = self._install_poetry_requirements(repo_info)
        if api:
            package_api = api
            return

        package_api.upgrade_pip()
        package_api.set_selected_package_manager()
        # always install cython,
        # if we have a specific version in the requirements,
        # the PriorityPackageRequirement(SimpleSubstitution) will reinstall cython with the specific version
        if not self.is_conda:
            package_api.out_of_scope_install_package('Cython')

        cached_requirements_failed = False
        if cached_requirements and ('pip' in cached_requirements or 'conda' in cached_requirements):
            self.log("Found task requirements section, trying to install")
            try:
                package_api.load_requirements(cached_requirements)
            except Exception as e:
                self.log_traceback(e)
                cached_requirements_failed = True
                raise ValueError("Could not install task requirements!\n{}".format(e))
            else:
                self.log("task requirements installation passed")
                return

        if not repo_info:
            if cached_requirements_failed:
                raise ValueError("Could not install task requirements!")
            self.log("no repository to install requirements from")
            return

        repo_requirements_installed = False
        for parent in reversed(
            (Path(execution.working_dir) / execution.entry_point).parents
        ):
            requirements_file = (
                repo_info.root / parent / "requirements.txt"
            )  # type: Path
            if not requirements_file.is_file():
                continue
            print("Trying pip install: {}".format(requirements_file))
            requirements_text = requirements_file.read_text()
            new_requirements = requirements_manager.replace(requirements_text)

            temp_file = None
            try:
                with self.named_temporary_file(
                    mode="w", prefix="requirements_", suffix=".txt", delete=False
                ) as temp_file:
                    temp_file.write(new_requirements)
                    temp_file.flush()
                # close the file before reading in install_from_file for Windows compatibility
                package_api.install_from_file(temp_file.name)
            except Exception as e:
                print('ERROR: Failed installing requirements.txt:\n{}'.format(requirements_text))
                raise e
            finally:
                if self._session.debug_mode and temp_file:
                    rm_file(temp_file.name)
            # call post installation callback
            requirements_manager.post_install(self._session)
            # mark as successful installation
            repo_requirements_installed = True

        # if we reached here without installing anything, and
        # we failed installing from cached requirements, them this is an error
        if cached_requirements_failed and not repo_requirements_installed:
            raise ValueError("Failed installing task requirements and repository requirements")

    def named_temporary_file(self, *args, **kwargs):
        kwargs.setdefault("delete", not self._session.debug_mode)
        return named_temporary_file(*args, **kwargs)

    def list(self, **_):
        # display all registered working machines
        workers = self.get("get_all", last_seen=0)
        print_parameters(workers)

    def _register(self, queues=()):
        self.queues = queues
        try:
            self.get("register", worker=self.worker_id, queues=queues)
            # If we got here - we've registered
            self._worker_registered = True
        # except APIError as error:
        #     if error.codes != WORKER_ALREADY_REGISTERED:
        #         raise
        # except ValueError:
        #     raise ValueError("Worker cannot register itself with backend service")
        except Exception as e:
            self.log("Worker failed registering itself with backend service: {}".format(e))

    def _unregister(self, queues=()):
        self.queues = ()
        try:
            self.get("unregister", worker=self.worker_id, queues=queues)
            self._worker_registered = False
        except Exception as e:
            self.log("Worker failed unregistering itself with backend service: {}".format(e))

    def log_traceback(self, err):
        if isinstance(err, APIError):
            tb = err.get_traceback()
            if tb:
                print("Server traceback:\n{}".format(tb))
        if self._session.debug_mode:
            self.log(traceback.format_exc())

    def debug(self, message):
        if self._session.debug_mode:
            print("trains_agent: {}".format(message))

    def find_python_executable_for_version(self, config_version):
        # type: (Text) -> Tuple[Text, Text, Text]
        """
        Find python executable with version ``config_version``.
        The search starts with the executable named ``python<config_version>``.
        If not found in the path, increasingly major components of the semantic version are dropped
        with the last executable to be searched being a plain ``python``.
        A result is returned only if ``python --version``'s output matches config_version.
        For example: if config_version=3.6.5, the search order is:
        1. python3.6.5
        2. python3.6
        3. python3
        4. python
        if config_version=3.6, the search starts with ``python3.6`` and so on.
        :return: 3-tuple:
            1. Python executable version as reported by itself with ``--version``
            2. The version suffix of the executable name (e.g. ``python3`` -> ``3``)
            3. The executable name itself (e.g. ``python3``)
        """

        def suffixes(it):
            it = list(it)
            for i in range(len(it) + 1):
                yield it[:i]

        def rreplace(s, old, new, count):
            return (s[::-1].replace(old[::-1], new[::-1], count))[::-1]

        if is_windows_platform():
            python_executables = [
                (version, config_version if os.path.sep in config_version else 'python{}'.format(version))
                for version in map(
                    ".".join, reversed(list(suffixes(
                        rreplace(
                            rreplace(config_version.split(os.path.sep)[-1].lower(), 'python', '', 1),
                            '.exe', '', 1).split("."))))
                )
            ]
        else:
            python_executables = [
                (version, config_version if os.path.sep in config_version else 'python{}'.format(version))
                for version in map(
                    ".".join, reversed(list(suffixes(
                        rreplace(config_version.split(os.path.sep)[-1], 'python', '', 1).split("."))))
                )
            ]

        for version, executable in python_executables:
            self.log.debug("Searching for {}".format(executable))
            if find_executable(executable):
                try:
                    output = Argv(executable, "--version").get_output(
                        stderr=subprocess.STDOUT
                    )
                except subprocess.CalledProcessError as ex:
                    self.log.warning("error getting %s version: %s", executable, ex)
                    continue
                match = re.search(
                    r"Python ({}(?:\.\d+)*)".format(
                        r"\d+" if not config_version or os.path.sep in config_version else config_version), output
                )
                if match:
                    self.log.debug("Found: {}".format(executable))
                    return match.group(1), version or '.'.join(match.group(1).split('.')[:2]), executable
        raise CommandFailedError(
            "Python executable with version {!r} defined in configuration file, "
            "key 'agent.default_python', not found in path, tried: {}".format(
                config_version, list(zip(*python_executables))[1]
            )
        )

    def install_virtualenv(
            self, venv_dir=None, requested_python_version=None, standalone_mode=False, execution_info=None):
        # type: (str, str, bool, ExecutionInfo) -> Tuple[Path, RequirementsManager]
        """
        Install a new python virtual environment, removing the old one if exists
        :return: virtualenv directory and requirements manager to use with task
        """
        requested_python_version = requested_python_version or \
                                   Text(self._session.config.get("agent.python_binary", None)) or \
                                   Text(self._session.config.get("agent.default_python", None))
        if self.is_conda:
            executable_version_suffix = \
                requested_python_version[max(requested_python_version.find('python'), 0):].replace('python', '')
            executable_name = 'python'
        else:
            try:
                executable_version, executable_version_suffix, executable_name = \
                    self.find_python_executable_for_version(requested_python_version)
            except Exception:
                def_python_version = Text(self._session.config.get("agent.python_binary", None)) or \
                                     Text(self._session.config.get("agent.default_python", None))
                print('Warning: could not locate requested Python version {}, reverting to version {}'.format(
                    requested_python_version, def_python_version))
                executable_version, executable_version_suffix, executable_name = \
                    self.find_python_executable_for_version(def_python_version)
            self._session.config.put("agent.default_python", executable_version)
            self._session.config.put("agent.python_binary", executable_name)

        venv_dir = Path(venv_dir) if venv_dir else \
            Path(self._session.config["agent.venvs_dir"], executable_version_suffix)

        first_time = not standalone_mode and (
            is_windows_platform()
            or self.is_conda
            or not venv_dir.is_dir()
            or not self.is_venv_update
        )

        requirements_manager = self._get_requirements_manager(
            base_interpreter=executable_name
        )

        if not standalone_mode:
            rm_tree(normalize_path(venv_dir, WORKING_REPOSITORY_DIR))
        package_manager_params = dict(
            session=self._session,
            python=executable_version_suffix if self.is_conda else executable_name,
            path=venv_dir,
            requirements_manager=requirements_manager,
            execution_info=execution_info,
        )

        global_package_manager_params = dict(
            interpreter=executable_name,
            session=self._session,
        )

        if not self.is_conda and standalone_mode:
            # pip with standalone mode
            get_pip = partial(VirtualenvPip, **package_manager_params)
            self.package_api = get_pip()
            self.global_package_api = SystemPip(**global_package_manager_params)
        elif not self.is_conda:
            if self.is_venv_update:
                self.package_api = VenvUpdateAPI(
                    url=self._session.config["agent.venv_update.url"] or DEFAULT_VENV_UPDATE_URL,
                    **package_manager_params
                )
            else:
                self.package_api = VirtualenvPip(**package_manager_params)
            if first_time:
                self.package_api.remove()
                self.package_api.create()
            self.global_package_api = SystemPip(**global_package_manager_params)
        elif standalone_mode:
            # conda with standalone mode
            get_conda = partial(CondaAPI, **package_manager_params)
            self.package_api = get_conda()
        else:

            get_conda = partial(CondaAPI, **package_manager_params)
            self.package_api = get_conda()
            # no support for reusing Conda environments
            self.package_api.remove()

            if venv_dir.exists():
                timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
                new_venv_folder = venv_dir.with_name(
                    "{}_{}".format(venv_dir.name, timestamp)
                )
                self.warning(
                    'Path "{}" exists, using "{}" instead'.format(
                        venv_dir, new_venv_folder
                    )
                )
                venv_dir = new_venv_folder
                self.package_api = get_conda(path=venv_dir)

            self.package_api.create()

        return venv_dir, requirements_manager

    def parse_requirements(self, reqs_file=None, overrides=None):
        os = None
        session = self._session
        if overrides:
            overrides = ConfigFactory.parse_string("\n".join(overrides))
            os = overrides.pop("os", None)
            ConfigTree.merge_configs(session.config, overrides)
        if reqs_file:
            contents = Path(reqs_file).read_text()
        else:
            contents = ensure_text(sys.stdin.read())
        session.finalize_config(session.config)
        requirements_manager = self._get_requirements_manager(os_override=os)
        requirements_manager.translator.enabled = False
        print(requirements_manager.replace(contents))

    def get_docker_config_cmd(self, docker_args):
        def docker_cmd_functor(default_kwargs, temp_config, **kwargs):
            # Make sure we have created the configuration file for the executor
            if not self.dump_config(temp_config):
                self.log.warning('Could not update docker configuration file {}'.format(self.temp_config_path))
            args = deepcopy(default_kwargs)
            args.update(kwargs)
            return self._get_docker_cmd(**args)

        docker_image = str(os.environ.get("TRAINS_DOCKER_IMAGE") or
                           self._session.config.get("agent.default_docker.image", "nvidia/cuda")) \
            if not docker_args else docker_args[0]
        docker_arguments = docker_image.split(' ') if docker_image else []
        if len(docker_arguments) > 1:
            docker_image = docker_arguments[0]
            docker_arguments = docker_arguments[1:]
        else:
            docker_arguments = self._session.config.get("agent.default_docker.arguments", None) or []
            if isinstance(docker_arguments, six.string_types):
                docker_arguments = [docker_arguments]
        python_version = '3'
        if not python_version.startswith('python'):
            python_version = 'python'+python_version
        print("Running in Docker {} mode (v19.03 and above) - using default docker image: {} running {}\n".format(
            '*standalone*' if self._standalone_mode else '', docker_image, python_version))
        temp_config = deepcopy(self._session.config)
        mounted_cache_dir = self._docker_fixed_user_cache  # '/root/.trains/cache'
        mounted_pip_dl_dir = '/root/.trains/pip-download-cache'
        mounted_vcs_cache = '/root/.trains/vcs-cache'
        mounted_venv_dir = '/root/.trains/venvs-builds'
        host_cache = Path(os.path.expandvars(self._session.config["sdk.storage.cache.default_base_dir"])).expanduser().as_posix()
        host_pip_dl = Path(os.path.expandvars(self._session.config["agent.pip_download_cache.path"])).expanduser().as_posix()
        host_vcs_cache = Path(os.path.expandvars(self._session.config["agent.vcs_cache.path"])).expanduser().as_posix()
        temp_config.put("sdk.storage.cache.default_base_dir", mounted_cache_dir)
        temp_config.put("agent.pip_download_cache.path", mounted_pip_dl_dir)
        temp_config.put("agent.vcs_cache.path", mounted_vcs_cache)
        temp_config.put("agent.package_manager.system_site_packages", True)
        temp_config.put("agent.package_manager.conda_env_as_base_docker", False)
        temp_config.put("agent.default_python", "")
        temp_config.put("agent.python_binary", "")
        temp_config.put("agent.cuda_version", "")
        temp_config.put("agent.cudnn_version", "")
        temp_config.put("agent.venvs_dir", mounted_venv_dir)
        temp_config.put("agent.git_user", (ENV_AGENT_GIT_USER.get() or self._session.config.get("agent.git_user", None)))
        temp_config.put("agent.git_pass", (ENV_AGENT_GIT_PASS.get() or self._session.config.get("agent.git_pass", None)))

        host_apt_cache = Path(os.path.expandvars(self._session.config.get(
            "agent.docker_apt_cache", '~/.trains/apt-cache'))).expanduser().as_posix()
        host_pip_cache = Path(os.path.expandvars(self._session.config.get(
            "agent.docker_pip_cache", '~/.trains/pip-cache'))).expanduser().as_posix()
        host_ssh_cache = mkdtemp(prefix='trains_agent.ssh.')
        self._temp_cleanup_list.append(host_ssh_cache)

        # make sure all folders are valid
        Path(host_apt_cache).mkdir(parents=True, exist_ok=True)
        Path(host_pip_cache).mkdir(parents=True, exist_ok=True)
        Path(host_cache).mkdir(parents=True, exist_ok=True)
        Path(host_pip_dl).mkdir(parents=True, exist_ok=True)
        Path(host_vcs_cache).mkdir(parents=True, exist_ok=True)
        Path(host_ssh_cache).mkdir(parents=True, exist_ok=True)

        # copy the .ssh folder to a temp folder, to be mapped into docker
        try:
            src = Path(host_ssh_cache)
            if src.is_dir():
                src.rmdir()
            shutil.copytree(Path('~/.ssh').expanduser().as_posix(), host_ssh_cache)
        except Exception:
            host_ssh_cache = None
            self.log.warning('Failed creating temporary copy of ~/.ssh for git credential')
            pass

        # check if the .git credentials exist:
        try:
            host_git_credentials = [
                f.as_posix() for f in [Path('~/.git-credentials').expanduser(), Path('~/.gitconfig').expanduser()]
                if f.is_file()]
        except Exception:
            host_git_credentials = None

        # store docker arguments
        self._docker_image = docker_image
        self._docker_arguments = docker_arguments

        extra_shell_script_str = ""
        if self._extra_shell_script:
            cmds = self._extra_shell_script
            if not isinstance(cmds, (list, tuple)):
                cmds = [cmds]
            extra_shell_script_str = " ; ".join(map(str, cmds)) + " ; "

        bash_script = self._session.config.get("agent.docker_init_bash_script", None)
        preprocess_bash_script = self._session.config.get("agent.docker_preprocess_bash_script", None)

        self.temp_config_path = self.temp_config_path or safe_mkstemp(
            suffix=".cfg", prefix=".trains_agent.", text=True, name_only=True
        )

        docker_cmd = dict(worker_id=self.worker_id,
                          # docker_image=docker_image,
                          # docker_arguments=docker_arguments,
                          extra_docker_arguments=self._extra_docker_arguments,
                          extra_shell_script=extra_shell_script_str,
                          python_version=python_version, conf_file=self.temp_config_path,
                          host_apt_cache=host_apt_cache,
                          host_pip_cache=host_pip_cache,
                          host_ssh_cache=host_ssh_cache, host_git_credentials=host_git_credentials,
                          host_cache=host_cache, mounted_cache=mounted_cache_dir,
                          host_pip_dl=host_pip_dl, mounted_pip_dl=mounted_pip_dl_dir,
                          host_vcs_cache=host_vcs_cache, mounted_vcs_cache=mounted_vcs_cache,
                          standalone_mode=self._standalone_mode,
                          force_current_version=self._force_current_version,
                          bash_script=bash_script,
                          preprocess_bash_script=preprocess_bash_script,
                          )
        return temp_config, partial(docker_cmd_functor, docker_cmd, temp_config)

    @staticmethod
    def _get_docker_cmd(worker_id, docker_image, docker_arguments,
                        python_version, conf_file,
                        host_apt_cache,
                        host_pip_cache,
                        host_ssh_cache,
                        host_cache, mounted_cache,
                        host_pip_dl, mounted_pip_dl,
                        host_vcs_cache, mounted_vcs_cache,
                        standalone_mode=False, extra_docker_arguments=None, extra_shell_script=None,
                        force_current_version=None, host_git_credentials=None,
                        bash_script=None, preprocess_bash_script=None):
        docker = 'docker'

        base_cmd = [docker, 'run', '-t']
        update_scheme = ""
        dockers_nvidia_visible_devices = 'all'
        gpu_devices = os.environ.get('NVIDIA_VISIBLE_DEVICES', None)
        if gpu_devices is None or gpu_devices.lower().strip() == 'all':
            if os.environ.get('TRAINS_DOCKER_SKIP_GPUS_FLAG', None):
                dockers_nvidia_visible_devices = os.environ.get('NVIDIA_VISIBLE_DEVICES') or \
                                                 dockers_nvidia_visible_devices
            else:
                base_cmd += ['--gpus', 'all', ]
        elif gpu_devices.strip() and gpu_devices.strip() != 'none':
            if os.environ.get('TRAINS_DOCKER_SKIP_GPUS_FLAG', None):
                dockers_nvidia_visible_devices = gpu_devices
            else:
                base_cmd += ['--gpus', 'device='+gpu_devices, ]
            # We are using --gpu, so we should not pass NVIDIA_VISIBLE_DEVICES, I think.
            # base_cmd += ['-e', 'NVIDIA_VISIBLE_DEVICES=' + gpu_devices, ]
        elif gpu_devices.strip() == 'none':
            dockers_nvidia_visible_devices = gpu_devices

        if docker_arguments:
            docker_arguments = list(docker_arguments) \
                if isinstance(docker_arguments, (list, tuple)) else [docker_arguments]
            base_cmd += [a for a in docker_arguments if a]

        if extra_docker_arguments:
            extra_docker_arguments = [extra_docker_arguments] \
                if isinstance(extra_docker_arguments, six.string_types) else extra_docker_arguments
            base_cmd += [str(a) for a in extra_docker_arguments if a]

        # check if running inside a kubernetes
        if ENV_DOCKER_HOST_MOUNT.get() or (os.environ.get('KUBERNETES_SERVICE_HOST') and os.environ.get('KUBERNETES_PORT')):
            # map network to sibling docker, unless we have other network argument
            if not any(a.strip().startswith('--network') for a in base_cmd):
                try:
                    network_mode = get_bash_output(
                        'docker inspect --format=\'{{.HostConfig.NetworkMode}}\' $(basename $(cat /proc/1/cpuset))')
                    base_cmd += ['--network', network_mode]
                except:
                    pass
            base_cmd += ['-e', 'NVIDIA_VISIBLE_DEVICES={}'.format(dockers_nvidia_visible_devices)]

        # check if we need to map host folders
        if ENV_DOCKER_HOST_MOUNT.get():
            # expect TRAINS_AGENT_K8S_HOST_MOUNT = '/mnt/host/data:/root/.trains'
            k8s_node_mnt, _, k8s_pod_mnt = ENV_DOCKER_HOST_MOUNT.get().partition(':')
            # search and replace all the host folders with the k8s
            host_mounts = [host_apt_cache, host_pip_cache, host_pip_dl, host_cache, host_vcs_cache]
            for i, m in enumerate(host_mounts):
                if k8s_pod_mnt not in m:
                    print('Warning: K8S mount missing, ignoring cached folder {}'.format(m))
                    host_mounts[i] = None
                else:
                    host_mounts[i] = m.replace(k8s_pod_mnt, k8s_node_mnt, 1)
            host_apt_cache, host_pip_cache, host_pip_dl, host_cache, host_vcs_cache = host_mounts

            # copy the configuration file into the mounted folder
            new_conf_file = os.path.join(k8s_pod_mnt, '.trains_agent.{}.cfg'.format(quote(worker_id, safe="")))
            try:
                rm_tree(new_conf_file)
                rm_file(new_conf_file)
                shutil.copy(conf_file, new_conf_file)
                conf_file = new_conf_file.replace(k8s_pod_mnt, k8s_node_mnt)
            except Exception:
                raise ValueError('Error: could not copy configuration file into: {}'.format(new_conf_file))

            if host_ssh_cache:
                new_ssh_cache = os.path.join(k8s_pod_mnt, '.trains_agent.{}.ssh'.format(quote(worker_id, safe="")))
                try:
                    rm_tree(new_ssh_cache)
                    shutil.copytree(host_ssh_cache, new_ssh_cache)
                    host_ssh_cache = new_ssh_cache.replace(k8s_pod_mnt, k8s_node_mnt)
                except Exception:
                    raise ValueError('Error: could not copy .ssh directory into: {}'.format(new_ssh_cache))

        base_cmd += ['-e', 'TRAINS_WORKER_ID='+worker_id, ]

        # if we are running a RC version, install the same version in the docker
        # because the default latest, will be a release version (not RC)
        specify_version = ''
        try:
            from trains_agent.version import __version__
            _version_parts = __version__.split('.')
            if force_current_version or 'rc' in _version_parts[-1].lower() or 'rc' in _version_parts[-2].lower():
                specify_version = '=={}'.format(__version__)
        except:
            pass

        if os.environ.get('FORCE_LOCAL_TRAINS_AGENT_WHEEL'):
            local_wheel = os.path.expanduser(os.environ.get('FORCE_LOCAL_TRAINS_AGENT_WHEEL'))
            docker_wheel = str(Path('/tmp') / Path(local_wheel).name)
            base_cmd += ['-v', local_wheel + ':' + docker_wheel]
            trains_agent_wheel = '\"{}\"'.format(docker_wheel)
        else:
            # trains-agent{specify_version}
            trains_agent_wheel = 'trains-agent{specify_version}'.format(specify_version=specify_version)

        if not standalone_mode:
            if not bash_script:
                # Find the highest python version installed, or install from apt-get
                # python+pip is the requirement to match
                bash_script = [
                    "echo 'Binary::apt::APT::Keep-Downloaded-Packages \"true\";' > /etc/apt/apt.conf.d/docker-clean",
                    "chown -R root /root/.cache/pip",
                    "export DEBIAN_FRONTEND=noninteractive",
                    "apt-get update",
                    "apt-get install -y git libsm6 libxext6 libxrender-dev libglib2.0-0",
                    "declare LOCAL_PYTHON",
                    "for i in {{10..5}}; do which {python_single_digit}.$i && " +
                    "{python_single_digit}.$i -m pip --version && " +
                    "export LOCAL_PYTHON=$(which {python_single_digit}.$i) && break ; done",
                    "[ ! -z $LOCAL_PYTHON ] || apt-get install -y {python_single_digit}-pip",
                ]

            if preprocess_bash_script:
                bash_script = preprocess_bash_script + bash_script

            docker_bash_script = " ; ".join(bash_script) if not isinstance(bash_script, str) else bash_script

            # make sure that if we do not have $LOCAL_PYTHON defined
            # we set it to python3
            update_scheme += (
                    docker_bash_script + " ; " +
                    "[ ! -z $LOCAL_PYTHON ] || export LOCAL_PYTHON={python} ; " +
                    "$LOCAL_PYTHON -m pip install -U \"pip{pip_version}\" ; " +
                    "$LOCAL_PYTHON -m pip install -U {trains_agent_wheel} ; ").format(
                python_single_digit=python_version.split('.')[0],
                python=python_version, pip_version=PackageManager.get_pip_version(),
                trains_agent_wheel=trains_agent_wheel)

        if host_git_credentials:
            for git_credentials in host_git_credentials:
                base_cmd += ['-v', '{}:/root/{}'.format(git_credentials, Path(git_credentials).name)]

        base_cmd += (
            ['-v', conf_file+':'+DOCKER_ROOT_CONF_FILE] +
            (['-v', host_ssh_cache+':/root/.ssh'] if host_ssh_cache else []) +
            (['-v', host_apt_cache+':/var/cache/apt/archives'] if host_apt_cache else []) +
            (['-v', host_pip_cache+':/root/.cache/pip'] if host_pip_cache else []) +
            (['-v', host_pip_dl+':'+mounted_pip_dl] if host_pip_dl else []) +
            (['-v', host_cache+':'+mounted_cache] if host_cache else []) +
            (['-v', host_vcs_cache+':'+mounted_vcs_cache] if host_vcs_cache else []) +
            ['--rm', docker_image, 'bash', '-c',
                update_scheme +
                extra_shell_script +
                "cp {} {} ; ".format(DOCKER_ROOT_CONF_FILE, DOCKER_DEFAULT_CONF_FILE) +
                "NVIDIA_VISIBLE_DEVICES={nv_visible} $LOCAL_PYTHON -u -m trains_agent ".format(
                    nv_visible=dockers_nvidia_visible_devices, python=python_version)
             ])

        return base_cmd

    def _run_as_user_patch(self, command, trains_conf, script_dir, venv_folder, sdk_cache_folder, user_uid):
        class RunasArgv(Argv):
            def __init__(self, *args):
                super(RunasArgv, self).__init__(*args)
                self.uid = 0
                self.gid = 0

            def call_subprocess(self, func, censor_password=False, *args, **kwargs):
                self._log.debug("running: %s: %s", func.__name__, list(self))
                with self.normalize_exception(censor_password):
                    return func(list(self), *args, preexec_fn=self._change_uid, **kwargs)

            def set_uid(self, user_uid, user_gid):
                from pwd import getpwnam
                try:
                    self.uid = getpwnam(user_uid).pw_uid
                    self.gid = getpwnam(user_gid).pw_gid
                except Exception:
                    raise ValueError("Could not find requested user uid={} gid={}".format(user_uid, user_gid))

            def _change_uid(self):
                os.setgid(self.gid)
                os.setuid(self.uid)

        # create a home folder for our user
        trains_agent_home = self._run_as_user_home + '{}'.format('.'+str(Singleton.get_slot()) if Singleton.get_slot() else '')
        try:
            home_folder = self._run_as_user_home
            rm_tree(home_folder)
            Path(home_folder).mkdir(parents=True, exist_ok=True)
        except:
            home_folder = os.path.join('/home', self._run_as_user_home)
            rm_tree(home_folder)
            Path(home_folder).mkdir(parents=True, exist_ok=True)

        # move our entire venv into the new home
        venv_folder = venv_folder.as_posix()
        if not venv_folder.endswith(os.path.sep):
            venv_folder += os.path.sep
        new_venv_folder = os.path.join(home_folder, 'venv/')
        shutil.move(venv_folder, new_venv_folder)
        # allow everyone to access it
        for f in Path(new_venv_folder).rglob('*'):
            try:
                f.chmod(0o0777)
            except:
                pass
        # make sure we will be able to access the cache folder (we assume we have the ability change mod)
        if sdk_cache_folder:
            sdk_cache_folder = Path(os.path.expandvars(sdk_cache_folder)).expanduser().absolute()
            try:
                sdk_cache_folder.chmod(0o0777)
            except:
                pass
            for f in sdk_cache_folder.rglob('*'):
                try:
                    f.chmod(0o0777)
                except:
                    pass

        # make sure we could access the trains_conf file
        try:
            user_trains_conf = os.path.join(home_folder, 'trains.conf')
            shutil.copy(trains_conf, user_trains_conf)
            Path(user_trains_conf).chmod(0o0777)
        except:
            user_trains_conf = trains_conf

        # patch venv folder to new location
        script_dir = script_dir.replace(venv_folder, new_venv_folder)
        # New command line execution
        command = RunasArgv('bash', '-c', 'HOME=\"{}\" PATH=\"{}\" PYTHONPATH=\"{}\" TRAINS_CONFIG_FILE={} {}'.format(
            home_folder,
            os.environ.get('PATH', '').replace(venv_folder, new_venv_folder),
            os.environ.get('PYTHONPATH', '').replace(venv_folder, new_venv_folder),
            user_trains_conf,
            command.serialize().replace(venv_folder, new_venv_folder)))
        command.set_uid(user_uid=user_uid, user_gid=user_uid)

        return command, script_dir

    def _kill_daemon(self):
        worker_id, worker_name = self._generate_worker_id_name()
        # Iterate over all running process
        for pid, uid, slot, file in sorted(Singleton.get_running_pids(), key=lambda x: x[1] or ''):
            # wither we have a match for the worker_id or we just pick the first one
            if pid >= 0 and uid is not None and (
                    (worker_id and uid == worker_id) or
                    (not worker_id and uid.startswith('{}:'.format(worker_name)))):
                # this is us kill it
                print('Terminating trains-agent worker_id={} pid={}'.format(uid, pid))
                if not terminate_process(pid, timeout=10):
                    error('Could not terminate process pid={}'.format(pid))
                return True
        print('Could not find a running trains-agent instance with worker_name={} worker_id={}'.format(
            worker_name, worker_id))
        return False

    def _singleton(self):
        # ensure singleton
        worker_id, worker_name = self._generate_worker_id_name()

        # if we are running in services mode, we allow double register since
        # docker-compose will kill instances before they cleanup
        self.worker_id, worker_slot = Singleton.register_instance(
            unique_worker_id=worker_id, worker_name=worker_name, api_client=self._session.api_client,
            allow_double=bool(ENV_DOCKER_HOST_MOUNT.get())  # and bool(self._services_mode),
        )

        if self.worker_id is None:
            error('Instance with the same WORKER_ID [{}] is already running'.format(worker_id))
            exit(1)
        # update folders based on free slot
        self._session.create_cache_folders(slot_index=worker_slot)

    def _generate_worker_id_name(self):
        worker_id = self._session.config["agent.worker_id"]
        worker_name = self._session.config["agent.worker_name"]
        if not worker_id and os.environ.get('NVIDIA_VISIBLE_DEVICES') is not None:
            nvidia_visible_devices = os.environ.get('NVIDIA_VISIBLE_DEVICES')
            if nvidia_visible_devices and nvidia_visible_devices.lower() != 'none':
                worker_id = '{}:gpu{}'.format(worker_name, nvidia_visible_devices)
            elif nvidia_visible_devices == '':
                pass
            else:
                worker_name = '{}:cpu'.format(worker_name)
        return worker_id, worker_name

    def _resolve_queue_names(self, queues, create_if_missing=False):
        if not queues:
            default_queue = self._session.send_api(queues_api.GetDefaultRequest())
            return [default_queue.id]

        queues = return_list(queues)
        if not create_if_missing:
            return [self._resolve_name(q.name, "queues") for q in queues]

        queue_ids = []
        for q in queues:
            try:
                q_id = self._resolve_name(q.name, "queues")
            except:
                self._session.send_api(queues_api.CreateRequest(name=q.name))
                q_id = self._resolve_name(q.name, "queues")
            queue_ids.append(q_id)
        return queue_ids


if __name__ == "__main__":
    pass
