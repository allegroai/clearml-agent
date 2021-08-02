import shlex
from datetime import timedelta
from distutils.util import strtobool
from enum import IntEnum
from os import getenv, environ
from typing import Text, Optional, Union, Tuple, Any

from pathlib2 import Path

import six
from clearml_agent.helper.base import normalize_path

PROGRAM_NAME = "clearml-agent"
FROM_FILE_PREFIX_CHARS = "@"

CONFIG_DIR = normalize_path("~/.clearml")
TOKEN_CACHE_FILE = normalize_path("~/.clearml.clearml_agent.tmp")

CONFIG_FILE_CANDIDATES = ["~/clearml.conf"]


def find_config_path():
    for candidate in CONFIG_FILE_CANDIDATES:
        if Path(candidate).expanduser().exists():
            return candidate
    return CONFIG_FILE_CANDIDATES[0]


CONFIG_FILE = normalize_path(find_config_path())


class EnvironmentConfig(object):

    conversions = {
        bool: lambda value: bool(strtobool(value)),
        six.text_type: lambda s: six.text_type(s).strip(),
        list: lambda s: shlex.split(s.strip()),
    }

    def __init__(self, *names, **kwargs):
        self.vars = names
        self.type = kwargs.pop("type", six.text_type)

    def pop(self):
        for k in self.vars:
            environ.pop(k, None)

    def set(self, value):
        for k in self.vars:
            environ[k] = str(value)

    def convert(self, value):
        return self.conversions.get(self.type, self.type)(value)

    def get(self, key=False):  # type: (bool) -> Optional[Union[Any, Tuple[Text, Any]]]
        for name in self.vars:
            value = getenv(name)
            if value:
                value = self.convert(value)
                if key:
                    return name, value
                return value
        return None


ENV_AGENT_SECRET_KEY = EnvironmentConfig("CLEARML_API_SECRET_KEY", "TRAINS_API_SECRET_KEY")
ENV_AGENT_AUTH_TOKEN = EnvironmentConfig("CLEARML_AUTH_TOKEN")
ENV_AWS_SECRET_KEY = EnvironmentConfig("AWS_SECRET_ACCESS_KEY")
ENV_AZURE_ACCOUNT_KEY = EnvironmentConfig("AZURE_STORAGE_KEY")

ENVIRONMENT_CONFIG = {
    "api.api_server": EnvironmentConfig("CLEARML_API_HOST", "TRAINS_API_HOST", ),
    "api.files_server": EnvironmentConfig("CLEARML_FILES_HOST", "TRAINS_FILES_HOST", ),
    "api.web_server": EnvironmentConfig("CLEARML_WEB_HOST", "TRAINS_WEB_HOST", ),
    "api.credentials.access_key": EnvironmentConfig(
        "CLEARML_API_ACCESS_KEY", "TRAINS_API_ACCESS_KEY",
    ),
    "api.credentials.secret_key": ENV_AGENT_SECRET_KEY,
    "agent.worker_name": EnvironmentConfig("CLEARML_WORKER_NAME", "TRAINS_WORKER_NAME", ),
    "agent.worker_id": EnvironmentConfig("CLEARML_WORKER_ID", "TRAINS_WORKER_ID", ),
    "agent.cuda_version": EnvironmentConfig(
        "CLEARML_CUDA_VERSION", "TRAINS_CUDA_VERSION", "CUDA_VERSION"
    ),
    "agent.cudnn_version": EnvironmentConfig(
        "CLEARML_CUDNN_VERSION", "TRAINS_CUDNN_VERSION", "CUDNN_VERSION"
    ),
    "agent.cpu_only": EnvironmentConfig(
        names=("CLEARML_CPU_ONLY", "TRAINS_CPU_ONLY", "CPU_ONLY"), type=bool
    ),
    "sdk.aws.s3.key": EnvironmentConfig("AWS_ACCESS_KEY_ID"),
    "sdk.aws.s3.secret": ENV_AWS_SECRET_KEY,
    "sdk.aws.s3.region": EnvironmentConfig("AWS_DEFAULT_REGION"),
    "sdk.azure.storage.containers.0": {'account_name': EnvironmentConfig("AZURE_STORAGE_ACCOUNT"),
                                       'account_key': ENV_AZURE_ACCOUNT_KEY},
    "sdk.google.storage.credentials_json": EnvironmentConfig("GOOGLE_APPLICATION_CREDENTIALS"),
}

ENVIRONMENT_SDK_PARAMS = {
    "task_id": ("CLEARML_TASK_ID", "TRAINS_TASK_ID", ),
    "config_file": ("CLEARML_CONFIG_FILE", "TRAINS_CONFIG_FILE", ),
    "log_level": ("CLEARML_LOG_LEVEL", "TRAINS_LOG_LEVEL", ),
    "log_to_backend": ("CLEARML_LOG_TASK_TO_BACKEND", "TRAINS_LOG_TASK_TO_BACKEND", ),
}

ENVIRONMENT_BACKWARD_COMPATIBLE = EnvironmentConfig(
    names=("CLEARML_AGENT_ALG_ENV", "TRAINS_AGENT_ALG_ENV"), type=bool)

VIRTUAL_ENVIRONMENT_PATH = {
    "python2": normalize_path(CONFIG_DIR, "py2venv"),
    "python3": normalize_path(CONFIG_DIR, "py3venv"),
}

DEFAULT_BASE_DIR = normalize_path(CONFIG_DIR, "data_cache")
DEFAULT_HOST = "https://demoapi.demo.clear.ml"
MAX_DATASET_SOURCES_COUNT = 50000

INVALID_WORKER_ID = (400, 1001)
WORKER_ALREADY_REGISTERED = (400, 1003)

API_VERSION = "v1.5"
TOKEN_EXPIRATION_SECONDS = int(timedelta(days=2).total_seconds())

METADATA_EXTENSION = ".json"

DEFAULT_VENV_UPDATE_URL = (
    "https://raw.githubusercontent.com/Yelp/venv-update/v3.2.4/venv_update.py"
)
WORKING_REPOSITORY_DIR = "task_repository"
DEFAULT_VCS_CACHE = normalize_path(CONFIG_DIR, "vcs-cache")
PIP_EXTRA_INDICES = [
]
DEFAULT_PIP_DOWNLOAD_CACHE = normalize_path(CONFIG_DIR, "pip-download-cache")
ENV_DOCKER_IMAGE = EnvironmentConfig('CLEARML_DOCKER_IMAGE', 'TRAINS_DOCKER_IMAGE')
ENV_WORKER_ID = EnvironmentConfig('CLEARML_WORKER_ID', 'TRAINS_WORKER_ID')
ENV_WORKER_TAGS = EnvironmentConfig('CLEARML_WORKER_TAGS')
ENV_AGENT_SKIP_PIP_VENV_INSTALL = EnvironmentConfig('CLEARML_AGENT_SKIP_PIP_VENV_INSTALL')
ENV_DOCKER_SKIP_GPUS_FLAG = EnvironmentConfig('CLEARML_DOCKER_SKIP_GPUS_FLAG', 'TRAINS_DOCKER_SKIP_GPUS_FLAG')
ENV_AGENT_GIT_USER = EnvironmentConfig('CLEARML_AGENT_GIT_USER', 'TRAINS_AGENT_GIT_USER')
ENV_AGENT_GIT_PASS = EnvironmentConfig('CLEARML_AGENT_GIT_PASS', 'TRAINS_AGENT_GIT_PASS')
ENV_AGENT_GIT_HOST = EnvironmentConfig('CLEARML_AGENT_GIT_HOST', 'TRAINS_AGENT_GIT_HOST')
ENV_AGENT_DISABLE_SSH_MOUNT = EnvironmentConfig('CLEARML_AGENT_DISABLE_SSH_MOUNT', type=bool)
ENV_SSH_AUTH_SOCK = EnvironmentConfig('SSH_AUTH_SOCK')
ENV_TASK_EXECUTE_AS_USER = EnvironmentConfig('CLEARML_AGENT_EXEC_USER', 'TRAINS_AGENT_EXEC_USER')
ENV_TASK_EXTRA_PYTHON_PATH = EnvironmentConfig('CLEARML_AGENT_EXTRA_PYTHON_PATH', 'TRAINS_AGENT_EXTRA_PYTHON_PATH')
ENV_DOCKER_HOST_MOUNT = EnvironmentConfig('CLEARML_AGENT_K8S_HOST_MOUNT', 'CLEARML_AGENT_DOCKER_HOST_MOUNT',
                                          'TRAINS_AGENT_K8S_HOST_MOUNT', 'TRAINS_AGENT_DOCKER_HOST_MOUNT')
ENV_VENV_CACHE_PATH = EnvironmentConfig('CLEARML_AGENT_VENV_CACHE_PATH')
ENV_EXTRA_DOCKER_ARGS = EnvironmentConfig('CLEARML_AGENT_EXTRA_DOCKER_ARGS', type=list)


class FileBuffering(IntEnum):
    """
    File buffering options:
    - UNSET: follows the defaults for the type of file,
        line-buffered for interactive (tty) text files and with a default chunk size otherwise
    - UNBUFFERED: no buffering at all
    - LINE_BUFFERED: per-line buffering, only valid for text files
    - values bigger than 1 indicate the size of the buffer in bytes and are not represented by the enum
    """

    UNSET = -1
    UNBUFFERED = 0
    LINE_BUFFERING = 1
