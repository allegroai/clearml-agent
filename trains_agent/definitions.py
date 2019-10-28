from datetime import timedelta
from distutils.util import strtobool
from enum import IntEnum
from os import getenv
from typing import Text, Optional, Union, Tuple, Any

from furl import furl
from pathlib2 import Path

import six
from trains_agent.helper.base import normalize_path

PROGRAM_NAME = "trains-agent"
FROM_FILE_PREFIX_CHARS = "@"

CONFIG_DIR = normalize_path("~/.trains")
TOKEN_CACHE_FILE = normalize_path("~/.trains.trains_agent.tmp")

CONFIG_FILE_CANDIDATES = ["~/trains.conf"]


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
    }

    def __init__(self, *names, **kwargs):
        self.vars = names
        self.type = kwargs.pop("type", six.text_type)

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


ENVIRONMENT_CONFIG = {
    "api.api_server": EnvironmentConfig("TRAINS_API_HOST", "ALG_API_HOST"),
    "api.credentials.access_key": EnvironmentConfig(
        "TRAINS_API_ACCESS_KEY", "ALG_API_ACCESS_KEY"
    ),
    "api.credentials.secret_key": EnvironmentConfig(
        "TRAINS_API_SECRET_KEY", "ALG_API_SECRET_KEY"
    ),
    "agent.worker_name": EnvironmentConfig("TRAINS_WORKER_NAME", "ALG_WORKER_NAME"),
    "agent.worker_id": EnvironmentConfig("TRAINS_WORKER_ID", "ALG_WORKER_ID"),
    "agent.cuda_version": EnvironmentConfig(
        "TRAINS_CUDA_VERSION", "ALG_CUDA_VERSION", "CUDA_VERSION"
    ),
    "agent.cudnn_version": EnvironmentConfig(
        "TRAINS_CUDNN_VERSION", "ALG_CUDNN_VERSION", "CUDNN_VERSION"
    ),
    "agent.cpu_only": EnvironmentConfig(
        "TRAINS_CPU_ONLY", "ALG_CPU_ONLY", "CPU_ONLY", type=bool
    ),
}

CONFIG_FILE_ENV = EnvironmentConfig("ALG_CONFIG_FILE")

ENVIRONMENT_SDK_PARAMS = {
    "task_id": ("TRAINS_TASK_ID", "ALG_TASK_ID"),
    "config_file": ("TRAINS_CONFIG_FILE", "ALG_CONFIG_FILE", "TRAINS_CONFIG_FILE"),
    "log_level": ("TRAINS_LOG_LEVEL", "ALG_LOG_LEVEL"),
    "log_to_backend": ("TRAINS_LOG_TASK_TO_BACKEND", "ALG_LOG_TASK_TO_BACKEND"),
}

VIRTUAL_ENVIRONMENT_PATH = {
    "python2": normalize_path(CONFIG_DIR, "py2venv"),
    "python3": normalize_path(CONFIG_DIR, "py3venv"),
}

DEFAULT_BASE_DIR = normalize_path(CONFIG_DIR, "data_cache")
DEFAULT_HOST = "https://demoapi.trains.allegro.ai"
MAX_DATASET_SOURCES_COUNT = 50000

INVALID_WORKER_ID = (400, 1001)
WORKER_ALREADY_REGISTERED = (400, 1003)

API_VERSION = "v1.5"
TOKEN_EXPIRATION_SECONDS = int(timedelta(days=2).total_seconds())

HTTP_HEADERS = {
    "worker": "X-Trains-Worker",
    "act-as": "X-Trains-Act-As",
    "client": "X-Trains-Agent",
}
METADATA_EXTENSION = ".json"

DEFAULT_VENV_UPDATE_URL = (
    "https://raw.githubusercontent.com/Yelp/venv-update/v3.2.2/venv_update.py"
)
WORKING_REPOSITORY_DIR = "task_repository"
DEFAULT_VCS_CACHE = normalize_path(CONFIG_DIR, "vcs-cache")
PIP_EXTRA_INDICES = [
]
DEFAULT_PIP_DOWNLOAD_CACHE = normalize_path(CONFIG_DIR, "pip-download-cache")


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
