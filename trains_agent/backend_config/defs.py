from os.path import expanduser
from pathlib2 import Path

ENV_VAR = 'TRAINS_ENV'
""" Name of system environment variable that can be used to specify the config environment name """


DEFAULT_CONFIG_FOLDER = 'config'
""" Default config folder to search for when loading relative to a given path """


ENV_CONFIG_PATHS = [
]


""" Environment-related config paths """


LOCAL_CONFIG_PATHS = [
    # '/etc/opt/trains',               # used by servers for docker-generated configuration
    # expanduser('~/.trains/config'),
]
""" Local config paths, not related to environment """


LOCAL_CONFIG_FILES = [
    expanduser('~/trains.conf'),    # used for workstation configuration (end-users, workers)
]
""" Local config files (not paths) """


LOCAL_CONFIG_FILE_OVERRIDE_VAR = 'TRAINS_CONFIG_FILE'
""" Local config file override environment variable. If this is set, no other local config files will be used. """


ENV_CONFIG_PATH_OVERRIDE_VAR = 'TRAINS_CONFIG_PATH'
""" 
Environment-related config path override environment variable. If this is set, no other env config path will be used. 
"""


class Environment(object):
    """ Supported environment names """
    default = 'default'
    demo = 'demo'
    local = 'local'


class UptimeConf(object):
    min_api_version = "2.10"
    queue_tag_on = "force_workers:on"
    queue_tag_off = "force_workers:off"
    worker_key = "force"
    worker_value_off = ["off"]
    worker_value_on = ["on"]


CONFIG_FILE_EXTENSION = '.conf'


def is_config_file(path):
    """
    Return true if path is a config file.

    Args:
        path: (str): write your description
    """
    return Path(path).suffix == CONFIG_FILE_EXTENSION
