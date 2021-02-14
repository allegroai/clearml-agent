from __future__ import print_function, unicode_literals

import json
import logging
import os
import platform
import sys
from copy import deepcopy
from typing import Any, Callable

import attr
from pathlib2 import Path
from pyhocon import ConfigFactory, HOCONConverter, ConfigTree

from clearml_agent.backend_api.session import Session as _Session, Request
from clearml_agent.backend_api.session.client import APIClient
from clearml_agent.backend_config.defs import LOCAL_CONFIG_FILE_OVERRIDE_VAR, LOCAL_CONFIG_FILES
from clearml_agent.definitions import ENVIRONMENT_CONFIG, ENV_TASK_EXECUTE_AS_USER, ENVIRONMENT_BACKWARD_COMPATIBLE
from clearml_agent.errors import APIError
from clearml_agent.helper.base import HOCONEncoder
from clearml_agent.helper.process import Argv
from .version import __version__

POETRY = "poetry"


@attr.s
class ConfigValue(object):

    """
    Manages a single config key
    """

    config = attr.ib(type=ConfigTree)
    key = attr.ib(type=str)

    def get(self, default=None):
        """
        Get value of key with default
        """
        return self.config.get(self.key, default=default)

    def set(self, value):
        """
        Change the value of key
        """
        self.config.put(self.key, value)

    def modify(self, fn):
        # type: (Callable[[Any], Any]) -> ()
        """
        Change the value of a key using a function
        """
        self.set(fn(self.get()))


def tree(*args):
    """
    Helper function for creating config trees
    """
    return ConfigTree(args)


class Session(_Session):
    version = __version__
    force_debug = False

    def __init__(self, *args, **kwargs):
        # make sure we set the environment variable so the parent session opens the correct file
        if kwargs.get('config_file'):
            config_file = Path(os.path.expandvars(kwargs.get('config_file'))).expanduser().absolute().as_posix()
            kwargs['config_file'] = config_file
            LOCAL_CONFIG_FILE_OVERRIDE_VAR.set(config_file)
            if not Path(config_file).is_file():
                raise ValueError("Could not open configuration file: {}".format(config_file))

        cpu_only = kwargs.get('cpu_only')
        if cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['NVIDIA_VISIBLE_DEVICES'] = 'none'

        if kwargs.get('gpus') and not os.environ.get('KUBERNETES_SERVICE_HOST') \
                and not os.environ.get('KUBERNETES_PORT'):
            # CUDA_VISIBLE_DEVICES does not support 'all'
            if kwargs.get('gpus') == 'all':
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                os.environ['NVIDIA_VISIBLE_DEVICES'] = kwargs.get('gpus')
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['NVIDIA_VISIBLE_DEVICES'] = kwargs.get('gpus')

        if kwargs.get('only_load_config'):
            from clearml_agent.backend_api.config import load
            self.config = load()
        else:
            super(Session, self).__init__(*args, **kwargs)

        # set force debug mode, if it's on:
        if Session.force_debug:
            self.config["agent"]["debug"] = True

        self.log = self.get_logger(__name__)
        self.trace = kwargs.get('trace', False)
        self._config_file = kwargs.get('config_file') or LOCAL_CONFIG_FILE_OVERRIDE_VAR.get()
        if not self._config_file:
            for f in reversed(LOCAL_CONFIG_FILES):
                if os.path.exists(os.path.expanduser(os.path.expandvars(f))):
                    self._config_file = f
                    break
        self.api_client = APIClient(session=self, api_version="2.5")
        # HACK make sure we have python version to execute,
        # if nothing was specific, use the one that runs us
        def_python = ConfigValue(self.config, "agent.default_python")
        if not def_python.get():
            def_python.set("{version.major}.{version.minor}".format(version=sys.version_info))

        # HACK: backwards compatibility
        if ENVIRONMENT_BACKWARD_COMPATIBLE.get():
            os.environ['ALG_CONFIG_FILE'] = self._config_file
            os.environ['SM_CONFIG_FILE'] = self._config_file

        if not self.config.get('api.host', None) and self.config.get('api.api_server', None):
            self.config['api']['host'] = self.config.get('api.api_server')

        # initialize nvidia visibility variables
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        if os.environ.get('NVIDIA_VISIBLE_DEVICES') and not os.environ.get('CUDA_VISIBLE_DEVICES'):
            # do not create CUDA_VISIBLE_DEVICES if it doesn't exist, it breaks TF/PyTotch CUDA detection
            # os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('NVIDIA_VISIBLE_DEVICES')
            pass
        elif os.environ.get('CUDA_VISIBLE_DEVICES') and not os.environ.get('NVIDIA_VISIBLE_DEVICES'):
            os.environ['NVIDIA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES')

        # override with environment variables
        # cuda_version & cudnn_version are overridden with os.environ here, and normalized in the next section
        for config_key, env_config in ENVIRONMENT_CONFIG.items():
            # check if the propery is of a list:
            if config_key.endswith('.0'):
                if all(not i.get() for i in env_config.values()):
                    continue
                parent = config_key.partition('.0')[0]
                if not self.config[parent]:
                    self.config.put(parent, [])

                self.config.put(parent, self.config[parent] + [ConfigTree((k, v.get()) for k, v in env_config.items())])
                continue

            value = env_config.get()
            if not value:
                continue
            env_key = ConfigValue(self.config, config_key)
            env_key.set(value)

        # initialize cuda versions
        try:
            from clearml_agent.helper.package.requirements import RequirementsManager
            agent = self.config['agent']
            agent['cuda_version'], agent['cudnn_version'] = \
                RequirementsManager.get_cuda_version(self.config) if not cpu_only else ('0', '0')
        except Exception:
            pass

        # initialize worker name
        worker_name = ConfigValue(self.config, "agent.worker_name")
        if not worker_name.get():
            worker_name.set(platform.node())

        if not kwargs.get('only_load_config'):
            self.create_cache_folders()

    @staticmethod
    def get_logger(name):
        logger = logging.getLogger(name)
        logger.propagate = True
        return TrainsAgentLogger(logger)

    @staticmethod
    def set_debug_mode(enable):
        if enable:
            import logging
            logging.basicConfig(level=logging.DEBUG)
        Session.force_debug = enable

    @property
    def debug_mode(self):
        return Session.force_debug or self.config.get("agent.debug", False)

    @property
    def config_file(self):
        return self._config_file

    def create_cache_folders(self, slot_index=0):
        """
        create and update the cache folders
        notice we support multiple instances sharing the same cache on some folders
        and on some we use "instance slot" numbers in order to differentiate between the different instances running
        notice slot_index=0 is the default, meaning no suffix is added to the singleton_folders

        Note: do not call this function twice with non zero slot_index
            it will add a suffix to the folders on each call

        :param slot_index: integer
        """

        # create target folders:
        folder_keys = ('agent.venvs_dir', 'agent.vcs_cache.path',
                       'agent.pip_download_cache.path',
                       'agent.docker_pip_cache', 'agent.docker_apt_cache')
        singleton_folders = ('agent.venvs_dir', 'agent.docker_apt_cache')

        if ENV_TASK_EXECUTE_AS_USER.get():
            folder_keys = tuple(list(folder_keys) + ['sdk.storage.cache.default_base_dir'])
            singleton_folders = tuple(list(singleton_folders) + ['sdk.storage.cache.default_base_dir'])

        for key in folder_keys:
            folder_key = ConfigValue(self.config, key)
            if not folder_key.get():
                continue

            if slot_index and key in singleton_folders:
                f = folder_key.get()
                if f.endswith(os.path.sep):
                    f = f[:-1]
                folder_key.set(f + '.{}'.format(slot_index))

            # update the configuration for full path
            folder = Path(os.path.expandvars(folder_key.get())).expanduser().absolute()
            folder_key.set(folder.as_posix())
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except:
                pass

    def print_configuration(self, remove_secret_keys=("secret", "pass", "token", "account_key")):
        # remove all the secrets from the print
        def recursive_remove_secrets(dictionary, secret_keys=()):
            for k in list(dictionary):
                for s in secret_keys:
                    if s in k:
                        dictionary.pop(k)
                        break
                if isinstance(dictionary.get(k, None), dict):
                    recursive_remove_secrets(dictionary[k], secret_keys=secret_keys)
                elif isinstance(dictionary.get(k, None), (list, tuple)):
                    for item in dictionary[k]:
                        if isinstance(item, dict):
                            recursive_remove_secrets(item, secret_keys=secret_keys)

        config = deepcopy(self.config.to_dict())
        # remove the env variable, it's not important
        config.pop('env', None)
        if remove_secret_keys:
            recursive_remove_secrets(config, secret_keys=remove_secret_keys)
        # remove logging.loggers.urllib3.level from the print
        try:
            config['logging']['loggers']['urllib3'].pop('level', None)
        except (KeyError, TypeError, AttributeError):
            pass
        try:
            config['logging'].pop('version', None)
        except (KeyError, TypeError, AttributeError):
            pass
        config = ConfigFactory.from_dict(config)
        self.log.debug("Run by interpreter: %s", sys.executable)
        print(
            "Current configuration (clearml_agent v{}, location: {}):\n"
            "----------------------\n{}\n".format(
                self.version, self._config_file, HOCONConverter.convert(config, "properties")
            )
        )

    def send_api(self, request):
        # type: (Request) -> Any
        result = self.send(request)
        if not result.ok():
            raise APIError(result)
        if not result.response:
            raise APIError(result, extra_info="Invalid response")
        return result.response

    def get(self, service, action, version=None, headers=None,
            data=None, json=None, async_enable=False, **kwargs):
        return self._manual_request(service=service, action=action,
                                    version=version, method="get", headers=headers,
                                    data=data, async_enable=async_enable,
                                    json=json or kwargs)

    def post(self, service, action, version=None, headers=None,
             data=None, json=None, async_enable=False, **kwargs):
        return self._manual_request(service=service, action=action,
                                    version=version, method="post", headers=headers,
                                    data=data, async_enable=async_enable,
                                    json=json or kwargs)

    def _manual_request(self, service, action, version=None, method="get", headers=None,
            data=None, json=None, async_enable=False, **kwargs):

        res = self.send_request(service=service, action=action,
                                version=version, method=method, headers=headers,
                                data=data, async_enable=async_enable,
                                json=json or kwargs)

        try:
            res_json = res.json()
            return_code = res_json["meta"]["result_code"]
        except (ValueError, KeyError, TypeError):
            raise APIError(res)

        # check return code
        if return_code != 200:
            raise APIError(res)

        return res_json["data"]

    def to_json(self):
        return json.dumps(
            self.config.as_plain_ordered_dict(), cls=HOCONEncoder, indent=4
        )

    def command(self, *args):
        return Argv(*args, log=self.get_logger(Argv.__module__))


@attr.s
class TrainsAgentLogger(object):
    """
    Proxy around logging.Logger because inheriting from it is difficult.
    """

    logger = attr.ib(type=logging.Logger)

    def _log_with_error(self, level, *args, **kwargs):
        """
        Include error information when in debug mode
        """
        kwargs.setdefault("exc_info", self.logger.isEnabledFor(logging.DEBUG))
        return self.logger.log(level, *args, **kwargs)

    def warning(self, *args, **kwargs):
        return self._log_with_error(logging.WARNING, *args, **kwargs)

    def error(self, *args, **kwargs):
        return self._log_with_error(logging.ERROR, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.logger, item)

    def __call__(self, *args, **kwargs):
        """
        Compatibility with old ``Command.log()`` method
        """
        return self.logger.info(*args, **kwargs)


def normalize_cuda_version(value):
    # type: (Any) -> str
    """
    Take variably formatted cuda version string/number and return it in the same format:
    string decimal representation of 10 * major + minor.

    >>> normalize_cuda_version(100)
    '100'
    >>> normalize_cuda_version("100")
    '100'
    >>> normalize_cuda_version(10)
    '10'
    >>> normalize_cuda_version(10.0)
    '100'
    >>> normalize_cuda_version("10.0")
    '100'
    >>> normalize_cuda_version("10.0.130")
    '100'
    """
    value = str(value)
    if "." in value:
        try:
            value = str(int(float(".".join(value.split(".")[:2])) * 10))
        except (ValueError, TypeError):
            pass
    return value
