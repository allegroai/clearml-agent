from copy import deepcopy
from functools import wraps

import attr
import sys
import os
from pathlib2 import Path
from trains_agent.helper.process import Argv, DEVNULL, check_if_command_exists
from trains_agent.session import Session, POETRY


def prop_guard(prop, log_prop=None):
    """
    Decorator to set a property.

    Args:
        prop: (dict): write your description
        log_prop: (dict): write your description
    """
    assert isinstance(prop, property)
    assert not log_prop or isinstance(log_prop, property)

    def decorator(func):
        """
        Decorator to mark a function as a warning.

        Args:
            func: (todo): write your description
        """
        message = "%s:%s calling {}, {} = %s".format(
            func.__name__, prop.fget.__name__
        )

        @wraps(func)
        def new_func(self, *args, **kwargs):
            """
            Creates a new function instance

            Args:
                self: (todo): write your description
            """
            prop_value = prop.fget(self)
            if log_prop:
                log_prop.fget(self).debug(
                    message,
                    type(self).__name__,
                    "" if prop_value else " not",
                    prop_value,
                )
            if prop_value:
                return func(self, *args, **kwargs)

        return new_func

    return decorator


class PoetryConfig:

    def __init__(self, session, interpreter=None):
        """
        Initialize the session.

        Args:
            self: (todo): write your description
            session: (todo): write your description
            interpreter: (int): write your description
        """
        # type: (Session, str) -> ()
        self.session = session
        self._log = session.get_logger(__name__)
        self._python = interpreter or sys.executable
        self._initialized = False

    @property
    def log(self):
        """
        Returns the current log.

        Args:
            self: (todo): write your description
        """
        return self._log

    @property
    def enabled(self):
        """
        Return true if the session is enabled.

        Args:
            self: (todo): write your description
        """
        return self.session.config["agent.package_manager.type"] == POETRY

    _guard_enabled = prop_guard(enabled, log)

    def run(self, *args, **kwargs):
        """
        Run the command.

        Args:
            self: (todo): write your description
        """
        func = kwargs.pop("func", Argv.get_output)
        kwargs.setdefault("stdin", DEVNULL)
        kwargs['env'] = deepcopy(os.environ)
        if 'VIRTUAL_ENV' in kwargs['env'] or 'CONDA_PREFIX' in kwargs['env']:
            kwargs['env'].pop('VIRTUAL_ENV', None)
            kwargs['env'].pop('CONDA_PREFIX', None)
            kwargs['env'].pop('PYTHONPATH', None)
            if hasattr(sys, "real_prefix") and hasattr(sys, "base_prefix"):
                path = ':'+kwargs['env']['PATH']
                path = path.replace(':'+sys.base_prefix, ':'+sys.real_prefix, 1)
                kwargs['env']['PATH'] = path

        if check_if_command_exists("poetry"):
            argv = Argv("poetry", *args)
        else:
            argv = Argv(self._python, "-m", "poetry", *args)
        self.log.debug("running: %s", argv)
        return func(argv, **kwargs)

    def _config(self, *args, **kwargs):
        """
        Wrapper.

        Args:
            self: (todo): write your description
        """
        return self.run("config", *args, **kwargs)

    @_guard_enabled
    def initialize(self, cwd=None):
        """
        Initialize the configuration.

        Args:
            self: (todo): write your description
            cwd: (array): write your description
        """
        if not self._initialized:
            self._initialized = True
            try:
                self._config("--local", "virtualenvs.in-project",  "true", cwd=cwd)
                # self._config("repositories.{}".format(self.REPO_NAME), PYTHON_INDEX)
                # self._config("http-basic.{}".format(self.REPO_NAME), *PYTHON_INDEX_CREDENTIALS)
            except Exception as ex:
                print("Exception: {}\nError: Failed configuring Poetry virtualenvs.in-project".format(ex))
                raise

    def get_api(self, path):
        """
        Return the api call.

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        # type: (Path) -> PoetryAPI
        return PoetryAPI(self, path)


@attr.s
class PoetryAPI(object):
    config = attr.ib(type=PoetryConfig)
    path = attr.ib(type=Path, converter=Path)

    INDICATOR_FILES = "pyproject.toml", "poetry.lock"

    def install(self):
        """
        Installs the install.

        Args:
            self: (todo): write your description
        """
        # type: () -> bool
        if self.enabled:
            self.config.run("install", "-n", cwd=str(self.path), func=Argv.check_call)
            return True
        return False

    @property
    def enabled(self):
        """
        Return true if the config is enabled.

        Args:
            self: (todo): write your description
        """
        return self.config.enabled and (
            any((self.path / indicator).exists() for indicator in self.INDICATOR_FILES)
        )

    def freeze(self):
        """
        Freeze the config file.

        Args:
            self: (todo): write your description
        """
        lines = self.config.run("show", cwd=str(self.path)).splitlines()
        lines = [[p for p in line.split(' ') if p] for line in lines]
        return {"pip": [parts[0]+'=='+parts[1]+' # '+' '.join(parts[2:]) for parts in lines]}

    def get_python_command(self, extra):
        """
        Returns the command to run.

        Args:
            self: (todo): write your description
            extra: (str): write your description
        """
        if check_if_command_exists("poetry"):
            return Argv("poetry", "run", "python", *extra)
        else:
            return Argv(self.config._python, "-m", "poetry", "run", "python", *extra)

    def upgrade_pip(self, *args, **kwargs):
        """
        Upgrade pip.

        Args:
            self: (todo): write your description
        """
        pass

    def set_selected_package_manager(self, *args, **kwargs):
        """
        Sets the package manager.

        Args:
            self: (todo): write your description
        """
        pass

    def out_of_scope_install_package(self, *args, **kwargs):
        """
        Installs packages_of_package_install_install

        Args:
            self: (todo): write your description
        """
        pass

    def install_from_file(self, *args, **kwargs):
        """
        Installs from a file.

        Args:
            self: (todo): write your description
        """
        pass
