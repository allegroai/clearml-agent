from functools import wraps

import attr
from pathlib2 import Path
from trains_agent.helper.process import Argv, DEVNULL
from trains_agent.session import Session, POETRY


def prop_guard(prop, log_prop=None):
    assert isinstance(prop, property)
    assert not log_prop or isinstance(log_prop, property)

    def decorator(func):
        message = "%s:%s calling {}, {} = %s".format(
            func.__name__, prop.fget.__name__
        )

        @wraps(func)
        def new_func(self, *args, **kwargs):
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

    def __init__(self, session):
        # type: (Session) -> ()
        self.session = session
        self._log = session.get_logger(__name__)

    @property
    def log(self):
        return self._log

    @property
    def enabled(self):
        return self.session.config["agent.package_manager.type"] == POETRY

    _guard_enabled = prop_guard(enabled, log)

    def run(self, *args, **kwargs):
        func = kwargs.pop("func", Argv.get_output)
        kwargs.setdefault("stdin", DEVNULL)
        argv = Argv("poetry", "-n", *args)
        self.log.debug("running: %s", argv)
        return func(argv, **kwargs)

    def _config(self, *args, **kwargs):
        return self.run("config", *args, **kwargs)

    @_guard_enabled
    def initialize(self):
        self._config("settings.virtualenvs.in-project",  "true")
        # self._config("repositories.{}".format(self.REPO_NAME), PYTHON_INDEX)
        # self._config("http-basic.{}".format(self.REPO_NAME), *PYTHON_INDEX_CREDENTIALS)

    def get_api(self, path):
        # type: (Path) -> PoetryAPI
        return PoetryAPI(self, path)


@attr.s
class PoetryAPI(object):
    config = attr.ib(type=PoetryConfig)
    path = attr.ib(type=Path, converter=Path)

    INDICATOR_FILES = "pyproject.toml", "poetry.lock"

    def install(self):
        # type: () -> bool
        if self.enabled:
            self.config.run("install", cwd=str(self.path), func=Argv.check_call)
            return True
        return False

    @property
    def enabled(self):
        return self.config.enabled and (
            any((self.path / indicator).exists() for indicator in self.INDICATOR_FILES)
        )

    def freeze(self):
        return {"poetry": self.config.run("show", cwd=str(self.path)).splitlines()}

    def get_python_command(self, extra):
        return Argv("poetry", "run", "python", *extra)

    def upgrade_pip(self, *args, **kwargs):
        pass

    def set_selected_package_manager(self, *args, **kwargs):
        pass

    def out_of_scope_install_package(self, *args, **kwargs):
        pass

    def install_from_file(self, *args, **kwargs):
        pass
