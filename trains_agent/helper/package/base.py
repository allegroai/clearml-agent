from __future__ import unicode_literals

import abc
from contextlib import contextmanager
from typing import Text, Iterable, Union

import six
from trains_agent.helper.base import mkstemp, safe_remove_file, join_lines
from trains_agent.helper.process import Executable, Argv, PathLike


@six.add_metaclass(abc.ABCMeta)
class PackageManager(object):
    """
    ABC for classes providing python package management interface
    """

    _selected_manager = None
    _cwd = None
    _pip_version = None

    @abc.abstractproperty
    def bin(self):
        # type: () -> PathLike
        pass

    @abc.abstractmethod
    def create(self):
        pass

    @abc.abstractmethod
    def remove(self):
        pass

    @abc.abstractmethod
    def install_from_file(self, path):
        pass

    @abc.abstractmethod
    def freeze(self):
        pass

    @abc.abstractmethod
    def load_requirements(self, requirements):
        pass

    @abc.abstractmethod
    def install_packages(self, *packages):
        # type: (Iterable[Text]) -> None
        """
        Install packages, upgrading depends on config
        """
        pass

    @abc.abstractmethod
    def _install(self, *packages):
        # type: (Iterable[Text]) -> None
        """
        Run install command
        """
        pass

    @abc.abstractmethod
    def uninstall_packages(self, *packages):
        # type: (Iterable[Text]) -> None
        pass

    def upgrade_pip(self):
        return self._install("pip"+self.get_pip_version(), "--upgrade")

    def get_python_command(self, extra=()):
        # type: (...) -> Executable
        return Argv(self.bin, *extra)

    @contextmanager
    def temp_file(self, prefix, contents, suffix=".txt"):
        # type: (Union[Text, Iterable[Text]], Iterable[Text], Text) -> Text
        """
        Write contents to a temporary file, yielding its path. Finally, delete it.
        :param prefix: file name prefix
        :param contents: text lines to write
        :param suffix: file name suffix
        """
        f, temp_path = mkstemp(suffix=suffix, prefix=prefix)
        with f:
            f.write(
                contents
                if isinstance(contents, six.text_type)
                else join_lines(contents)
            )
        try:
            yield temp_path
        finally:
            if not self.session.debug_mode:
                safe_remove_file(temp_path)

    def set_selected_package_manager(self):
        # set this instance as the selected package manager
        # this is helpful when we want out of context requirement installations
        PackageManager._selected_manager = self

    @property
    def cwd(self):
        return self._cwd

    @cwd.setter
    def cwd(self, value):
        self._cwd = value

    @classmethod
    def out_of_scope_install_package(cls, package_name, *args):
        if PackageManager._selected_manager is not None:
            try:
                return PackageManager._selected_manager._install(package_name, *args)
            except Exception:
                pass
        return

    @classmethod
    def out_of_scope_freeze(cls):
        if PackageManager._selected_manager is not None:
            try:
                return PackageManager._selected_manager.freeze()
            except Exception:
                pass
        return []

    @classmethod
    def set_pip_version(cls, version):
        if not version:
            return
        version = version.replace(' ', '')
        if ('=' in version) or ('~' in version) or ('<' in version) or ('>' in version):
            cls._pip_version = version
        else:
            cls._pip_version = "=="+version

    @classmethod
    def get_pip_version(cls):
        return cls._pip_version or ''
