from __future__ import unicode_literals

import abc
from collections import OrderedDict
from contextlib import contextmanager
from hashlib import md5
from typing import Text, Iterable, Union, Optional, Dict, List

import six
from pathlib2 import Path

from clearml_agent.definitions import ENV_VENV_CACHE_PATH
from clearml_agent.helper.base import mkstemp, safe_remove_file, join_lines, select_for_platform
from clearml_agent.helper.console import ensure_binary
from clearml_agent.helper.os.folder_cache import FolderCache
from clearml_agent.helper.process import Executable, Argv, PathLike


@six.add_metaclass(abc.ABCMeta)
class PackageManager(object):
    """
    ABC for classes providing python package management interface
    """

    _selected_manager = None
    _cwd = None
    _pip_version = None
    _config_cache_folder = 'agent.venvs_cache.path'
    _config_cache_max_entries = 'agent.venvs_cache.max_entries'
    _config_cache_free_space_threshold = 'agent.venvs_cache.free_space_threshold_gb'
    _config_cache_lock_timeout = 'agent.venvs_cache.lock_timeout'

    def __init__(self):
        self._cache_manager = None

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
    def freeze(self, freeze_full_environment=False):
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
        result = self._install(
            *select_for_platform(
                windows=self.get_pip_versions(),
                linux=self.get_pip_versions()
            ),
            "--upgrade"
        )
        packages = self.run_with_env(('list',), output=True).splitlines()
        # p.split is ('pip', 'x.y.z')
        pip = [p.split() for p in packages if len(p.split()) == 2 and p.split()[0] == 'pip']
        if pip:
            # noinspection PyBroadException
            try:
                from .requirements import MarkerRequirement
                pip = pip[0][1].split('.')
                MarkerRequirement.pip_new_version = bool(int(pip[0]) >= 20)
            except Exception:
                pass
        return result

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
            # noinspection PyBroadException
            try:
                result = PackageManager._selected_manager.install_packages(package_name, *args)
                if result not in (0, None, True):
                    return False
            except Exception:
                return False
        return True

    @classmethod
    def out_of_scope_freeze(cls, freeze_full_environment=False):
        if PackageManager._selected_manager is not None:
            # noinspection PyBroadException
            try:
                return PackageManager._selected_manager.freeze(freeze_full_environment)
            except Exception:
                pass
        return []

    @classmethod
    def set_pip_version(cls, version):
        if not version:
            return

        if isinstance(version, (list, tuple)):
            versions = version
        else:
            versions = [version]

        cls._pip_version = []
        for version in versions:
            version = version.strip()
            if ('=' in version) or ('~' in version) or ('<' in version) or ('>' in version):
                cls._pip_version.append(version)
            else:
                cls._pip_version.append("==" + version)

    @classmethod
    def get_pip_versions(cls, pip="pip", wrap=''):
        return [
            (wrap + pip + version + wrap)
            for version in cls._pip_version or [""]
        ]

    def get_cached_venv(self, requirements, docker_cmd, python_version, cuda_version, destination_folder):
        # type: (Dict, Optional[Union[dict, str]], Optional[str], Optional[str], Path) -> Optional[Path]
        """
        Copy a cached copy of the venv (based on the requirements) into destination_folder.
        Return None if failed or cached entry does not exist
        """
        if not self._get_cache_manager():
            return None

        try:
            keys = self._generate_reqs_hash_keys(requirements, docker_cmd, python_version, cuda_version)
            return self._get_cache_manager().copy_cached_entry(keys, destination_folder)
        except Exception as ex:
            print("WARNING: Failed accessing venvs cache at {}: {}".format(destination_folder, ex))
            print("WARNING: Skipping venv cache - folder not accessible!")
            return None

    def add_cached_venv(
            self,
            requirements,  # type: Union[Dict, List[Dict]]
            docker_cmd,   # type: Optional[Union[dict, str]]
            python_version,  # type: Optional[str]
            cuda_version,  # type: Optional[str]
            source_folder,  # type:  Path
            exclude_sub_folders=None  # type: Optional[List[str]]
    ):
        # type: (...) -> ()
        """
        Copy the local venv folder into the venv cache (keys are based on the requirements+python+docker).
        """
        if not self._get_cache_manager():
            return

        print('Adding venv into cache: {}'.format(source_folder))

        try:
            keys = self._generate_reqs_hash_keys(requirements, docker_cmd, python_version, cuda_version)
            return self._get_cache_manager().add_entry(
                keys=keys, source_folder=source_folder, exclude_sub_folders=exclude_sub_folders)
        except Exception as ex:
            print("WARNING: Failed accessing venvs cache at {}: {}".format(source_folder, ex))
            print("WARNING: Skipping venv cache - folder not accessible!")
            return None

    def get_cache_folder(self):
        # type: () -> Optional[Path]
        if not self._get_cache_manager():
            return
        return self._get_cache_manager().get_cache_folder()

    def get_last_used_entry_cache(self):
        # type: () -> Optional[Path]
        """
        :return: the last used cached folder entry
        """
        if not self._get_cache_manager():
            return
        return self._get_cache_manager().get_last_copied_entry()

    def is_cached_enabled(self):
        if not self._cache_manager:
            cache_folder = ENV_VENV_CACHE_PATH.get() or self.session.config.get(self._config_cache_folder, None)
            if not cache_folder:
                return False
        return True

    @classmethod
    def _generate_reqs_hash_keys(cls, requirements_list, docker_cmd, python_version, cuda_version):
        # type: (Union[Dict, List[Dict]], Optional[Union[dict, str]], Optional[str], Optional[str]) -> List[str]
        requirements_list = requirements_list or dict()
        if not isinstance(requirements_list, (list, tuple)):
            requirements_list = [requirements_list]
        docker_cmd = dict(docker_cmd=docker_cmd) if isinstance(docker_cmd, str) else docker_cmd or dict()
        docker_cmd = OrderedDict(sorted(docker_cmd.items(), key=lambda t: t[0]))
        if 'docker_cmd' in docker_cmd:
            # we only take the first part of the docker_cmd which is the docker image name
            docker_cmd['docker_cmd'] = docker_cmd['docker_cmd'].strip('\r\n\t ').split(' ')[0]

        keys = []
        strip_chars = '\n\r\t '
        for requirements in requirements_list:
            pip, conda = ('pip', 'conda')
            pip_reqs = requirements.get(pip, '')
            conda_reqs = requirements.get(conda, '')
            if isinstance(pip_reqs, str):
                pip_reqs = pip_reqs.split('\n')
            if isinstance(conda_reqs, str):
                conda_reqs = conda_reqs.split('\n')
            pip_reqs = sorted([p.strip(strip_chars) for p in pip_reqs
                               if p.strip(strip_chars) and not p.strip(strip_chars).startswith('#')])
            conda_reqs = sorted([p.strip(strip_chars) for p in conda_reqs
                                 if p.strip(strip_chars) and not p.strip(strip_chars).startswith('#')])
            if not pip_reqs and not conda_reqs:
                continue
            # do not process "-r" or "--requirement" because we cannot know what we have in the git repo.
            if any(r.strip().startswith('-r ') or r.strip().startswith('--requirement ') for r in pip_reqs):
                continue
            hash_text = '{class_type}\n{docker_cmd}\n{cuda_ver}\n{python_version}\n{pip_reqs}\n{conda_reqs}'.format(
                class_type=str(cls),
                docker_cmd=str(docker_cmd or ''),
                cuda_ver=str(cuda_version or ''),
                python_version=str(python_version or ''),
                pip_reqs=str(pip_reqs or ''),
                conda_reqs=str(conda_reqs or ''),
            )
            keys.append(md5(ensure_binary(hash_text)).hexdigest())
        return sorted(list(set(keys)))

    def _get_cache_manager(self):
        if not self._cache_manager:
            cache_folder = None
            try:
                cache_folder = ENV_VENV_CACHE_PATH.get() or self.session.config.get(self._config_cache_folder, None)
                if not cache_folder:
                    return None

                max_entries = int(self.session.config.get(self._config_cache_max_entries, 10))
                free_space_threshold = float(self.session.config.get(self._config_cache_free_space_threshold, 0))
                self._cache_manager = FolderCache(
                    cache_folder, max_cache_entries=max_entries,
                    min_free_space_gb=free_space_threshold,
                    lock_timeout_seconds=self.session.config.get(self._config_cache_lock_timeout, None))
            except Exception as ex:
                print("WARNING: Failed accessing venvs cache at {}: {}".format(cache_folder, ex))
                print("WARNING: Skipping venv cache - folder not accessible!")
                return None

        return self._cache_manager
