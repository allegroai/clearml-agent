import sys
from itertools import chain
from typing import Text, Optional

from trains_agent.definitions import PIP_EXTRA_INDICES, PROGRAM_NAME
from trains_agent.helper.package.base import PackageManager
from trains_agent.helper.process import Argv, DEVNULL
from trains_agent.session import Session


class SystemPip(PackageManager):

    indices_args = None

    def __init__(self, interpreter=None, session=None):
        # type: (Optional[Text], Optional[Session]) -> ()
        """
        Program interface to the system pip.
        """
        self._bin = interpreter or sys.executable
        self.session = session

    @property
    def bin(self):
        """
        The bin. numpy.

        Args:
            self: (todo): write your description
        """
        return self._bin

    def create(self):
        """
        Create a new : attr

        Args:
            self: (int): write your description
        """
        pass

    def remove(self):
        """
        Remove the filter.

        Args:
            self: (todo): write your description
        """
        pass

    def install_from_file(self, path):
        """
        Install the file.

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        self.run_with_env(('install', '-r', path) + self.install_flags(), cwd=self.cwd)

    def install_packages(self, *packages):
        """
        Install packages.

        Args:
            self: (todo): write your description
            packages: (str): write your description
        """
        self._install(*(packages + self.install_flags()))

    def _install(self, *args):
        """
        Install the environment.

        Args:
            self: (todo): write your description
        """
        self.run_with_env(('install',) + args, cwd=self.cwd)

    def uninstall_packages(self, *packages):
        """
        Uninstall packages.

        Args:
            self: (todo): write your description
            packages: (str): write your description
        """
        self.run_with_env(('uninstall', '-y') + packages)

    def download_package(self, package, cache_dir):
        """
        Downloads a package.

        Args:
            self: (todo): write your description
            package: (str): write your description
            cache_dir: (str): write your description
        """
        self.run_with_env(
            (
                'download',
                package,
                '--dest', cache_dir,
                '--no-deps',
            ) + self.install_flags()
        )

    def load_requirements(self, requirements):
        """
        Load the requirements from requirements file.

        Args:
            self: (todo): write your description
            requirements: (dict): write your description
        """
        requirements = requirements.get('pip') if isinstance(requirements, dict) else requirements
        if not requirements:
            return
        with self.temp_file('cached-reqs', requirements) as path:
            self.install_from_file(path)

    def uninstall(self, package):
        """
        Uninstall a package.

        Args:
            self: (todo): write your description
            package: (str): write your description
        """
        self.run_with_env(('uninstall', '-y', package))

    def freeze(self):
        """
        pip freeze to all install packages except the running program
        :return: Dict contains pip as key and pip's packages to install
        :rtype: Dict[str: List[str]]
        """
        packages = self.run_with_env(('freeze',), output=True).splitlines()
        packages_without_program = [package for package in packages if PROGRAM_NAME not in package]
        return {'pip': packages_without_program}

    def run_with_env(self, command, output=False, **kwargs):
        """
        Run a shell command using environment from a virtualenv script
        :param command: command to run
        :type command: Iterable[Text]
        :param output: return output
        :param kwargs: kwargs for get_output/check_output command
        """
        command = self._make_command(command)
        return (command.get_output if output else command.check_call)(stdin=DEVNULL, **kwargs)

    def _make_command(self, command):
        """
        Make a command.

        Args:
            self: (todo): write your description
            command: (str): write your description
        """
        return Argv(self.bin, '-m', 'pip', '--disable-pip-version-check', *command)

    def install_flags(self):
        """
        Install the arguments from the arguments.

        Args:
            self: (todo): write your description
        """
        if self.indices_args is None:
            self.indices_args = tuple(
                chain.from_iterable(('--extra-index-url', x) for x in PIP_EXTRA_INDICES)
            )
        return self.indices_args
