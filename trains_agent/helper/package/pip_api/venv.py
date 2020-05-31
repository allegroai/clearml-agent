from pathlib2 import Path

from trains_agent.helper.base import select_for_platform, rm_tree
from trains_agent.helper.package.base import PackageManager
from trains_agent.helper.process import Argv, PathLike
from trains_agent.session import Session
from ..pip_api.system import SystemPip
from ..requirements import RequirementsManager


class VirtualenvPip(SystemPip, PackageManager):
    def __init__(self, session, python, requirements_manager, path, interpreter=None):
        # type: (Session, float, RequirementsManager, PathLike, PathLike) -> ()
        """
        Program interface to virtualenv pip.
        Must be given either path to virtualenv or source command.
        Either way, ``self.source`` is exposed.
        :param session: a Session object for communication
        :param python: interpreter path
        :param path: path of virtual environment to create/manipulate
        :param python: python version
        :param interpreter: path of python interpreter
        """
        super(VirtualenvPip, self).__init__(
            session=session,
            interpreter=interpreter or Path(
                path, select_for_platform(linux="bin/python", windows="scripts/python.exe"))
        )
        self.path = path
        self.requirements_manager = requirements_manager
        self.python = python

    def _make_command(self, command):
        return self.session.command(self.bin, "-m", "pip", "--disable-pip-version-check", *command)

    def load_requirements(self, requirements):
        if isinstance(requirements, dict) and requirements.get("pip"):
            requirements["pip"] = self.requirements_manager.replace(requirements["pip"])
        super(VirtualenvPip, self).load_requirements(requirements)
        self.requirements_manager.post_install()

    def create_flags(self):
        """
        Configurable environment creation arguments
        """
        return Argv.conditional_flag(
            self.session.config["agent.package_manager.system_site_packages"],
            "--system-site-packages",
        )

    def install_flags(self):
        """
        Configurable package installation creation arguments
        """
        return super(VirtualenvPip, self).install_flags() + Argv.conditional_flag(
            self.session.config["agent.package_manager.force_upgrade"], "--upgrade"
        )

    def create(self):
        """
        Create virtualenv.
        Only valid if instantiated with path.
        Use self.python as self.bin does not exist.
        """
        self.session.command(
            self.python, "-m", "virtualenv", self.path, *self.create_flags()
        ).check_call()
        return self

    def remove(self):
        """
        Delete virtualenv.
        Only valid if instantiated with path.
        """
        rm_tree(self.path)
