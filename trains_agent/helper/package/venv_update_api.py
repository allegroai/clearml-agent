from typing import Optional, Text

import requests
from pathlib2 import Path

import six
from trains_agent.definitions import CONFIG_DIR
from trains_agent.helper.process import Argv, DEVNULL
from .pip_api.venv import VirtualenvPip


class VenvUpdateAPI(VirtualenvPip):
    URL_FILE_PATH = Path(CONFIG_DIR, "venv-update-url.txt")
    SCRIPT_PATH = Path(CONFIG_DIR, "venv-update")

    def __init__(self, url, *args, **kwargs):
        """
        Initialize an environment.

        Args:
            self: (todo): write your description
            url: (str): write your description
        """
        super(VenvUpdateAPI, self).__init__(*args, **kwargs)
        self.url = url
        self._script_path = None
        self._first_install = True

    @property
    def downloaded_venv_url(self):
        """
        Return the url of the virtualenv.

        Args:
            self: (todo): write your description
        """
        # type: () -> Optional[Text]
        try:
            return self.URL_FILE_PATH.read_text()
        except OSError:
            return None

    @downloaded_venv_url.setter
    def downloaded_venv_url(self, value):
        """
        Updates the url.

        Args:
            self: (todo): write your description
            value: (str): write your description
        """
        self.URL_FILE_PATH.write_text(value)

    def _check_script_validity(self, path):
        """
        Make sure script in ``path`` is a valid python script
        :param path:
        :return:
        """
        result = Argv(self.bin, path, "--version").call(
            stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL
        )
        return result == 0

    @property
    def script_path(self):
        """
        Returns the path of the script.

        Args:
            self: (todo): write your description
        """
        # type: () -> Text
        if not self._script_path:
            self._script_path = self.SCRIPT_PATH
            if not (
                self._script_path.exists()
                and self.downloaded_venv_url
                and self.downloaded_venv_url == self.url
                and self._check_script_validity(self._script_path)
            ):
                with self._script_path.open("wb") as f:
                    for data in requests.get(self.url, stream=True):
                        f.write(data)
                self.downloaded_venv_url = self.url
        return self._script_path

    def install_from_file(self, path):
        """
        Install python script from a file.

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        first_install = (
            Argv(
                self.python,
                six.text_type(self.script_path),
                "venv=",
                "-p",
                self.python,
                self.path,
            )
            + self.create_flags()
            + ("install=", "-r", path)
            + self.install_flags()
        )
        later_install = first_install + (
            "pip-command=",
            "pip-faster",
            "install",
            "--upgrade",  # no --prune
        )
        self._choose_install(first_install, later_install)

    def install_packages(self, *packages):
        """
        Install packages.

        Args:
            self: (todo): write your description
            packages: (str): write your description
        """
        first_install = (
            Argv(
                self.python,
                six.text_type(self.script_path),
                "venv=",
                self.path,
                "install=",
            )
            + packages
        )
        later_install = first_install + (
            "pip-command=",
            "pip-faster",
            "install",
            "--upgrade",  # no --prune
        )
        self._choose_install(first_install, later_install)

    def _choose_install(self, first, rest):
        """
        Chooses install command.

        Args:
            self: (todo): write your description
            first: (todo): write your description
            rest: (todo): write your description
        """
        if self._first_install:
            command = first
            self._first_install = False
        else:
            command = rest
        command.check_call(stdin=DEVNULL)

    def upgrade_pip(self):
        """
        pip and venv-update versions are coupled, venv-update installs the latest compatible pip
        """
        pass
