from typing import Text

from furl import furl
from pathlib2 import Path

from trains_agent.config import Config
from .pip_api.system import SystemPip


class RequirementsTranslator(object):

    """
    Translate explicit URLs to local URLs after downloading them to cache
    """

    SUPPORTED_SCHEMES = ["http", "https", "ftp"]

    def __init__(self, session, interpreter=None, cache_dir=None):
        self._session = session
        config = session.config
        self.cache_dir = cache_dir or Path(config["agent.pip_download_cache.path"]).expanduser().as_posix()
        self.enabled = config["agent.pip_download_cache.enabled"]
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.config = Config()
        self.pip = SystemPip(interpreter=interpreter, session=self._session)

    def download(self, url):
        self.pip.download_package(url, cache_dir=self.cache_dir)

    @classmethod
    def is_supported_link(cls, line):
        # type: (Text) -> bool
        """
        Return whether requirement is a link that should be downloaded to cache
        """
        url = furl(line)
        return (
            url.scheme
            and url.scheme.lower() in cls.SUPPORTED_SCHEMES
            and line.lstrip().lower().startswith(url.scheme.lower())
        )

    def translate(self, line):
        """
        If requirement is supported, download it to cache and return the download path
        """
        if not (self.enabled and self.is_supported_link(line)):
            return line
        command = self.config.command
        command.log('Downloading "{}" to pip cache'.format(line))
        url = furl(line)
        try:
            wheel_name = url.path.segments[-1]
        except IndexError:
            command.error('Could not parse wheel name of "{}"'.format(line))
            return line
        try:
            self.download(line)
            downloaded = Path(self.cache_dir, wheel_name).expanduser().as_uri()
        except Exception:
            command.error('Could not download wheel name of "{}"'.format(line))
            return line
        return downloaded
