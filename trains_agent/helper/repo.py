import abc
import re
import shutil
import subprocess
from distutils.spawn import find_executable
from hashlib import md5
from os import environ, getenv
from typing import Text, Sequence, Mapping, Iterable, TypeVar, Callable, Tuple

import attr
from furl import furl
from pathlib2 import Path

import six

from trains_agent.definitions import ENV_AGENT_GIT_USER, ENV_AGENT_GIT_PASS
from trains_agent.helper.console import ensure_text, ensure_binary
from trains_agent.errors import CommandFailedError
from trains_agent.helper.base import (
    select_for_platform,
    rm_tree,
    ExecutionInfo,
    normalize_path,
    create_file_if_not_exists,
)
from trains_agent.helper.process import DEVNULL, Argv, PathLike, COMMAND_SUCCESS
from trains_agent.session import Session


class VcsFactory(object):
    """
    Creates VCS instances
    """

    GIT_SUFFIX = ".git"

    @classmethod
    def create(cls, session, execution_info, location):
        # type: (Session, ExecutionInfo, PathLike) -> VCS
        """
        Create a VCS instance for config and url
        :param session: program session
        :param execution_info: task ExecutionInfo
        :param location: (desired) clone location
        """
        url = execution_info.repository
        # We only support git, hg is deprecated
        is_git = True
        # is_git = url.endswith(cls.GIT_SUFFIX)
        vcs_cls = Git if is_git else Hg
        revision = (
            execution_info.version_num
            or execution_info.tag
            or vcs_cls.remote_branch_name(execution_info.branch or vcs_cls.main_branch)
        )
        return vcs_cls(session, url, location, revision)


# noinspection PyUnresolvedReferences
@attr.s
class RepoInfo(object):
    """
    Cloned repository information
    :param type: VCS type
    :param url: repository url
    :param branch: revision branch
    :param commit: revision number
    :param root: clone location path
    """

    type = attr.ib(type=str)
    url = attr.ib(type=str)
    branch = attr.ib(type=str)
    commit = attr.ib(type=str)
    root = attr.ib(type=str)


RType = TypeVar("RType")


@six.add_metaclass(abc.ABCMeta)
class VCS(object):

    """
    Provides overloaded utilities for handling repositories of different types
    """

    # additional environment variables for VCS commands
    COMMAND_ENV = {}

    PATCH_ADDED_FILE_RE = re.compile(r"^\+\+\+ b/(?P<path>.*)")

    def __init__(self, session, url, location, revision):
        # type: (Session, Text, PathLike, Text) -> ()
        """
        Create a VCS instance for config and url
        :param session: program session
        :param url: repository url
        :param location: (desired) clone location
        :param: desired clone revision
        """
        self.session = session
        self.log = self.session.get_logger(
            "{}.{}".format(__name__, type(self).__name__)
        )
        self.url = url
        self.location = Text(location)
        self.revision = revision
        self.log = self.session.get_logger(__name__)

    @property
    def url_with_auth(self):
        """
        Return URL with configured user/password
        """
        return self.add_auth(self.session.config, self.url)

    @abc.abstractproperty
    def executable_name(self):
        """
        Name of command executable
        """
        pass

    @abc.abstractproperty
    def main_branch(self):
        """
        Name of default/main branch
        """
        pass

    @abc.abstractproperty
    def checkout_flags(self):
        # type: () -> Sequence[Text]
        """
        Command-line flags for checkout command
        """
        pass

    @abc.abstractproperty
    def patch_base(self):
        # type: () -> Sequence[Text]
        """
        Command and flags for applying patches
        """
        pass

    def patch(self, location, patch_content):
        # type: (PathLike, Text) -> bool
        """
        Apply patch repository at `location`
        """
        self.log.info("applying diff to %s", location)

        for match in filter(
            None, map(self.PATCH_ADDED_FILE_RE.match, patch_content.splitlines())
        ):
            create_file_if_not_exists(normalize_path(location, match.group("path")))

        return_code, errors = self.call_with_stdin(
            patch_content, *self.patch_base, cwd=location
        )
        if return_code:
            self.log.error("Failed applying diff")
            lines = errors.splitlines()
            if any(l for l in lines if "no such file or directory" in l.lower()):
                self.log.warning(
                    "NOTE: files were not found when applying diff, perhaps you forgot to push your changes?"
                )
            return False
        else:
            self.log.info("successfully applied uncommitted changes")
        return True

    # Command-line flags for clone command
    clone_flags = ()

    @abc.abstractmethod
    def executable_not_found_error_help(self):
        # type: () -> Text
        """
        Instructions for when executable is not found
        """
        pass

    @staticmethod
    def remote_branch_name(branch):
        # type: (Text) -> Text
        """
        Creates name of remote branch from name of local/ambiguous branch.
        Returns same name by default.
        """
        return branch

    # parse scp-like git ssh URLs, e.g: git@host:user/project.git
    SSH_URL_GIT_SYNTAX = re.compile(
        r"""
        ^
        (?:(?P<user>{regular}*?)@)?
        (?P<host>{regular}*?)
        :
        (?P<path>{regular}.*)?
        $
        """.format(
            regular=r"[^/@:#]"
        ),
        re.VERBOSE,
    )

    @classmethod
    def resolve_ssh_url(cls, url):
        # type: (Text) -> Text
        """
        Replace SSH URL with HTTPS URL when applicable
        """

        def get_username(user_, password=None):
            """
            Remove special SSH users hg/git
            """
            return (
                None
                if user_ and user_.lower() in ["hg", "git"] and not password
                else user_
            )

        match = cls.SSH_URL_GIT_SYNTAX.match(url)
        if match:
            user, host, path = match.groups()
            return (
                furl()
                .set(scheme="https", username=get_username(user), host=host, path=path)
                .url
            )
        parsed_url = furl(url)
        if parsed_url.scheme == "ssh":
            return parsed_url.set(
                scheme="https",
                username=get_username(
                    parsed_url.username, password=parsed_url.password
                ),
            ).url
        return url

    def _set_ssh_url(self):
        """
        Replace instance URL with SSH substitution result and report to log.
        According to ``man ssh-add``, ``SSH_AUTH_SOCK`` must be set in order for ``ssh-add`` to work.
        """
        if not self.session.config.agent.translate_ssh:
            return

        ssh_agent_variable = "SSH_AUTH_SOCK"
        if not getenv(ssh_agent_variable) and (
                (ENV_AGENT_GIT_USER.get() or self.session.config.get('agent.git_user', None)) and
                (ENV_AGENT_GIT_PASS.get() or self.session.config.get('agent.git_pass', None))
        ):
            new_url = self.resolve_ssh_url(self.url)
            if new_url != self.url:
                print("Using user/pass credentials - replacing ssh url '{}' with https url '{}'".format(
                    self.url, new_url))
                self.url = new_url

    def clone(self, branch=None):
        # type: (Text) -> None
        """
        Clone repository to destination and checking out `branch`.
        If not in debug mode, filter VCS password from output.
        """
        self._set_ssh_url()
        clone_command = ("clone", self.url_with_auth, self.location) + self.clone_flags
        # clone all branches regardless of when we want to later checkout
        # if branch:
        #    clone_command += ("-b", branch)
        if self.session.debug_mode:
            self.call(*clone_command)
            return

        def normalize_output(result):
            """
            Returns result string without user's password.
            NOTE: ``self.get_stderr``'s result might or might not have the same type as ``e.output`` in case of error.
            """
            string_type = (
                ensure_text
                if isinstance(result, six.text_type)
                else ensure_binary
            )
            return result.replace(
                string_type(self.url),
                string_type(furl(self.url).remove(password=True).tostr()),
            )

        def print_output(output):
            print(ensure_text(output))

        try:
            print_output(normalize_output(self.get_stderr(*clone_command)))
        except subprocess.CalledProcessError as e:
            # In Python 3, subprocess.CalledProcessError has a `stderr` attribute,
            # but since stderr is redirect to `subprocess.PIPE` it will appear in the usual `output` attribute
            if e.output:
                e.output = normalize_output(e.output)
                print_output(e.output)
            raise

    def checkout(self):
        # type: () -> None
        """
        Checkout repository at specified revision
        """
        self.call("checkout", self.revision, *self.checkout_flags, cwd=self.location)

    @abc.abstractmethod
    def pull(self):
        # type: () -> None
        """
        Pull remote changes for revision
        """
        pass

    def call(self, *argv, **kwargs):
        """
        Execute argv without stdout/stdin.
        Remove stdin so git/hg can't ask for passwords.
        ``kwargs`` can override all arguments passed to subprocess.
        """
        return self._call_subprocess(subprocess.check_call, argv, **kwargs)

    def call_with_stdin(self, input_, *argv, **kwargs):
        # type: (...) -> Tuple[int, str]
        """
        Run command with `input_` as stdin
        """
        input_ = input_.encode("latin1")
        if not input_.endswith(b"\n"):
            input_ += b"\n"
        process = self._call_subprocess(
            subprocess.Popen,
            argv,
            **dict(
                kwargs,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
        _, stderr = process.communicate(input_)
        if stderr:
            self.log.warning("%s: %s", self._get_vcs_command(argv), stderr)
        return process.returncode, Text(stderr)

    def get_stderr(self, *argv, **kwargs):
        """
        Execute argv without stdout/stdin in <cwd> and get stderr output.
        Remove stdin so git/hg can't ask for passwords.
        ``kwargs`` can override all arguments passed to subprocess.
        """
        process = self._call_subprocess(
            subprocess.Popen, argv, **dict(kwargs, stderr=subprocess.PIPE, stdout=None)
        )
        _, stderr = process.communicate()
        code = process.poll()
        if code == COMMAND_SUCCESS:
            return stderr
        with Argv.normalize_exception(censor_password=True):
            raise subprocess.CalledProcessError(
                returncode=code, cmd=argv, output=stderr
            )

    def _call_subprocess(self, func, argv, **kwargs):
        # type: (Callable[..., RType], Iterable[Text], dict) ->  RType
        cwd = kwargs.pop("cwd", None)
        cwd = cwd and str(cwd)
        kwargs = dict(
            dict(
                censor_password=True,
                cwd=cwd,
                stdin=DEVNULL,
                stdout=DEVNULL,
                env=dict(self.COMMAND_ENV, **environ),
            ),
            **kwargs
        )
        command = self._get_vcs_command(argv)
        self.log.debug("Running: %s", list(command))
        return command.call_subprocess(func, **kwargs)

    def _get_vcs_command(self, argv):
        # type: (Iterable[PathLike]) -> Argv
        return Argv(self.executable_name, *argv)

    @classmethod
    def add_auth(cls, config, url):
        """
        Add username and password to URL if missing from URL and present in config.
        Does not modify ssh URLs.
        """
        parsed_url = furl(url)
        if parsed_url.scheme in ["", "ssh"] or parsed_url.scheme.startswith("git"):
            return parsed_url.url
        config_user = ENV_AGENT_GIT_USER.get() or config.get("agent.{}_user".format(cls.executable_name), None)
        config_pass = ENV_AGENT_GIT_PASS.get() or config.get("agent.{}_pass".format(cls.executable_name), None)
        if (
            (not (parsed_url.username and parsed_url.password))
            and config_user
            and config_pass
        ):
            parsed_url.set(username=config_user, password=config_pass)
        return parsed_url.url

    @abc.abstractproperty
    def info_commands(self):
        # type: () -> Mapping[Text, Argv]
        """
    `   Mapping from `RepoInfo` attribute name (except `type`) to command which acquires it
        """
        pass

    def get_repository_copy_info(self, path):
        """
        Get `RepoInfo` instance from copy of clone in `path`
        """
        path = Text(path)
        commands_result = {
            name: command.get_output(cwd=path)
            # name: subprocess.check_output(command.split(), cwd=path).decode().strip()
            for name, command in self.info_commands.items()
        }
        return RepoInfo(type=self.executable_name, **commands_result)


class Git(VCS):
    executable_name = "git"
    main_branch = "master"
    clone_flags = ("--quiet", "--recursive")
    checkout_flags = ("--force",)
    COMMAND_ENV = {
        # do not prompt for password
        "GIT_TERMINAL_PROMPT": "0",
        # do not prompt for ssh key passphrase
        "GIT_SSH_COMMAND": "ssh -oBatchMode=yes",
    }

    @staticmethod
    def remote_branch_name(branch):
        return "origin/{}".format(branch)

    def executable_not_found_error_help(self):
        return 'Cannot find "{}" executable. {}'.format(
            self.executable_name,
            select_for_platform(
                linux="You can install it by running: sudo apt-get install {}".format(
                    self.executable_name
                ),
                windows="You can download it here: {}".format(
                    "https://gitforwindows.org/"
                ),
            ),
        )

    def pull(self):
        self.call("fetch", "--all", "--recurse-submodules", cwd=self.location)

    def checkout(self):  # type: () -> None
        """
        Checkout repository at specified revision
        """
        self.call("checkout", self.revision, *self.checkout_flags, cwd=self.location)
        try:
            self.call("submodule", "update", "--recursive", cwd=self.location)
        except:
            pass

    info_commands = dict(
        url=Argv(executable_name, "ls-remote", "--get-url", "origin"),
        branch=Argv(executable_name, "rev-parse", "--abbrev-ref", "HEAD"),
        commit=Argv(executable_name, "rev-parse", "HEAD"),
        root=Argv(executable_name, "rev-parse", "--show-toplevel"),
    )

    patch_base = ("apply",)


class Hg(VCS):
    executable_name = "hg"
    main_branch = "default"
    checkout_flags = ("--clean",)
    patch_base = ("import", "--no-commit")

    def executable_not_found_error_help(self):
        return 'Cannot find "{}" executable. {}'.format(
            self.executable_name,
            select_for_platform(
                linux="You can install it by running: sudo apt-get install {}".format(
                    self.executable_name
                ),
                windows="You can download it here: {}".format(
                    "https://www.mercurial-scm.org/wiki/Download"
                ),
            ),
        )

    def pull(self):
        self.call(
            "pull",
            self.url_with_auth,
            cwd=self.location,
            *(("-r", self.revision) if self.revision else ())
        )

    info_commands = dict(
        url=Argv(executable_name, "paths", "--verbose"),
        branch=Argv(executable_name, "--debug", "id", "-b"),
        commit=Argv(executable_name, "--debug", "id", "-i"),
        root=Argv(executable_name, "root"),
    )


def clone_repository_cached(session, execution, destination):
    # type: (Session, ExecutionInfo, Path) -> Tuple[VCS, RepoInfo]
    """
    Clone a remote repository.
    :param execution: execution info
    :param destination: directory to clone to (in which a directory for the repository will be created)
    :param session: program session
    :return: repository information
    :raises: CommandFailedError if git/hg is not installed
    """
    repo_url = execution.repository  # type: str
    parsed_url = furl(repo_url)
    no_password_url = parsed_url.copy().remove(password=True).url

    clone_folder_name = Path(str(furl(repo_url).path)).name  # type: str
    clone_folder = Path(destination) / clone_folder_name

    standalone_mode = session.config.get("agent.standalone_mode", False)
    if standalone_mode:
        cached_repo_path = clone_folder
    else:
        cached_repo_path = (
            Path(session.config["agent.vcs_cache.path"]).expanduser()
            / "{}.{}".format(clone_folder_name, md5(ensure_binary(repo_url)).hexdigest())
            / clone_folder_name
        )  # type: Path

    vcs = VcsFactory.create(
        session, execution_info=execution, location=cached_repo_path
    )
    if not find_executable(vcs.executable_name):
        raise CommandFailedError(vcs.executable_not_found_error_help())

    if not standalone_mode:
        if session.config["agent.vcs_cache.enabled"] and cached_repo_path.exists():
            print('Using cached repository in "{}"'.format(cached_repo_path))

        else:
            print("cloning: {}".format(no_password_url))
            rm_tree(cached_repo_path)
            # We clone the entire repository, not a specific branch
            vcs.clone()  # branch=execution.branch)

        vcs.pull()
        rm_tree(destination)
        shutil.copytree(Text(cached_repo_path), Text(clone_folder))
        if not clone_folder.is_dir():
            raise CommandFailedError(
                "copying of repository failed: from {} to {}".format(
                    cached_repo_path, clone_folder
                )
            )

    # checkout in the newly copy destination
    vcs.location = Text(clone_folder)
    vcs.checkout()

    repo_info = vcs.get_repository_copy_info(clone_folder)

    # make sure we have no user/pass in the returned repository structure
    repo_info = attr.evolve(repo_info, url=no_password_url)

    return vcs, repo_info
