from __future__ import unicode_literals

import json
import re
import shutil
import subprocess
from distutils.spawn import find_executable
from functools import partial
from itertools import chain
from typing import Text, Iterable, Union, Dict, Set, Sequence, Any

import six
import yaml
from time import time
from attr import attrs, attrib, Factory
from pathlib2 import Path
from semantic_version import Version
from requirements import parse
from requirements.requirement import Requirement

from trains_agent.errors import CommandFailedError
from trains_agent.helper.base import rm_tree, NonStrictAttrs, select_for_platform, is_windows_platform
from trains_agent.helper.process import Argv, Executable, DEVNULL, CommandSequence, PathLike
from trains_agent.session import Session
from .base import PackageManager
from .pip_api.venv import VirtualenvPip
from .requirements import RequirementsManager, MarkerRequirement

package_normalize = partial(re.compile(r"""\[version=['"](.*)['"]\]""").sub, r"\1")


def package_set(packages):
    return set(map(package_normalize, packages))


def _package_diff(path, packages):
    # type: (Union[Path, Text], Iterable[Text]) -> Set[Text]
    return package_set(Path(path).read_text().splitlines()) - package_set(packages)


class CondaPip(VirtualenvPip):
    def __init__(self, source=None, *args, **kwargs):
        super(CondaPip, self).__init__(*args, interpreter=Path(kwargs.get('path'), "python.exe") \
            if is_windows_platform() and kwargs.get('path') else None, **kwargs)
        self.source = source

    def run_with_env(self, command, output=False, **kwargs):
        if not self.source:
            return super(CondaPip, self).run_with_env(command, output=output, **kwargs)
        command = CommandSequence(self.source, Argv("pip", *command))
        return (command.get_output if output else command.check_call)(
            stdin=DEVNULL, **kwargs
        )


class CondaAPI(PackageManager):

    """
    A programmatic interface for controlling conda
    """

    MINIMUM_VERSION = Version("4.3.30", partial=True)

    def __init__(self, session, path, python, requirements_manager):
        # type: (Session, PathLike, float, RequirementsManager) -> None
        """
        :param python: base python version to use (e.g python3.6)
        :param path: path of env
        """
        self.session = session
        self.python = python
        self.source = None
        self.requirements_manager = requirements_manager
        self.path = path
        self.extra_channels = self.session.config.get('agent.package_manager.conda_channels', [])
        self.pip = CondaPip(
            session=self.session,
            source=self.source,
            python=self.python,
            requirements_manager=self.requirements_manager,
            path=self.path,
        )
        self.conda = (
            find_executable("conda")
            or Argv(select_for_platform(windows="where", linux="which"), "conda").get_output(shell=True).strip()
        )
        try:
            output = Argv(self.conda, "--version").get_output(stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as ex:
            raise CommandFailedError(
                "Unable to determine conda version: {ex}, output={ex.output}".format(
                    ex=ex
                )
            )
        self.conda_version = self.get_conda_version(output)
        if Version(self.conda_version, partial=True) < self.MINIMUM_VERSION:
            raise CommandFailedError(
                "conda version '{}' is smaller than minimum supported conda version '{}'".format(
                    self.conda_version, self.MINIMUM_VERSION
                )
            )

    @staticmethod
    def get_conda_version(output):
        match = re.search(r"(\d+\.){0,2}\d+", output)
        if not match:
            raise CommandFailedError("Unidentified conda version string:", output)
        return match.group(0)

    @property
    def bin(self):
        return self.pip.bin

    def upgrade_pip(self):
        return self.pip.upgrade_pip()

    def create(self):
        """
        Create a new environment
        """
        output = Argv(
            self.conda,
            "create",
            "--yes",
            "--mkdir",
            "--prefix",
            self.path,
            "python={}".format(self.python),
        ).get_output(stderr=DEVNULL)
        match = re.search(
            r"\W*(.*activate) ({})".format(re.escape(str(self.path))), output
        )
        self.source = self.pip.source = (
            tuple(match.group(1).split()) + (match.group(2),)
            if match
            else ("activate", self.path)
        )
        conda_env = Path(self.conda).parent.parent / 'etc' / 'profile.d' / 'conda.sh'
        if conda_env.is_file() and not is_windows_platform():
            self.source = self.pip.source = CommandSequence(('source', conda_env.as_posix()), self.source)

        # install cuda toolkit
        try:
            cuda_version = float(int(self.session.config['agent.cuda_version'])) / 10.0
            if cuda_version > 0:
                self._install('cudatoolkit={:.1f}'.format(cuda_version))
        except Exception:
            pass
        return self

    def remove(self):
        """
        Delete a conda environment.
        Use 'conda env remove', then 'rm_tree' to be safe.

        Conda seems to load "vcruntime140.dll" from all its environment on startup.
        This means environment have to be deleted using 'conda env remove'.
        If necessary, conda can be fooled into deleting a partially-deleted environment by creating an empty file
        in '<ENV>\conda-meta\history' (value found in 'conda.gateways.disk.test.PREFIX_MAGIC_FILE').
        Otherwise, it complains that said directory is not a conda environment.

        See: https://github.com/conda/conda/issues/7682
        """
        try:
            self._run_command(("env", "remove", "-p", self.path))
        except Exception:
            pass
        rm_tree(self.path)
        # if we failed removing the path, change it's name
        if is_windows_platform() and Path(self.path).exists():
            try:
                Path(self.path).rename(Path(self.path).as_posix() + '_' + str(time()))
            except Exception:
                pass

    def _install_from_file(self, path):
        """
        Install packages from requirement file.
        """
        self._install("--file", path)

    def _install(self, *args):
        # type: (*PathLike) -> ()
        channels_args = tuple(
            chain.from_iterable(("-c", channel) for channel in self.extra_channels)
        )
        self._run_command(("install", "-p", self.path) + channels_args + args)

    def _get_pip_packages(self, packages):
        # type: (Iterable[Text]) -> Sequence[Text]
        """
        Return subset of ``packages`` which are not available on conda
        """
        pips = []
        while True:
            with self.temp_file("conda_reqs", packages) as path:
                try:
                    self._install_from_file(path)
                except PackageNotFoundError as e:
                    pips.append(e.pkg)
                    packages = _package_diff(path, {e.pkg})
                else:
                    break
        return pips

    def install_packages(self, *packages):
        # type: (*Text) -> ()
        return self._install(*packages)

    def uninstall_packages(self, *packages):
        return self._run_command(("uninstall", "-p", self.path))

    def install_from_file(self, path):
        """
        Try to install packages from conda. Install packages which are not available from conda with pip.
        """
        try:
            self._install_from_file(path)
            return
        except PackageNotFoundError as e:
            pip_packages = [e.pkg]
        except PackagesNotFoundError as e:
            pip_packages = package_set(e.packages)
        with self.temp_file("conda_reqs", _package_diff(path, pip_packages)) as reqs:
            self.install_from_file(reqs)
        with self.temp_file("pip_reqs", pip_packages) as reqs:
            self.pip.install_from_file(reqs)

    def freeze(self):
        # result = yaml.load(
        #     self._run_command((self.conda, "env", "export", "-p", self.path), raw=True)
        # )
        # for key in "name", "prefix":
        #     result.pop(key, None)
        # freeze = {"conda": result}
        # try:
        #     freeze["pip"] = result["dependencies"][-1]["pip"]
        # except (TypeError, KeyError):
        #     freeze["pip"] = []
        # else:
        #     del result["dependencies"][-1]
        # return freeze
        return self.pip.freeze()

    def load_requirements(self, requirements):
        # create new environment file
        conda_env = dict()
        conda_env['channels'] = self.extra_channels
        reqs = []
        if isinstance(requirements['pip'], six.string_types):
            requirements['pip'] = requirements['pip'].split('\n')
        has_torch = False
        has_matplotlib = False
        try:
            cuda_version = int(self.session.config.get('agent.cuda_version', 0))
        except:
            cuda_version = 0

        for r in requirements['pip']:
            marker = list(parse(r))
            if marker:
                m = MarkerRequirement(marker[0])
                if m.req.name.lower() == 'matplotlib':
                    has_matplotlib = True
                elif m.req.name.lower().startswith('torch'):
                    has_torch = True

                if m.req.name.lower() in ('torch', 'pytorch'):
                    has_torch = True
                    m.req.name = 'pytorch'

                if m.req.name.lower() in ('tensorflow_gpu', 'tensorflow-gpu', 'tensorflow'):
                    has_torch = True
                    m.req.name = 'tensorflow-gpu' if cuda_version > 0 else 'tensorflow'

                reqs.append(m)
        pip_requirements = []

        # Conda requirements Hacks:
        if has_matplotlib:
            reqs.append(MarkerRequirement(Requirement.parse('graphviz')))
            reqs.append(MarkerRequirement(Requirement.parse('kiwisolver')))
        if has_torch and cuda_version == 0:
            reqs.append(MarkerRequirement(Requirement.parse('cpuonly')))

        while reqs:
            conda_env['dependencies'] = [r.tostr().replace('==', '=') for r in reqs]
            with self.temp_file("conda_env", yaml.dump(conda_env), suffix=".yml") as name:
                print('Conda: Trying to install requirements:\n{}'.format(conda_env['dependencies']))
                result = self._run_command(
                    ("env", "update", "-p", self.path, "--file", name)
                )
            # check if we need to remove specific packages
            bad_req = self._parse_conda_result_bad_packges(result)
            if not bad_req:
                break

            solved = False
            for bad_r in bad_req:
                name = bad_r.split('[')[0].split('=')[0]
                # look for name in requirements
                for r in reqs:
                    if r.name.lower() == name.lower():
                        pip_requirements.append(r)
                        reqs.remove(r)
                        solved = True
                        break

            # we couldn't remove even one package,
            # nothing we can do but try pip
            if not solved:
                pip_requirements.extend(reqs)
                break

        if pip_requirements:
            try:
                pip_req_str = [r.tostr() for r in pip_requirements]
                print('Conda: Installing requirements: step 2 - using pip:\n{}'.format(pip_req_str))
                self.pip.load_requirements('\n'.join(pip_req_str))
            except Exception as e:
                print(e)
                raise e

        self.requirements_manager.post_install()
        return True

    def _parse_conda_result_bad_packges(self, result_dict):
        if not result_dict:
            return None

        if 'bad_deps' in result_dict and result_dict['bad_deps']:
            return result_dict['bad_deps']

        if result_dict.get('error'):
            error_lines = result_dict['error'].split('\n')
            if error_lines[0].strip().lower().startswith("unsatisfiableerror:"):
                empty_lines = [i for i, l in enumerate(error_lines) if not l.strip()]
                if len(empty_lines) >= 2:
                    deps = error_lines[empty_lines[0]+1:empty_lines[1]]
                    try:
                        return yaml.load('\n'.join(deps))
                    except:
                        return None
        return None

    def _run_command(self, command, raw=False, **kwargs):
        # type: (Iterable[Text], bool, Any) -> Union[Dict, Text]
        """
        Run a conda command, returning JSON output.
        The command is prepended with 'conda' and run with JSON output flags.
        :param command: command to run
        :param raw: return text output and don't change command
        :param kwargs: kwargs for Argv.get_output()
        :return: JSON output or text output
        """
        def escape_ansi(line):
            ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
            return ansi_escape.sub('', line)

        command = Argv(*command)  # type: Executable
        if not raw:
            command = (self.conda,) + command + ("--quiet", "--json")
        try:
            print('Executing Conda: {}'.format(command.serialize()))
            result = command.get_output(stdin=DEVNULL, **kwargs)
        except Exception as e:
            if raw:
                raise
            result = e.output if hasattr(e, 'output') else ''
        if raw:
            return result

        result = json.loads(escape_ansi(result)) if result else {}
        if result.get('success', False):
            print('Pass')
        elif result.get('error'):
            print('Conda error: {}'.format(result.get('error')))
        return result

    def get_python_command(self, extra=()):
        return CommandSequence(self.source, self.pip.get_python_command(extra=extra))


# enable hashing with cmp=False because pdb fails on unhashable exceptions
exception = attrs(str=True, cmp=False)


@exception
class CondaException(Exception, NonStrictAttrs):
    command = attrib()
    message = attrib(default=None)


@exception
class UnknownCondaError(CondaException):
    data = attrib(default=Factory(dict))


@exception
class PackagesNotFoundError(CondaException):
    """
    Conda 4.5 exception - this reports all missing packages.
    """

    packages = attrib(default=())


@exception
class PackageNotFoundError(CondaException):
    """
    Conda 4.3 exception - this reports one missing package at a time,
                          as a singleton YAML list.
    """

    pkg = attrib(default="", converter=lambda val: yaml.load(val)[0].replace(" ", ""))
