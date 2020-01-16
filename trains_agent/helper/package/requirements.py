from __future__ import absolute_import, unicode_literals

import operator
import os
import re
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from itertools import chain, starmap
from operator import itemgetter
from os import path
from typing import Text, List, Type, Optional, Tuple, Dict

from packaging import version as packaging_version
from pathlib2 import Path
from pyhocon import ConfigTree
from requirements import parse
# noinspection PyPackageRequirements
from requirements.requirement import Requirement

import six
from trains_agent.definitions import PIP_EXTRA_INDICES
from trains_agent.helper.base import warning, is_conda, which, join_lines, is_windows_platform
from trains_agent.helper.process import Argv, PathLike
from trains_agent.session import Session, normalize_cuda_version
from .translator import RequirementsTranslator


class SpecsResolutionError(Exception):
    pass


class FatalSpecsResolutionError(Exception):
    pass


@six.python_2_unicode_compatible
class MarkerRequirement(object):

    def __init__(self, req):  # type: (Requirement) -> None
        self.req = req

    @property
    def marker(self):
        match = re.search(r';\s*(.*)', self.req.line)
        if match:
            return match.group(1)
        return None

    def tostr(self, markers=True):
        if not self.uri:
            parts = [self.name or self.line]

            if self.extras:
                parts.append('[{0}]'.format(','.join(sorted(self.extras))))

            if self.specifier:
                parts.append(self.format_specs())

        else:
            parts = [self.uri]

        if markers and self.marker:
            parts.append('; {0}'.format(self.marker))

        return ''.join(parts)

    __str__ = tostr

    def __repr__(self):
        return '{self.__class__.__name__}[{self}]'.format(self=self)

    def format_specs(self):
        return ','.join(starmap(operator.add, self.specs))

    def __getattr__(self, item):
        return getattr(self.req, item)

    @property
    def specs(self):  # type: () -> List[Tuple[Text, Text]]
        return self.req.specs

    @specs.setter
    def specs(self, value):  # type: (List[Tuple[Text, Text]]) -> None
        self.req.specs = value

    def fix_specs(self):
        def solve_by(func, op_is, specs):
            return func([(op, version) for op, version in specs if op == op_is])

        def solve_equal(specs):
            if len(set(version for _, version in self.specs)) > 1:
                raise SpecsResolutionError('more than one "==" spec: {}'.format(specs))
            return specs
        greater = solve_by(lambda specs: [max(specs, key=itemgetter(1))], '<=', self.specs)
        smaller = solve_by(lambda specs: [min(specs, key=itemgetter(1))], '>=', self.specs)
        equal = solve_by(solve_equal, '==', self.specs)
        if equal:
            self.specs = equal
        else:
            self.specs = greater + smaller


@six.add_metaclass(ABCMeta)
class RequirementSubstitution(object):

    _pip_extra_index_url = PIP_EXTRA_INDICES

    def __init__(self, session):
        # type: (Session) -> ()
        self._session = session
        self.config = session.config  # type: ConfigTree
        self.suffix = '.post{config[agent.cuda_version]}.dev{config[agent.cudnn_version]}'.format(config=self.config)
        self.package_manager = self.config['agent.package_manager.type']

    @abstractmethod
    def match(self, req):  # type: (MarkerRequirement) -> bool
        """
        Returns whether a requirement needs to be modified by this substitution.
        """
        pass

    @abstractmethod
    def replace(self, req):  # type: (MarkerRequirement) -> Text
        """
        Replace a requirement
        """
        pass

    def post_install(self):
        pass

    @classmethod
    def get_pip_version(cls, package):
        output = Argv(
            'pip',
            'search',
            package,
            *(chain.from_iterable(('-i', x) for x in cls._pip_extra_index_url))
        ).get_output()
        # ad-hoc pattern to duplicate the behavior of the old code
        return re.search(r'{} \((\d+\.\d+\.[^.]+)'.format(package), output).group(1)

    @property
    def cuda_version(self):
        return self.config['agent.cuda_version']

    @property
    def cudnn_version(self):
        return self.config['agent.cudnn_version']


class SimpleSubstitution(RequirementSubstitution):

    @property
    @abstractmethod
    def name(self):
        pass

    def match(self, req):  # type: (MarkerRequirement) -> bool
        return (self.name == req.name or (
            req.uri and
            re.match(r'https?://', req.uri) and
            self.name in req.uri
        ))

    def replace(self, req):  # type: (MarkerRequirement) -> Text
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        if req.uri:
            return re.sub(
                r'({})(.*?)(-cp)'.format(self.name),
                r'\1\2{}\3'.format(self.suffix),
                req.uri,
                count=1)

        if req.specs:
            _, version_number = req.specs[0]
            assert packaging_version.parse(version_number)
        else:
            version_number = self.get_pip_version(self.name)

        req.specs = [('==', version_number + self.suffix)]
        return Text(req)

    def replace_back(self, list_of_requirements):  # type: (Dict) -> Dict
        """
        :param list_of_requirements: {'pip': ['a==1.0', ]}
        :return: {'pip': ['a==1.0', ]}
        """
        return list_of_requirements


@six.add_metaclass(ABCMeta)
class CudaSensitiveSubstitution(SimpleSubstitution):

    def match(self, req):  # type: (MarkerRequirement) -> bool
        return self.cuda_version and self.cudnn_version and \
            super(CudaSensitiveSubstitution, self).match(req)


class CudaNotFound(Exception):
    pass


class RequirementsManager(object):

    def __init__(self, session, base_interpreter=None):
        # type: (Session, PathLike) -> ()
        self._session = session
        self.config = deepcopy(session.config)  # type: ConfigTree
        self.handlers = []  # type: List[RequirementSubstitution]
        agent = self.config['agent']
        self.active = not agent.get('cpu_only', False)
        self.found_cuda = False
        if self.active:
            try:
                agent['cuda_version'], agent['cudnn_version'] = self.get_cuda_version(self.config)
                self.found_cuda = True
            except Exception:
                # if we have a cuda version, it is good enough (we dont have to have cudnn version)
                if agent.get('cuda_version'):
                    self.found_cuda = True
        pip_cache_dir = Path(self.config["agent.pip_download_cache.path"]).expanduser() / (
            'cu'+agent['cuda_version'] if self.found_cuda else 'cpu')
        self.translator = RequirementsTranslator(session, interpreter=base_interpreter,
                                                 cache_dir=pip_cache_dir.as_posix())

    def register(self, cls):  # type: (Type[RequirementSubstitution]) -> None
        self.handlers.append(cls(self._session))

    def _replace_one(self, req):  # type: (MarkerRequirement) -> Optional[Text]
        match = re.search(r';\s*(.*)', Text(req))
        if match:
            req.markers = match.group(1).split(',')
        if not self.active:
            return None
        for handler in self.handlers:
            if handler.match(req):
                return handler.replace(req)
        return None

    def replace(self, requirements):  # type: (Text) -> Text
        def safe_parse(req_str):
            try:
                return next(parse(req_str))
            except Exception as ex:
                return Requirement(req_str)

        parsed_requirements = tuple(
            map(
                MarkerRequirement,
                [safe_parse(line) for line in (requirements.splitlines()
                                               if isinstance(requirements, six.text_type) else requirements)]
            )
        )
        if not parsed_requirements:
            # return the original requirements just in case
            return requirements

        def replace_one(i, req):
            # type: (int, MarkerRequirement) -> Optional[Text]
            try:
                return self._replace_one(req)
            except FatalSpecsResolutionError:
                warning('could not resolve python wheel replacement for {}'.format(req))
                raise
            except Exception:
                warning('could not resolve python wheel replacement for \"{}\", '
                        'using original requirements line: {}'.format(req, i))
                return None

        new_requirements = tuple(replace_one(i, req) for i, req in enumerate(parsed_requirements))
        conda = is_conda(self.config)
        result = map(
            lambda x, y: (x if x is not None else y.tostr(markers=not conda)),
            new_requirements,
            parsed_requirements
        )
        if not conda:
            result = map(self.translator.translate, result)
        return join_lines(result)

    def post_install(self):
        for h in self.handlers:
            try:
                h.post_install()
            except Exception as ex:
                print('RequirementsManager handler {} raised exception: {}'.format(h, ex))

    def replace_back(self, requirements):
        for h in self.handlers:
            try:
                requirements = h.replace_back(requirements)
            except Exception:
                pass
        return requirements

    @staticmethod
    def get_cuda_version(config):  # type: (ConfigTree) -> (Text, Text)
        # we assume os.environ already updated the config['agent.cuda_version'] & config['agent.cudnn_version']
        cuda_version = config['agent.cuda_version']
        cudnn_version = config['agent.cudnn_version']
        if cuda_version and cudnn_version:
            return normalize_cuda_version(cuda_version), normalize_cuda_version(cudnn_version)

        if not cuda_version and is_windows_platform():
            try:
                cuda_vers = [int(k.replace('CUDA_PATH_V', '').replace('_', '')) for k in os.environ.keys()
                             if k.startswith('CUDA_PATH_V')]
                cuda_vers = max(cuda_vers)
                if cuda_vers > 40:
                    cuda_version = cuda_vers
            except:
                pass

        if not cuda_version:
            try:
                try:
                    nvcc = 'nvcc.exe' if is_windows_platform() else 'nvcc'
                    if is_windows_platform() and 'CUDA_PATH' in os.environ:
                        nvcc = os.path.join(os.environ['CUDA_PATH'], nvcc)

                    output = Argv(nvcc, '--version').get_output()
                except OSError:
                    raise CudaNotFound('nvcc not found')
                match = re.search(r'release (.{3})', output).group(1)
                cuda_version = Text(int(float(match) * 10))
            except:
                pass

        if not cuda_version:
            try:
                try:
                    output = Argv('nvidia-smi',).get_output()
                except OSError:
                    raise CudaNotFound('nvcc not found')
                match = re.search(r'CUDA Version: ([0-9]+).([0-9]+)', output)
                match = match.group(1)+'.'+match.group(2)
                cuda_version = Text(int(float(match) * 10))
            except:
                pass

        if not cudnn_version:
            try:
                cuda_lib = which('nvcc')
                if is_windows_platform:
                    cudnn_h = path.sep.join(cuda_lib.split(path.sep)[:-2] + ['include', 'cudnn.h'])
                else:
                    cudnn_h = path.join(path.sep, *(cuda_lib.split(path.sep)[:-2] + ['include', 'cudnn.h']))

                cudnn_major, cudnn_minor = None, None
                try:
                    include_file = open(cudnn_h)
                except OSError:
                    raise CudaNotFound('Could not read cudnn.h')
                with include_file:
                    for line in include_file:
                        if 'CUDNN_MAJOR' in line:
                            cudnn_major = line.split()[-1]
                        if 'CUDNN_MINOR' in line:
                            cudnn_minor = line.split()[-1]
                        if cudnn_major and cudnn_minor:
                            break
                cudnn_version = cudnn_major + (cudnn_minor or '0')
            except:
                pass

        return (normalize_cuda_version(cuda_version or 0),
                normalize_cuda_version(cudnn_version or 0))

