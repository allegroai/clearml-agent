from __future__ import unicode_literals

import re
import sys
from furl import furl
import urllib.parse
from operator import itemgetter
from html.parser import HTMLParser
from typing import Text

import attr
import requests

import six
from .requirements import SimpleSubstitution, FatalSpecsResolutionError, SimpleVersion

OS_TO_WHEEL_NAME = {"linux": "linux_x86_64", "windows": "win_amd64"}


def os_to_wheel_name(x):
    return OS_TO_WHEEL_NAME[x]


def fix_version(version):
    def replace(nums, prerelease):
        if prerelease:
            return "{}-{}".format(nums, prerelease)
        return nums

    return re.sub(
        r"(\d+(?:\.\d+){,2})(?:\.(.*))?",
        lambda match: replace(*match.groups()),
        version,
    )


class LinksHTMLParser(HTMLParser):
    def __init__(self):
        super(LinksHTMLParser, self).__init__()
        self.links = []

    def handle_data(self, data):
        if data and data.strip():
            self.links += [data]


@attr.s
class PytorchWheel(object):
    os_name = attr.ib(type=str, converter=os_to_wheel_name)
    cuda_version = attr.ib(converter=lambda x: "cu{}".format(x) if x else "cpu")
    python = attr.ib(type=str, converter=lambda x: str(x).replace(".", ""))
    torch_version = attr.ib(type=str, converter=fix_version)

    url_template = (
        "http://download.pytorch.org/whl/"
        "{0.cuda_version}/torch-{0.torch_version}-cp{0.python}-cp{0.python}m{0.unicode}-{0.os_name}.whl"
    )

    def __attrs_post_init__(self):
        self.unicode = "u" if self.python.startswith("2") else ""

    def make_url(self):
        # type: () -> Text
        return self.url_template.format(self)


class PytorchResolutionError(FatalSpecsResolutionError):
    pass


class SimplePytorchRequirement(SimpleSubstitution):
    name = "torch"

    packages = ("torch", "torchvision", "torchaudio")

    page_lookup_template = 'https://download.pytorch.org/whl/cu{}/torch_stable.html'
    nightly_page_lookup_template = 'https://download.pytorch.org/whl/nightly/cu{}/torch_nightly.html'
    torch_page_lookup = {
        0: 'https://download.pytorch.org/whl/cpu/torch_stable.html',
        80: 'https://download.pytorch.org/whl/cu80/torch_stable.html',
        90: 'https://download.pytorch.org/whl/cu90/torch_stable.html',
        92: 'https://download.pytorch.org/whl/cu92/torch_stable.html',
        100: 'https://download.pytorch.org/whl/cu100/torch_stable.html',
        101: 'https://download.pytorch.org/whl/cu101/torch_stable.html',
    }

    def __init__(self, *args, **kwargs):
        super(SimplePytorchRequirement, self).__init__(*args, **kwargs)
        self._matched = False

    def match(self, req):
        # match both any of out packages
        return req.name in self.packages

    def replace(self, req):
        """
        Replace a requirement
        :raises: ValueError if version is pre-release
        """
        # Get rid of +cpu +cu?? etc.
        try:
            req.specs[0] = (req.specs[0][0], req.specs[0][1].split('+')[0])
        except:
            pass
        self._matched = True
        return Text(req)

    def matching_done(self, reqs, package_manager):
        # type: (Sequence[MarkerRequirement], object) -> ()
        if not self._matched:
            return
        # TODO: add conda channel support
        from .pip_api.system import SystemPip
        if package_manager and isinstance(package_manager, SystemPip):
            extra_url, _ = self.get_torch_page(self.cuda_version)
            package_manager.add_extra_install_flags(('-f', extra_url))

    @classmethod
    def get_torch_page(cls, cuda_version, nightly=False):
        try:
            cuda = int(cuda_version)
        except:
            cuda = 0

        if nightly:
            # then try the nightly builds, it might be there...
            torch_url = cls.nightly_page_lookup_template.format(cuda)
            try:
                if requests.get(torch_url, timeout=10).ok:
                    cls.torch_page_lookup[cuda] = torch_url
                    return cls.torch_page_lookup[cuda], cuda
            except Exception:
                pass
            return

        # first check if key is valid
        if cuda in cls.torch_page_lookup:
            return cls.torch_page_lookup[cuda], cuda

        # then try a new cuda version page
        torch_url = cls.page_lookup_template.format(cuda)
        try:
            if requests.get(torch_url, timeout=10).ok:
                cls.torch_page_lookup[cuda] = torch_url
                return cls.torch_page_lookup[cuda], cuda
        except Exception:
            pass

        keys = sorted(cls.torch_page_lookup.keys(), reverse=True)
        for k in keys:
            if k <= cuda:
                return cls.torch_page_lookup[k], k
        # return default - zero
        return cls.torch_page_lookup[0], 0


class PytorchRequirement(SimpleSubstitution):

    name = "torch"
    packages = ("torch", "torchvision", "torchaudio")

    def __init__(self, *args, **kwargs):
        os_name = kwargs.pop("os_override", None)
        super(PytorchRequirement, self).__init__(*args, **kwargs)
        self.log = self._session.get_logger(__name__)
        self.package_manager = self.config["agent.package_manager.type"].lower()
        self.os = os_name or self.get_platform()
        self.cuda = "cuda{}".format(self.cuda_version).lower()
        self.python_version_string = str(self.config["agent.default_python"])
        self.python_major_minor_str = '.'.join(self.python_version_string.split('.')[:2])
        if '.' not in self.python_major_minor_str:
            raise PytorchResolutionError(
                "invalid python version {!r} defined in configuration file, key 'agent.default_python': "
                "must have both major and minor parts of the version (for example: '3.7')".format(
                    self.python_version_string
                )
            )
        self.python = "python{}".format(self.python_major_minor_str)

        self.exceptions = [
            PytorchResolutionError(message)
            for message in (
                None,
                'cuda version "{}" is not supported'.format(self.cuda),
                'python version "{}" is not supported'.format(
                    self.python_version_string
                ),
            )
        ]

        try:
            self.validate_python_version()
        except PytorchResolutionError as e:
            self.log.warn("will not be able to install pytorch wheels: %s", e.args[0])

        self._original_req = []

    @property
    def is_conda(self):
        return self.package_manager == "conda"

    @property
    def is_pip(self):
        return not self.is_conda

    def validate_python_version(self):
        """
        Make sure python version has both major and minor versions as required for choosing pytorch wheel
        """
        if self.is_pip and not self.python_major_minor_str:
            raise PytorchResolutionError(
                "invalid python version {!r} defined in configuration file, key 'agent.default_python': "
                "must have both major and minor parts of the version (for example: '3.7')".format(
                    self.python_version_string
                )
            )

    def match(self, req):
        return req.name in self.packages

    @staticmethod
    def get_platform():
        if sys.platform == "linux":
            return "linux"
        if sys.platform == "win32" or sys.platform == "cygwin":
            return "windows"
        if sys.platform == "darwin":
            return "macos"
        raise RuntimeError("unrecognized OS")

    def _get_link_from_torch_page(self, req, torch_url):
        links_parser = LinksHTMLParser()
        links_parser.feed(requests.get(torch_url, timeout=10).text)
        platform_wheel = "win" if self.get_platform() == "windows" else self.get_platform()
        py_ver = self.python_major_minor_str.replace('.', '')
        url = None
        last_v = None
        # search for our package
        for l in links_parser.links:
            parts = l.split('/')[-1].split('-')
            if len(parts) < 5:
                continue
            if parts[0] != req.name:
                continue
            # version (ignore +cpu +cu92 etc. + is %2B in the file link)
            # version ignore .postX suffix (treat as regular version)
            try:
                v = str(parts[1].split('%')[0].split('+')[0])
            except Exception:
                continue
            if not req.compare_version(v) or \
                    (last_v and SimpleVersion.compare_versions(last_v, '>', v, ignore_sub_versions=False)):
                continue
            if not parts[2].endswith(py_ver):
                continue
            if platform_wheel not in parts[4]:
                continue
            url = '/'.join(torch_url.split('/')[:-1] + l.split('/'))
            last_v = v
            # if we found an exact match, use it
            try:
                if req.specs[0][0] == '==' and \
                        SimpleVersion.compare_versions(req.specs[0][1], '==', v, ignore_sub_versions=False):
                    break
            except:
                pass

        return url

    def get_url_for_platform(self, req):
        # check if package is already installed with system packages
        try:
            if self.config.get("agent.package_manager.system_site_packages"):
                from pip._internal.commands.show import search_packages_info
                installed_torch = list(search_packages_info([req.name]))
                # notice the comparision order, the first part will make sure we have a valid installed package
                if installed_torch[0]['version'] and req.compare_version(installed_torch[0]['version']):
                    print('PyTorch: requested "{}" version {}, using pre-installed version {}'.format(
                        req.name, req.specs[0] if req.specs else 'unspecified', installed_torch[0]['version']))
                    # package already installed, do nothing
                    return str(req), True
        except:
            pass

        # make sure we have a specific version to retrieve
        if not req.specs:
            req.specs = [('>', '0')]

        try:
            req.specs[0] = (req.specs[0][0], req.specs[0][1].split('+')[0])
        except:
            pass
        op, version = req.specs[0]
        # assert op == "=="

        torch_url, torch_url_key = SimplePytorchRequirement.get_torch_page(self.cuda_version)
        url = self._get_link_from_torch_page(req, torch_url)
        if not url and self.config.get("agent.package_manager.torch_nightly"):
            torch_url, torch_url_key = SimplePytorchRequirement.get_torch_page(self.cuda_version, nightly=True)
            url = self._get_link_from_torch_page(req, torch_url)
        # try one more time, with a lower cuda version (never fallback to CPU):
        while not url and torch_url_key > 0:
            previous_cuda_key = torch_url_key
            torch_url, torch_url_key = SimplePytorchRequirement.get_torch_page(int(torch_url_key)-1)
            # never fallback to CPU
            if torch_url_key < 1:
                print('Warning! Could not locate PyTorch version {} matching CUDA version {}'.format(
                    req, previous_cuda_key))
                raise ValueError('Could not locate PyTorch version {} matching CUDA version {}'.format(
                    req, self.cuda_version))
            print('Warning! Could not locate PyTorch version {} matching CUDA version {}, trying CUDA version {}'.format(
                req, previous_cuda_key, torch_url_key))
            url = self._get_link_from_torch_page(req, torch_url)

        if not url:
            url = PytorchWheel(
                torch_version=fix_version(version),
                python=self.python_major_minor_str.replace('.', ''),
                os_name=self.os,
                cuda_version=self.cuda_version,
            ).make_url()
        if url:
            # normalize url (sometimes we will get ../ which we should not...
            url = '/'.join(url.split('/')[:3]) + urllib.parse.quote(str(furl(url).path.normalize()))

        self.log.debug("checking url: %s", url)
        return url, requests.head(url, timeout=10).ok

    @staticmethod
    def match_version(req, options):
        versioned_options = sorted(
            ((fix_version(key), value) for key, value in options.items()),
            key=itemgetter(0),
            reverse=True,
        )
        req.specs = [(op, fix_version(version)) for op, version in req.specs]

        try:
            return next(
                replacement
                for version, replacement in versioned_options
                if req.compare_version(version)
            )
        except StopIteration:
            raise PytorchResolutionError(
                'Could not find wheel for "{}", '
                "Available versions: {}".format(req, list(options))
            )

    def replace_conda(self, req):
        spec = "".join(req.specs[0]) if req.specs else ""
        if not self.cuda_version:
            return "pytorch-cpu{spec}\ntorchvision-cpu".format(spec=spec)
        return "pytorch{spec}\ntorchvision\ncuda{self.cuda_version}".format(
            self=self, spec=spec
        )

    def _table_lookup(self, req):
        """
        Look for pytorch wheel matching `req` in table
        :param req: python requirement
        """
        def check(base_, key_, exception_):
            result = base_.get(key_)
            if not result:
                if key_.startswith('cuda'):
                    print('Could not locate, {}'.format(exception_))
                    ver = sorted([float(a.replace('cuda', '').replace('none', '0')) for a in base_.keys()], reverse=True)[0]
                    key_ = 'cuda'+str(int(ver))
                    result = base_.get(key_)
                    print('Reverting to \"{}\"'.format(key_))
                    if not result:
                        raise exception_
                    return result
                raise exception_
            if isinstance(result, Exception):
                raise result
            return result

        if self.is_conda:
            return self.replace_conda(req)

        base = self.MAP
        for key, exception in zip((self.os, self.cuda, self.python), self.exceptions):
            base = check(base, key, exception)

        return self.match_version(req, base).replace(" ", "\n")

    def replace(self, req):
        try:
            new_req = self._replace(req)
            if new_req:
                self._original_req.append((req, new_req))
            return new_req
        except Exception as e:
            message = "Exception when trying to resolve python wheel"
            self.log.debug(message, exc_info=True)
            raise PytorchResolutionError("{}: {}".format(message, e))

    def _replace(self, req):
        self.validate_python_version()
        try:
            result, ok = self.get_url_for_platform(req)
            self.log.debug('Replacing requirement "%s" with %r', req, result)
            return result
        except:
            pass

        # try:
        #     result = self._table_lookup(req)
        # except Exception as e:
        #     exc = e
        # else:
        #     self.log.debug('Replacing requirement "%s" with %r', req, result)
        #     return result
        # self.log.debug(
        #     "Could not find Pytorch wheel in table, trying manually constructing URL"
        # )

        result = ok = None
        # try:
        #     result, ok = self.get_url_for_platform(req)
        # except Exception:
        #     pass

        if not ok:
            if result:
                self.log.debug("URL not found: {}".format(result))
            exc = PytorchResolutionError(
                "Could not find pytorch wheel URL for: {} with cuda {} support".format(req, self.cuda_version)
            )
            # cancel exception chaining
            six.raise_from(exc, None)

        self.log.debug('Replacing requirement "%s" with %r', req, result)
        return result

    def replace_back(self, list_of_requirements):  # type: (Dict) -> Dict
        """
        :param list_of_requirements: {'pip': ['a==1.0', ]}
        :return: {'pip': ['a==1.0', ]}
        """
        if not self._original_req:
            return list_of_requirements
        try:
            for k, lines in list_of_requirements.items():
                # k is either pip/conda
                if k not in ('pip', 'conda'):
                    continue
                for i, line in enumerate(lines):
                    if not line or line.lstrip().startswith('#'):
                        continue
                    parts = [p for p in re.split('\s|=|\.|<|>|~|!|@|#', line) if p]
                    if not parts:
                        continue
                    for req, new_req in self._original_req:
                        if req.req.name == parts[0]:
                            # support for pip >= 20.1
                            if '@' in line:
                                lines[i] = '{} # {}'.format(str(req), str(new_req))
                            else:
                                lines[i] = '{} # {}'.format(line, str(new_req))
                            break
        except:
            pass

        return list_of_requirements

    MAP = {
        "windows": {
            "cuda100": {
                "python3.7": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp37-cp37m-win_amd64.whl"
                },
                "python3.6": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-win_amd64.whl"
                },
                "python3.5": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp35-cp35m-win_amd64.whl"
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
            "cuda92": {
                "python3.7": {
                    "0.4.1",
                    "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp37-cp37m-win_amd64.whl",
                },
                "python3.6": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl"
                },
                "python3.5": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-win_amd64.whl"
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
            "cuda91": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-win_amd64.whl"
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-win_amd64.whl"
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
            "cuda90": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl",
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu90/torch-1.0.0-cp35-cp35m-win_amd64.whl",
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
            "cuda80": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu80/torch-1.0.0-cp36-cp36m-win_amd64.whl",
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu80/torch-1.0.0-cp35-cp35m-win_amd64.whl",
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
            "cudanone": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-win_amd64.whl",
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-win_amd64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cpu/torch-1.0.0-cp35-cp35m-win_amd64.whl",
                },
                "python2.7": PytorchResolutionError(
                    "PyTorch does not support Python 2.7 on Windows"
                ),
            },
        },
        "macos": {
            "cuda100": PytorchResolutionError(
                "MacOS Binaries dont support CUDA, install from source if CUDA is needed"
            ),
            "cuda92": PytorchResolutionError(
                "MacOS Binaries dont support CUDA, install from source if CUDA is needed"
            ),
            "cuda91": PytorchResolutionError(
                "MacOS Binaries dont support CUDA, install from source if CUDA is needed"
            ),
            "cuda90": PytorchResolutionError(
                "MacOS Binaries dont support CUDA, install from source if CUDA is needed"
            ),
            "cuda80": PytorchResolutionError(
                "MacOS Binaries dont support CUDA, install from source if CUDA is needed"
            ),
            "cudanone": {
                "python3.6": {"0.4.0": "torch"},
                "python3.5": {"0.4.0": "torch"},
                "python2.7": {"0.4.0": "torch"},
            },
        },
        "linux": {
            "cuda100": {
                "python3.7": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp37-cp37m-linux_x86_64.whl",
                    "1.0.1": "http://download.pytorch.org/whl/cu100/torch-1.0.1-cp37-cp37m-linux_x86_64.whl",
                    "1.1.0": "http://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl",
                    "1.2.0": "http://download.pytorch.org/whl/cu100/torch-1.2.0-cp37-cp37m-manylinux1_x86_64.whl",
                },
                "python3.6": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl",
                    "1.0.1": "http://download.pytorch.org/whl/cu100/torch-1.0.1-cp36-cp36m-linux_x86_64.whl",
                    "1.1.0": "http://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl",
                    "1.2.0": "http://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl",
                },
                "python3.5": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp35-cp35m-linux_x86_64.whl",
                    "1.0.1": "http://download.pytorch.org/whl/cu100/torch-1.0.1-cp35-cp35m-linux_x86_64.whl",
                    "1.1.0": "http://download.pytorch.org/whl/cu100/torch-1.1.0-cp35-cp35m-linux_x86_64.whl",
                    "1.2.0": "http://download.pytorch.org/whl/cu100/torch-1.2.0-cp35-cp35m-manylinux1_x86_64.whl",
                },
                "python2.7": {
                    "1.0.0": "http://download.pytorch.org/whl/cu100/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl",
                    "1.0.1": "http://download.pytorch.org/whl/cu100/torch-1.0.1-cp27-cp27mu-linux_x86_64.whl",
                    "1.1.0": "http://download.pytorch.org/whl/cu100/torch-1.1.0-cp27-cp27mu-linux_x86_64.whl",
                    "1.2.0": "http://download.pytorch.org/whl/cu100/torch-1.2.0-cp27-cp27mu-manylinux1_x86_64.whl",
                },
            },
            "cuda92": {
                "python3.7": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1.post2-cp37-cp37m-linux_x86_64.whl",
                    "1.2.0": "https://download.pytorch.org/whl/cu92/torch-1.2.0%2Bcu92-cp37-cp37m-manylinux1_x86_64.whl"
                },
                "python3.6": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl",
                    "1.2.0": "https://download.pytorch.org/whl/cu92/torch-1.2.0%2Bcu92-cp36-cp36m-manylinux1_x86_64.whl"
                },
                "python3.5": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-linux_x86_64.whl",
                    "1.2.0": "https://download.pytorch.org/whl/cu92/torch-1.2.0%2Bcu92-cp35-cp35m-manylinux1_x86_64.whl"
                },
                "python2.7": {
                    "0.4.1": "http://download.pytorch.org/whl/cu92/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl",
                    "1.2.0": "https://download.pytorch.org/whl/cu92/torch-1.2.0%2Bcu92-cp27-cp27mu-manylinux1_x86_64.whl"
                },
            },
            "cuda91": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl"
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cu91/torch-0.4.0-cp35-cp35m-linux_x86_64.whl"
                },
                "python2.7": {
                    "0.4.0": "http://download.pytorch.org/whl/cu91/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl"
                },
            },
            "cuda90": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cu90/torch-0.4.0-cp36-cp36m-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-linux_x86_64.whl",
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu90/torch-1.0.0-cp35-cp35m-linux_x86_64.whl",
                },
                "python2.7": {
                    "0.4.0": "http://download.pytorch.org/whl/cu90/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cu90/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl",
                },
            },
            "cuda80": {
                "python3.6": {
                    "0.4.1": "http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl",
                    "0.3.1": "torch==0.3.1",
                    "0.3.0.post4": "torch==0.3.0.post4",
                    "0.1.2.post1": "torch==0.1.2.post1",
                    "0.1.2": "torch==0.1.2",
                    "1.0.0": "http://download.pytorch.org/whl/cu80/torch-1.0.0-cp36-cp36m-linux_x86_64.whl",
                },
                "python3.5": {
                    "0.4.1": "http://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-linux_x86_64.whl",
                    "0.3.1": "torch==0.3.1",
                    "0.3.0.post4": "torch==0.3.0.post4",
                    "0.1.2.post1": "torch==0.1.2.post1",
                    "0.1.2": "torch==0.1.2",
                    "1.0.0": "http://download.pytorch.org/whl/cu80/torch-1.0.0-cp35-cp35m-linux_x86_64.whl",
                },
                "python2.7": {
                    "0.4.1": "http://download.pytorch.org/whl/cu80/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl",
                    "0.3.1": "torch==0.3.1",
                    "0.3.0.post4": "torch==0.3.0.post4",
                    "0.1.2.post1": "torch==0.1.2.post1",
                    "0.1.2": "torch==0.1.2",
                    "1.0.0": "http://download.pytorch.org/whl/cu80/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl",
                },
            },
            "cudanone": {
                "python3.6": {
                    "0.4.0": "http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl",
                },
                "python3.5": {
                    "0.4.0": "http://download.pytorch.org/whl/cpu/torch-0.4.0-cp35-cp35m-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cpu/torch-1.0.0-cp35-cp35m-linux_x86_64.whl",
                },
                "python2.7": {
                    "0.4.0": "http://download.pytorch.org/whl/cpu/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl",
                    "1.0.0": "http://download.pytorch.org/whl/cpu/torch-1.0.0-cp27-cp27mu-linux_x86_64.whl",
                },
            },
        },
    }
