from __future__ import unicode_literals

import re
import subprocess
from itertools import chain
from os import path
from os.path import sep
from sys import platform as sys_platform

import pytest
import requirements

from clearml_agent.commands.worker import Worker
from clearml_agent.helper.package.pytorch import PytorchRequirement
from clearml_agent.helper.package.requirements import RequirementsManager, \
    RequirementSubstitution, MarkerRequirement
from clearml_agent.helper.process import get_bash_output
from clearml_agent.session import Session

_cuda_based_packages_hack = ('seematics.caffe', 'lightnet')


def old_get_suffix(session):
    cuda_version = session.config['agent.cuda_version']
    cudnn_version = session.config['agent.cudnn_version']
    if cuda_version and cudnn_version:
        nvcc_ver = cuda_version.strip()
        cudnn_ver = cudnn_version.strip()
    else:
        if sys_platform == 'win32':
            nvcc_ver = subprocess.check_output('nvcc --version'.split()).decode('utf-8'). \
                replace('\r', '').split('\n')
        else:
            nvcc_ver = subprocess.check_output(
                'nvcc --version'.split()).decode('utf-8').split('\n')
        nvcc_ver = [l for l in nvcc_ver if 'release ' in l]
        nvcc_ver = nvcc_ver[0][nvcc_ver[0].find(
            'release ') + len('release '):][:3]
        nvcc_ver = str(int(float(nvcc_ver) * 10))
        if sys_platform == 'win32':
            cuda_lib = subprocess.check_output('where nvcc'.split()).decode('utf-8'). \
                replace('\r', '').split('\n')[0]
            cudnn_h = sep.join(cuda_lib.split(
                sep)[:-2] + ['include', 'cudnn.h'])
        else:
            cuda_lib = subprocess.check_output(
                'which nvcc'.split()).decode('utf-8').split('\n')[0]
            cudnn_h = path.join(
                sep, *(cuda_lib.split(sep)[:-2] + ['include', 'cudnn.h']))
        cudnn_mj, cudnn_mi = None, None
        for l in open(cudnn_h, 'r'):
            if 'CUDNN_MAJOR' in l:
                cudnn_mj = l.split()[-1]
            if 'CUDNN_MINOR' in l:
                cudnn_mi = l.split()[-1]
            if cudnn_mj and cudnn_mi:
                break
        cudnn_ver = cudnn_mj + ('0' if not cudnn_mi else cudnn_mi)
    # build cuda + cudnn version suffix
    # make sure these are integers, someone else will catch the exception
    pkg_suffix_ver = '.post' + \
                     str(int(nvcc_ver)) + '.dev' + str(int(cudnn_ver))
    return pkg_suffix_ver, nvcc_ver, cudnn_ver


def old_replace(session, line):
    try:
        cuda_ver_suffix, cuda_ver, cuda_cudnn_ver = old_get_suffix(session)
    except Exception:
        return line
    if line.lstrip().startswith('#'):
        return line
    for package_name in _cuda_based_packages_hack:
        if package_name not in line:
            continue
        try:
            line_lstrip = line.lstrip()
            if line_lstrip.startswith('http://') or line_lstrip.startswith('https://'):
                pos = line.find(package_name) + len(package_name)
                # patch line with specific version
                line = line[:pos] + \
                    line[pos:].replace('-cp', cuda_ver_suffix + '-cp', 1)
            else:
                # this is a pypi package
                tokens = line.replace('=', ' ').replace('<', ' ').replace('>', ' ').replace(';', ' '). \
                    replace('!', ' ').split()
                if package_name != tokens[0]:
                    # how did we get here, probably a mistake
                    found_cuda_based_package = False
                    continue

                version_number = None
                if len(tokens) > 1:
                    # get the package version info
                    test_version_number = tokens[1]
                    # check if we have a valid version, i.e. does not contain post/dev
                    version_number = '.'.join([v for v in test_version_number.split('.')
                                               if v and '0' <= v[0] <= '9'])
                    if version_number != test_version_number:
                        raise ValueError()

                # we have no version, but we have to have one
                if not version_number:
                    # get the latest version from the extra index list
                    pip_search_cmd = ['pip', 'search']
                    if Worker._pip_extra_index_url:
                        pip_search_cmd.extend(
                            chain.from_iterable(('-i', x) for x in Worker._pip_extra_index_url))
                    pip_search_cmd += [package_name]
                    pip_search_output = get_bash_output(
                        ' '.join(pip_search_cmd), strip=True)
                    version_number = pip_search_output.split(package_name)[1]
                    version_number = version_number.replace(
                        '(', ' ').replace(')', ' ').split()[0]
                    version_number = '.'.join([v for v in version_number.split('.')
                                               if v and '0' <= v[0] <= '9'])
                    if not version_number:
                        # somewhere along the way we failed
                        raise ValueError()

                package_name_version = package_name + '==' + version_number + cuda_ver_suffix
                if version_number in line:
                    # make sure we have the specific version not >=
                    tokens = line.split(';')
                    line = ';'.join([package_name_version] + tokens[1:])
                else:
                    # add version to the package_name
                    line = line.replace(package_name, package_name_version, 1)

            #  print('pip install %s using CUDA v%s CuDNN v%s' % (package_name, cuda_ver, cuda_cudnn_ver))
        except ValueError:
            pass
            #  print('Warning! could not find installed CUDA/CuDNN version for %s, '
            #  'using original requirements line: %s' % (package_name, line))
        # add the current line into the cuda requirements list
    return line


win_condition = 'sys_platform != "win_32"'
versions = ('1', '1.4', '1.4.9', '1.5.3.dev0',
            '1.5.3.dev3', '43.1.2.dev0.post1')


def normalize(result):
    return result and re.sub(' ?; ?', ';', result)


def parse_one(requirement):
    return MarkerRequirement(next(requirements.parse(requirement)))


def compare(manager, current_session, pytest_config, requirement, expected):
    try:
        res1 = normalize(manager._replace_one(parse_one(requirement)))
    except ValueError:
        res1 = None
    res2 = old_replace(current_session, requirement)
    if res2 == requirement:
        res2 = None
    res2 = normalize(res2)
    expected = normalize(expected)
    if pytest_config.option.cpu:
        assert res1 is None
    else:
        assert res1 == expected
    if requirement not in FAILURES:
        assert res1 == res2
    return res1


ARG_VERSIONED = pytest.mark.parametrize('arg', (
    'nothing{op}{version}{extra}',
    'something_else{op}{version}{extra}',
    'something-else{op}{version}{extra}',
    'something.else{op}{version}{extra}',
    'seematics.caffe{op}{version}{extra}',
    'lightnet{op}{version}{extra}',
))
OP = pytest.mark.parametrize('op', ('==', '<=', '>='))
VERSION = pytest.mark.parametrize('version', versions)
EXTRA = pytest.mark.parametrize('extra', ('', ' ; ' + win_condition))
ARG_PLAIN = pytest.mark.parametrize('arg', (
    'nothing',
    'something_else',
    'something-else',
    'something.else',
    'seematics.caffe',
    'lightnet',
    'https://s3.amazonaws.com/seematics-pip/public/windows64bit/static/seematics.caffe-1.0.3-cp35-cp35m-win_amd64.whl',
    'https://s3.amazonaws.com/seematics-pip/public/windows64bit/static/seematics.caffe-1.0.3-cp27-cp27m-win_amd64.whl',
    'https://s3.amazonaws.com/seematics-pip/public/static/seematics.caffe-1.0.3-cp35-cp35m-linux_x86_64.whl',
    'https://s3.amazonaws.com/seematics-pip/public/static/seematics.caffe-1.0.3-cp27-cp27mu-linux_x86_64.whl',
    'https://s3.amazonaws.com/seematics-pip/public/seematics.config-1.0.2-py2.py3-none-any.whl',
    'https://s3.amazonaws.com/seematics-pip/public/seematics.api-1.0.2-py2.py3-none-any.whl',
    'https://s3.amazonaws.com/seematics-pip/public/seematics.sdk-1.1.0-py2.py3-none-any.whl',
))


@ARG_VERSIONED
@OP
@VERSION
@EXTRA
def test_with_version(manager, current_session, pytestconfig, arg, op, version, extra):
    requirement = arg.format(**locals())
    # expected = EXPECTED.get((arg, op, version))
    expected = get_expected(current_session, (arg, op, version))
    expected = expected and expected + extra
    compare(manager, current_session, pytestconfig, requirement, expected)


@ARG_PLAIN
@EXTRA
def test_plain(arg, current_session, pytestconfig, extra, manager):
    # expected = EXPECTED.get(arg)
    expected = get_expected(current_session, arg)
    expected = expected and expected + extra
    compare(manager, current_session, pytestconfig, arg + extra, expected)


def get_expected(session, key):
    result = EXPECTED.get(key)
    cuda_version, cudnn_version = session.config['agent.cuda_version'], session.config['agent.cudnn_version']
    suffix = '.post{cuda_version}.dev{cudnn_version}'.format(**locals()) \
        if (cuda_version and cudnn_version) \
        else ''
    return result and result.format(suffix=suffix)


@ARG_VERSIONED
@OP
@VERSION
@EXTRA
def test_str_versioned(arg, op, version, extra):
    requirement = arg.format(**locals())
    assert normalize(str(parse_one(requirement))) == normalize(requirement)


@ARG_PLAIN
@EXTRA
def test_str_plain(arg, extra, manager):
    requirement = arg.format(**locals())
    assert normalize(str(parse_one(requirement))) == normalize(requirement)


@pytest.fixture(scope='session')
def current_session(pytestconfig):
    session = Session()
    if not pytestconfig.option.cpu:
        return session
    session.config['agent.cuda_version'] = None
    session.config['agent.cudnn_version'] = None
    return session


@pytest.fixture(scope='session')
def manager(current_session):
    manager = RequirementsManager(current_session.config)
    for requirement in (PytorchRequirement, ):
        manager.register(requirement)
    return manager


SPECS = (
    # plain
    '',

    # greater than
    '>=0', '>0.1', '>=0.1', '>0.1.2', '>=0.1.2', '>0.1.2.post30', '>=0.1.3.post30', '>0.post30.dev40',
    '>=0.post30.dev40', '>0.post30-dev40', '>=0.post30-dev40', '>0.0', '>=0.0', '>0.3', '>=0.3', '>=0.4',

    # smaller than
    '<4', '<4.0', '<4.1.0', '<4.1.0.post80.dev60', '<4.1.0-post80.dev60', '<3', '<2.0', '<1.0.3', '<=4', '<=4.0',
    '<=4.1.0', '<=4.1.0.post80.dev60', '<=4.1.0-post80.dev60', '<=3', '<=2.0', '<=1.0.3',

    # equals
    '==0.4.0',

    # equals and
    '==0.4.0,>=0', '==0.4.0,>=0.1', '==0.4.0,>0.1', '==0.4.0,>=0.1.2', '==0.4.0,>0.1.2', '==0.4.0,<4', '==0.4.0,<=4',
    '==0.4.0,<4.0', '==0.4.0,<=4.0', '==0.4.0,<4.0.2',

    # smaller and greater
    '>=0,<4', '>=0,<4.0', '>0.1,<1', '>0.1,<1.1.2', '>0.1,<1.1.2', '>=0,<=4', '>=0,<4.0',
    '>=0.1,<1', '>0.1,<=1.1.2', '>=0.1,<1.1.2',
)


@pytest.mark.parametrize('package_manager', ('pip', 'conda'))
@pytest.mark.parametrize('os', ('linux', 'windows', 'macos'))
@pytest.mark.parametrize('cuda', ('80', '90', '91'))
@pytest.mark.parametrize('python', ('2.7', '3.5', '3.6'))
@pytest.mark.parametrize('spec', SPECS)
@pytest.mark.parametrize('condition', ('', ' ; ' + win_condition))
def test_pytorch_success(manager, package_manager, os, cuda, python, spec, condition):
    pytorch_handler = manager.handlers[-1]
    pytorch_handler.package_manager = package_manager
    pytorch_handler.os = os
    pytorch_handler.cuda = cuda_ver = 'cuda{}'.format(cuda)
    pytorch_handler.python = python_ver = 'python{}'.format(python)
    req = 'torch{}{}'.format(spec, condition)
    expected = pytorch_handler.MAP[package_manager][os][cuda_ver][python_ver]
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)):
            manager._replace_one(parse_one(req))
    else:
        expected = expected['0.4.0']
        result = manager._replace_one(parse_one(req))
        assert result == expected


get_pip_version = RequirementSubstitution.get_pip_version

EXPECTED = {
    'https://s3.amazonaws.com/seematics-pip/public/windows64bit/static/seematics.caffe-1.0.3-cp35-cp35m-win_amd64.whl':
        'https://s3.amazonaws.com/seematics-pip/public/windows64bit/static/seematics.caffe-1.0.3{suffix}-cp35'
        '-cp35m-win_amd64.whl',
    'https://s3.amazonaws.com/seematics-pip/public/windows64bit/static/seematics.caffe-1.0.3-cp27-cp27m-win_amd64.whl':
        'https://s3.amazonaws.com/seematics-pip/public/windows64bit/static/seematics.caffe-1.0.3{suffix}-cp27'
        '-cp27m-win_amd64.whl',
    'https://s3.amazonaws.com/seematics-pip/public/static/seematics.caffe-1.0.3-cp35-cp35m-linux_x86_64.whl':
        'https://s3.amazonaws.com/seematics-pip/public/static/seematics.caffe-1.0.3{suffix}-cp35-cp35m'
        '-linux_x86_64.whl',
    'https://s3.amazonaws.com/seematics-pip/public/static/seematics.caffe-1.0.3-cp27-cp27mu-linux_x86_64.whl':
        'https://s3.amazonaws.com/seematics-pip/public/static/seematics.caffe-1.0.3{suffix}-cp27-cp27mu'
        '-linux_x86_64.whl',
    'seematics.caffe': 'seematics.caffe=={}{{suffix}}'.format(get_pip_version('seematics.caffe')),
    'lightnet': 'lightnet=={}{{suffix}}'.format(get_pip_version('lightnet')),
    ('seematics.caffe{op}{version}{extra}', '<=', '1'): 'seematics.caffe==1{suffix}',
    ('seematics.caffe{op}{version}{extra}', '<=', '1.4'): 'seematics.caffe==1.4{suffix}',
    ('seematics.caffe{op}{version}{extra}', '<=', '1.4.9'): 'seematics.caffe==1.4.9{suffix}',
    ('seematics.caffe{op}{version}{extra}', '==', '1'): 'seematics.caffe==1{suffix}',
    ('seematics.caffe{op}{version}{extra}', '==', '1.4'): 'seematics.caffe==1.4{suffix}',
    ('seematics.caffe{op}{version}{extra}', '==', '1.4.9'): 'seematics.caffe==1.4.9{suffix}',
    ('seematics.caffe{op}{version}{extra}', '>=', '1'): 'seematics.caffe==1{suffix}',
    ('seematics.caffe{op}{version}{extra}', '>=', '1.4'): 'seematics.caffe==1.4{suffix}',
    ('seematics.caffe{op}{version}{extra}', '>=', '1.4.9'): 'seematics.caffe==1.4.9{suffix}',
    ('lightnet{op}{version}{extra}', '<=', '1'): 'lightnet==1{suffix}',
    ('lightnet{op}{version}{extra}', '<=', '1.4'): 'lightnet==1.4{suffix}',
    ('lightnet{op}{version}{extra}', '<=', '1.4.9'): 'lightnet==1.4.9{suffix}',
    ('lightnet{op}{version}{extra}', '==', '1'): 'lightnet==1{suffix}',
    ('lightnet{op}{version}{extra}', '==', '1.4'): 'lightnet==1.4{suffix}',
    ('lightnet{op}{version}{extra}', '==', '1.4.9'): 'lightnet==1.4.9{suffix}',
    ('lightnet{op}{version}{extra}', '>=', '1'): 'lightnet==1{suffix}',
    ('lightnet{op}{version}{extra}', '>=', '1.4'): 'lightnet==1.4{suffix}',
    ('lightnet{op}{version}{extra}', '>=', '1.4.9'): 'lightnet==1.4.9{suffix}',
}
FAILURES = {
    'seematics.caffe ; {}'.format(win_condition),
    'lightnet ; {}'.format(win_condition)
}
