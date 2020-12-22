import attr
import pytest
import requests
from furl import furl

import six
from clearml_agent.helper.package.pytorch import PytorchRequirement


@attr.s
class PytorchURLWheel(object):
    os = attr.ib()
    cuda = attr.ib()
    python = attr.ib()
    pytorch = attr.ib()
    url = attr.ib()


wheels = [
    PytorchURLWheel(os=os, cuda=cuda, python=python, pytorch=pytorch_version, url=url)
    for os, os_d in PytorchRequirement.MAP.items()
    for cuda, cuda_d in os_d.items()
    if isinstance(cuda_d, dict)
    for python, python_d in cuda_d.items()
    if isinstance(python_d, dict)
    for pytorch_version, url in python_d.items()
    if isinstance(url, six.string_types) and furl(url).scheme
]


@pytest.mark.parametrize('wheel', wheels, ids=[','.join(map(str, attr.astuple(wheel))) for wheel in wheels])
def test_map(wheel):
    assert requests.head(wheel.url).ok
