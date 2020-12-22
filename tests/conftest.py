from argparse import Namespace
from contextlib import contextmanager

import pytest
import yaml
from pathlib2 import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='function')
def run_clearml_agent(script_runner):
    """ Execute clearml_agent agent app in subprocess and return stdout as a string.
    Args:
        script_runner (object): a pytest plugin for testing python scripts
        installed via console_scripts entry point of setup.py.
        It can run the scripts under test in a separate process or using the interpreter that's running
        the test suite. The former mode ensures that the script will run in an environment that
        is identical to normal execution whereas the latter one allows much quicker test runs during development
        while simulating the real runs as muh as possible.
        For more details: https://pypi.python.org/pypi/pytest-console-scripts
    Returns:
        string: The return value. stdout output
    """
    def _method(*args):
        clearml_agent_file = str(PROJECT_ROOT / "clearml_agent.sh")
        ret = script_runner.run(clearml_agent_file, *args)
        return ret
    return _method


@pytest.fixture(scope='function')
def clearml_agentyaml(tmpdir):
    @contextmanager
    def _method(template_file):
        file = tmpdir.join("clearml_agent.yaml")
        with (PROJECT_ROOT / "tests/templates" / template_file).open() as f:
            code = yaml.load(f, Loader=yaml.SafeLoader)
            yield Namespace(code=code, file=file.strpath)
        file.write(yaml.dump(code))
    return _method


# class Test(object):
#     def yaml_file(self, tmpdir, template_file):
#         file = tmpdir.join("clearml_agent.yaml")
#         with open(template_file) as f:
#             test_object = yaml.load(f)
#         self.let(test_object)
#         file.write(yaml.dump(test_object))
#         return file.strpath
