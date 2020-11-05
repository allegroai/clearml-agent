import os


def pytest_addoption(parser):
    """
    Add a pytest_addoption to the pytest.

    Args:
        parser: (todo): write your description
    """
    parser.addoption('--cpu', action='store_true')


def pytest_configure(config):
    """
    Configure the pytesture.

    Args:
        config: (dict): write your description
    """
    if not config.option.cpu:
        return
    os.environ['PATH'] = ':'.join(p for p in os.environ['PATH'].split(':') if 'cuda' not in p)
    os.environ['CUDA_VERSION'] = ''
    os.environ['CUDNN_VERSION'] = ''
