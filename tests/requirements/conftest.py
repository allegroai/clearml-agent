import os


def pytest_addoption(parser):
    parser.addoption('--cpu', action='store_true')


def pytest_configure(config):
    if not config.option.cpu:
        return
    os.environ['PATH'] = ':'.join(p for p in os.environ['PATH'].split(':') if 'cuda' not in p)
    os.environ['CUDA_VERSION'] = ''
    os.environ['CUDNN_VERSION'] = ''
