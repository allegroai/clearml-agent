import functools

from subprocess import DEVNULL

from clearml_agent.helper.process import get_bash_output as _get_bash_output


def get_path(d, *path, default=None):
    try:
        return functools.reduce(
            lambda a, b: a[b], path, d
        )
    except (IndexError, KeyError):
        return default


def get_bash_output(cmd, stderr=DEVNULL, raise_error=False):
    return _get_bash_output(cmd, stderr=stderr, raise_error=raise_error)
