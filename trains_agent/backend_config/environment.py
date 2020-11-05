from os import getenv, environ

from .converters import text_to_bool
from .entry import Entry, NotSet


class EnvEntry(Entry):
    @classmethod
    def default_conversions(cls):
        """
        Return the default.

        Args:
            cls: (todo): write your description
        """
        conversions = super(EnvEntry, cls).default_conversions().copy()
        conversions[bool] = text_to_bool
        return conversions

    def _get(self, key):
        """
        Return the value of key.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        value = getenv(key, "").strip()
        return value or NotSet

    def _set(self, key, value):
        """
        Sets a value of a value pair.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (todo): write your description
        """
        environ[key] = value

    def __str__(self):
        """
        Return the string representation of the field.

        Args:
            self: (todo): write your description
        """
        return "env:{}".format(super(EnvEntry, self).__str__())

    def error(self, message):
        """
        Print an error message

        Args:
            self: (todo): write your description
            message: (str): write your description
        """
        print("Environment configuration: {}".format(message))


def backward_compatibility_support():
    """
    Determine the support of support.

    Args:
    """
    from ..definitions import ENVIRONMENT_CONFIG, ENVIRONMENT_SDK_PARAMS, ENVIRONMENT_BACKWARD_COMPATIBLE
    if not ENVIRONMENT_BACKWARD_COMPATIBLE.get():
        return

    # Add ALG_ prefix on every TRAINS_ os environment we support
    for k, v in ENVIRONMENT_CONFIG.items():
        try:
            trains_vars = [var for var in v.vars if var.startswith('TRAINS_')]
            if not trains_vars:
                continue
            alg_var = trains_vars[0].replace('TRAINS_', 'ALG_', 1)
            if alg_var not in v.vars:
                v.vars = tuple(list(v.vars) + [alg_var])
        except:
            continue
    for k, v in ENVIRONMENT_SDK_PARAMS.items():
        try:
            trains_vars = [var for var in v if var.startswith('TRAINS_')]
            if not trains_vars:
                continue
            alg_var = trains_vars[0].replace('TRAINS_', 'ALG_', 1)
            if alg_var not in v:
                ENVIRONMENT_SDK_PARAMS[k] = tuple(list(v) + [alg_var])
        except:
            continue
