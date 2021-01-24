from os import getenv, environ

from .converters import text_to_bool
from .entry import Entry, NotSet


class EnvEntry(Entry):
    @classmethod
    def default_conversions(cls):
        conversions = super(EnvEntry, cls).default_conversions().copy()
        conversions[bool] = text_to_bool
        return conversions

    def pop(self):
        for k in self.keys:
            environ.pop(k, None)

    def _get(self, key):
        value = getenv(key, "").strip()
        return value or NotSet

    def _set(self, key, value):
        environ[key] = value

    def __str__(self):
        return "env:{}".format(super(EnvEntry, self).__str__())

    def error(self, message):
        print("Environment configuration: {}".format(message))


def backward_compatibility_support():
    from ..definitions import ENVIRONMENT_CONFIG, ENVIRONMENT_SDK_PARAMS, ENVIRONMENT_BACKWARD_COMPATIBLE
    if ENVIRONMENT_BACKWARD_COMPATIBLE.get():
        # Add TRAINS_ prefix on every CLEARML_ os environment we support
        for k, v in ENVIRONMENT_CONFIG.items():
            try:
                trains_vars = [var for var in v.vars if var.startswith('CLEARML_')]
                if not trains_vars:
                    continue
                alg_var = trains_vars[0].replace('CLEARML_', 'TRAINS_', 1)
                if alg_var not in v.vars:
                    v.vars = tuple(list(v.vars) + [alg_var])
            except:
                continue
        for k, v in ENVIRONMENT_SDK_PARAMS.items():
            try:
                trains_vars = [var for var in v if var.startswith('CLEARML_')]
                if not trains_vars:
                    continue
                alg_var = trains_vars[0].replace('CLEARML_', 'TRAINS_', 1)
                if alg_var not in v:
                    ENVIRONMENT_SDK_PARAMS[k] = tuple(list(v) + [alg_var])
            except:
                continue

    # set OS environ:
    keys = list(environ.keys())
    for k in keys:
        if not k.startswith('CLEARML_'):
            continue
        backwards_k = k.replace('CLEARML_', 'TRAINS_', 1)
        if backwards_k not in keys:
            environ[backwards_k] = environ[k]
