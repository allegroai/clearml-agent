from os import environ

from clearml_agent.helper.environment import EnvEntry


def backward_compatibility_support():
    from ..definitions import ENVIRONMENT_CONFIG, ENVIRONMENT_SDK_PARAMS, ENVIRONMENT_BACKWARD_COMPATIBLE
    if ENVIRONMENT_BACKWARD_COMPATIBLE.get():
        # Add TRAINS_ prefix on every CLEARML_ os environment we support
        for k, v in ENVIRONMENT_CONFIG.items():
            # noinspection PyBroadException
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
            # noinspection PyBroadException
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


__all__ = [
    "EnvEntry",
    "backward_compatibility_support"
]