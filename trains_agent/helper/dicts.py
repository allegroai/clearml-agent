from typing import Callable, Dict, Any


def filter_keys(filter_, dct):  # type: (Callable[[Any], bool], Dict) -> Dict
    return {key: value for key, value in dct.items() if filter_(key)}
