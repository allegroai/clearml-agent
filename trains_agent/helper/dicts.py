from typing import Callable, Dict, Any


def filter_keys(filter_, dct):  # type: (Callable[[Any], bool], Dict) -> Dict
    return {key: value for key, value in dct.items() if filter_(key)}


def merge_dicts(dict1, dict2):
    """ Recursively merges dict2 into dict1 """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for k in dict2:
        if k in dict1:
            dict1[k] = merge_dicts(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]
    return dict1
