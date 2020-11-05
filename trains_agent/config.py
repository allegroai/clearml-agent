from pyhocon import ConfigTree

import six
from trains_agent.helper.base import Singleton


@six.add_metaclass(Singleton)
class Config(object):

    def __init__(self, tree=None):
        """
        Initialize the tree.

        Args:
            self: (todo): write your description
            tree: (dict): write your description
        """
        self.__dict__['_tree'] = tree or ConfigTree()

    def __getitem__(self, item):
        """
        Return the value of item.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self._tree[item]

    def __setitem__(self, key, value):
        """
        Sets the value of the given value.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (str): write your description
        """
        return self._tree.__setitem__(key, value)

    def new(self, name):
        """
        Create a new configuration tree.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        return self._tree.setdefault(name, ConfigTree())

    __getattr__ = __getitem__
    __setattr__ = __setitem__


def get_config(name=None):
    """
    Get a configuration object from the given name.

    Args:
        name: (str): write your description
    """
    config = Config()
    if name:
        return getattr(config, name)
    return config


def make_config(name):
    """
    Create a new configuration.

    Args:
        name: (str): write your description
    """
    return get_config().new(name)
