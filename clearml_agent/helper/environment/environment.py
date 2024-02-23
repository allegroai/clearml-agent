from os import getenv, environ

from .converters import text_to_bool
from .entry import Entry, NotSet


class EnvEntry(Entry):
    def default_conversions(self):
        conversions = super(EnvEntry, self).default_conversions().copy()
        conversions[bool] = lambda x: text_to_bool(x.strip())
        return conversions

    def pop(self):
        for k in self.keys:
            environ.pop(k, None)

    def _get(self, key):
        value = getenv(key, "")
        return value or NotSet

    def _set(self, key, value):
        environ[key] = value

    def __str__(self):
        return "env:{}".format(super(EnvEntry, self).__str__())

    def error(self, message):
        print("Environment configuration: {}".format(message))
