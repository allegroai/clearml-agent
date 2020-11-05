from __future__ import unicode_literals, print_function

import csv
import sys
from collections import Iterable
from typing import List, Dict, Text, Any

from attr import attrs, attrib

import six
from six import binary_type, text_type
from trains_agent.helper.base import nonstrict_in_place_sort


def print_text(text, newline=True):
    """
    Prints a string.

    Args:
        text: (str): write your description
        newline: (str): write your description
    """
    if newline:
        text += '\n'
    data = text.encode(sys.stdout.encoding or 'utf8', errors='replace')
    try:
        sys.stdout.buffer.write(data)
    except AttributeError:
        sys.stdout.write(data)


def decode_binary_lines(binary_lines, encoding='utf-8', replace_cr=False, overwrite_cr=False):
    """
    Decode a list of binary lines.

    Args:
        binary_lines: (list): write your description
        encoding: (str): write your description
        replace_cr: (str): write your description
        overwrite_cr: (bool): write your description
    """
    # decode per line, if we failed decoding skip the line
    lines = []
    for b in binary_lines:
        # noinspection PyBroadException
        try:
            line = b.decode(encoding=encoding, errors='replace')
            if replace_cr:
                line = line.replace('\r', '\n')
            elif overwrite_cr:
                cr_lines = line.split('\r')
                line = cr_lines[-1] if cr_lines[-1] or len(cr_lines) < 2 else cr_lines[-2]
        except Exception:
            line = ''
        lines.append(line + '\n' if not line or line[-1] != '\n' else line)
    return lines


def ensure_text(s, encoding='utf-8', errors='strict'):
    """Coerce *s* to six.text_type.
    For Python 2:
      - `unicode` -> `unicode`
      - `str` -> `unicode`
    For Python 3:
      - `str` -> `str`
      - `bytes` -> decoded to `str`
    """
    if isinstance(s, binary_type):
        return s.decode(encoding, errors)
    elif isinstance(s, text_type):
        return s
    else:
        raise TypeError("not expecting type '%s'" % type(s))


def ensure_binary(s, encoding='utf-8', errors='strict'):
    """Coerce **s** to six.binary_type.
    For Python 2:
      - `unicode` -> encoded to `str`
      - `str` -> `str`
    For Python 3:
      - `str` -> encoded to `bytes`
      - `bytes` -> `bytes`
    """
    if isinstance(s, text_type):
        return s.encode(encoding, errors)
    elif isinstance(s, binary_type):
        return s
    else:
        raise TypeError("not expecting type '%s'" % type(s))


class ListFormatter(object):

    @attrs(init=False)
    class Table(object):
        entries = attrib(type=List[Dict])
        columns = attrib(type=List[Text])

        def __init__(self, entries, columns):
            """
            Initialize the table to the table.

            Args:
                self: (todo): write your description
                entries: (todo): write your description
                columns: (str): write your description
            """
            self.entries = entries
            if isinstance(columns, str):
                columns = columns.split('#')
            self.columns = columns

        def as_rows(self):  # type: () -> Iterable[Iterable[Any]]
            """
            : return : class :.

            Args:
                self: (todo): write your description
            """
            return (
                map(entry.get, self.columns)
                for entry in self.entries
            )

    def __init__(self, service_name):
        """
        Initialize a service.

        Args:
            self: (todo): write your description
            service_name: (str): write your description
        """
        self.service_name = service_name

    def get_total(self, entries):
        """
        Returns the total number of entries in the given entries.

        Args:
            self: (todo): write your description
            entries: (str): write your description
        """
        return '\nTotal {} {}'.format(self.service_name, len(entries))

    @classmethod
    def write_csv(cls, entries, columns, dest, headers=True):
        """
        Write a csv to a csv file.

        Args:
            cls: (todo): write your description
            entries: (list): write your description
            columns: (list): write your description
            dest: (str): write your description
            headers: (list): write your description
        """
        table = cls.Table(entries, columns)
        with open(dest, 'w') as output:
            writer = csv.DictWriter(output, fieldnames=table.columns, extrasaction='ignore')
            if headers:
                writer.writeheader()
            writer.writerows(table.entries)

    @staticmethod
    def sort_in_place(entries, key, reverse=None):
        """
        Sort place in place.

        Args:
            entries: (list): write your description
            key: (str): write your description
            reverse: (bool): write your description
        """
        if isinstance(key, six.string_types):
            nonstrict_in_place_sort(entries, reverse, *key.split('#'))
        elif callable(key):
            entries.sort(key=key, reverse=reverse)
        else:
            raise ValueError('"sort" argument must be either a string or a callable object')
