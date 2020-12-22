import itertools
from functools import partial
from importlib import import_module
import argparse

from trains_agent.definitions import PROGRAM_NAME
from .base import Parser, base_arguments, add_service, OnlyPluralChoicesHelpFormatter

SERVICES = [
    'worker',
]


def get_parser():
    top_parser = Parser(
        prog=PROGRAM_NAME,
        add_help=False,
        formatter_class=partial(
            OnlyPluralChoicesHelpFormatter,
            max_help_position=120,
            width=120,
        ),
    )
    base_arguments(top_parser)
    from .worker import COMMANDS
    subparsers = top_parser.add_subparsers(dest='command')
    for c in COMMANDS:
        parser = subparsers.add_parser(name=c, help=COMMANDS[c]["help"])
        groups = itertools.groupby(
            sorted(
                COMMANDS[c].get("args", {}).items(), key=lambda x: x[1].get("group", "")
            ),
            key=lambda x: x[1].pop("group", ""),
        )
        for group_name, group in groups:
            p = parser if not group_name else parser.add_argument_group(group_name)
            for key, value in group:
                aliases = value.pop("aliases", [])
                p.add_argument(key, *aliases, **value)

    return top_parser
