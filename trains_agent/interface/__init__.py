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
        parser = subparsers.add_parser(name=c, help=COMMANDS[c]['help'])
        for a in COMMANDS[c].get('args', {}).keys():
            parser.add_argument(a, **COMMANDS[c]['args'][a])

    return top_parser
