"""
Script for generating command-line completion.
Called by trains_agent/utilities/complete.sh (or a copy of it) like so:

python -m trains_agent.complete "current command line"

And writes line-separated completion targets to stdout.
Results are line-separated in order to enable other whitespace in results.
"""

from __future__ import print_function

import argparse
import sys

from trains_agent.interface import get_parser


def is_argument_required(action):
    return isinstance(action, argparse._StoreAction)


def format_option(option, argument_required):
    """
    Return appropriate string for flags requiring arguments and flags that do not
    :param option: flag to format
    :param argument_required: whether argument is required
    """
    return option + '=' if argument_required else option + ' '


def get_options(parser):
    """
    Return all possible flags for parser
    :param parser: argparse.ArgumentParser instance
    :return: list of options
    """
    return [
        format_option(option, is_argument_required(action))
        for action in parser._actions
        for option in action.option_strings
    ]


def main():

    if len(sys.argv) != 2:
        return 1

    comp_words = iter(sys.argv[1].split()[1:])

    parser = get_parser()

    seen = []
    for word in comp_words:
        if word in parser.choices:
            parser = parser[word]
            continue
        actions = {name: action for action in parser._actions for name in action.option_strings}
        first, _, rest = word.partition('=')
        is_one_word_store_action = rest and first in actions
        if is_one_word_store_action:
            word = first
        seen.append(word)
        try:
            action = actions[word]
        except KeyError:
            break
        if isinstance(action, argparse._StoreAction) and not isinstance(action, argparse._StoreConstAction):
            if not is_one_word_store_action:
                try:
                    next(comp_words)
                except StopIteration:
                    break

    options = list(parser.choices)

    options = [format_option(option, argument_required=False) for option in options]
    options.extend(get_options(parser))
    options = [option for option in options if option.rstrip('= ') not in seen]

    print('\n'.join(options))
    return 0


if __name__ == "__main__":
    sys.exit(main())
