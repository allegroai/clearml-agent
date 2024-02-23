import re
import shlex
from typing import Tuple, List, TYPE_CHECKING
from urllib.parse import urlunparse, urlparse

from clearml_agent.definitions import (
    ENV_AGENT_GIT_PASS,
    ENV_AGENT_SECRET_KEY,
    ENV_AWS_SECRET_KEY,
    ENV_AZURE_ACCOUNT_KEY,
    ENV_AGENT_AUTH_TOKEN,
    ENV_DOCKER_IMAGE,
    ENV_DOCKER_ARGS_HIDE_ENV,
)

if TYPE_CHECKING:
    from clearml_agent.session import Session


def sanitize_urls(s: str) -> Tuple[str, bool]:
    """
    Replaces passwords in URLs with asterisks.
    Returns the sanitized string and a boolean indicating whether sanitation was performed.
    """
    regex = re.compile("^([^:]*:)[^@]+(.*)$")
    tokens = re.split(r"\s", s)
    changed = False
    for k in range(len(tokens)):
        if "@" in tokens[k]:
            res = urlparse(tokens[k])
            if regex.match(res.netloc):
                changed = True
                tokens[k] = urlunparse((
                    res.scheme,
                    regex.sub("\\1********\\2", res.netloc),
                    res.path,
                    res.params,
                    res.query,
                    res.fragment
                ))
    return " ".join(tokens) if changed else s, changed


class DockerArgsSanitizer:
    @classmethod
    def sanitize_docker_command(cls, session, docker_command):
        # type: (Session, List[str]) -> List[str]
        if not docker_command:
            return docker_command

        enabled = (
            session.config.get('agent.hide_docker_command_env_vars.enabled', False) or ENV_DOCKER_ARGS_HIDE_ENV.get()
        )
        if not enabled:
            return docker_command

        keys = set(session.config.get('agent.hide_docker_command_env_vars.extra_keys', []))
        if ENV_DOCKER_ARGS_HIDE_ENV.get():
            keys.update(shlex.split(ENV_DOCKER_ARGS_HIDE_ENV.get().strip()))
        keys.update(
            ENV_AGENT_GIT_PASS.vars,
            ENV_AGENT_SECRET_KEY.vars,
            ENV_AWS_SECRET_KEY.vars,
            ENV_AZURE_ACCOUNT_KEY.vars,
            ENV_AGENT_AUTH_TOKEN.vars,
        )

        parse_embedded_urls = bool(session.config.get(
            'agent.hide_docker_command_env_vars.parse_embedded_urls', True
        ))

        skip_next = False
        result = docker_command[:]
        for i, item in enumerate(docker_command):
            if skip_next:
                skip_next = False
                continue
            try:
                if item in ("-e", "--env"):
                    key, sep, val = result[i + 1].partition("=")
                    if not sep:
                        continue
                    if key in ENV_DOCKER_IMAGE.vars:
                        # special case - this contains a complete docker command
                        val = " ".join(cls.sanitize_docker_command(session, re.split(r"\s", val)))
                    elif key in keys:
                        val = "********"
                    elif parse_embedded_urls:
                        val = sanitize_urls(val)[0]
                    result[i + 1] = "{}={}".format(key, val)
                    skip_next = True
                elif parse_embedded_urls and not item.startswith("-"):
                    item, changed = sanitize_urls(item)
                    if changed:
                        result[i] = item
            except (KeyError, TypeError):
                pass

        return result

    @staticmethod
    def get_list_of_switches(docker_args: List[str]) -> List[str]:
        args = []
        for token in docker_args:
            if token.strip().startswith("-"):
                args += [token.strip().split("=")[0].lstrip("-")]

        return args

    @staticmethod
    def filter_switches(docker_args: List[str], exclude_switches: List[str]) -> List[str]:
        # shortcut if we are sure we have no matches
        if (not exclude_switches or
                not any("-{}".format(s) in " ".join(docker_args) for s in exclude_switches)):
            return docker_args

        args = []
        in_switch_args = True
        for token in docker_args:
            if token.strip().startswith("-"):
                if "=" in token:
                    switch = token.strip().split("=")[0]
                    in_switch_args = False
                else:
                    switch = token
                    in_switch_args = True

                if switch.lstrip("-") in exclude_switches:
                    # if in excluded, skip the switch and following arguments
                    in_switch_args = False
                else:
                    args += [token]

            elif in_switch_args:
                args += [token]
            else:
                # this is the switch arguments we need to skip
                pass

        return args

    @staticmethod
    def merge_docker_args(config, task_docker_arguments: List[str], extra_docker_arguments: List[str]) -> List[str]:
        base_cmd = []
        # currently only resolving --network, --ipc, --privileged
        override_switches = config.get(
            "agent.protected_docker_extra_args",
            ["privileged", "security-opt", "network", "ipc"]
        )

        if config.get("agent.docker_args_extra_precedes_task", True):
            switches = []
            if extra_docker_arguments:
                switches = DockerArgsSanitizer.get_list_of_switches(extra_docker_arguments)
                switches = list(set(switches) & set(override_switches))
                base_cmd += [str(a) for a in extra_docker_arguments if a]
            if task_docker_arguments:
                docker_arguments = DockerArgsSanitizer.filter_switches(task_docker_arguments, switches)
                base_cmd += [a for a in docker_arguments if a]
        else:
            switches = []
            if task_docker_arguments:
                switches = DockerArgsSanitizer.get_list_of_switches(task_docker_arguments)
                switches = list(set(switches) & set(override_switches))
                base_cmd += [a for a in task_docker_arguments if a]
            if extra_docker_arguments:
                extra_docker_arguments = DockerArgsSanitizer.filter_switches(extra_docker_arguments, switches)
                base_cmd += [a for a in extra_docker_arguments if a]
        return base_cmd
