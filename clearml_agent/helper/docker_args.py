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
                        val = cls._sanitize_urls(val)[0]
                    result[i + 1] = "{}={}".format(key, val)
                    skip_next = True
                elif parse_embedded_urls and not item.startswith("-"):
                    item, changed = cls._sanitize_urls(item)
                    if changed:
                        result[i] = item
            except (KeyError, TypeError):
                pass

        return result

    @staticmethod
    def _sanitize_urls(s: str) -> Tuple[str, bool]:
        """ Replaces passwords in URLs with asterisks """
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
