import json
import re
import shlex
from copy import copy

from clearml_agent.backend_api.session import Request
from clearml_agent.helper.docker_args import DockerArgsSanitizer
from clearml_agent.helper.package.requirements import (
    RequirementsManager, MarkerRequirement,
    compare_version_rules, )


def resolve_default_container(session, task_id, container_config, ignore_match_rules=False):
    container_lookup = session.config.get('agent.default_docker.match_rules', None)
    if not session.check_min_api_version("2.13") or not container_lookup:
        return container_config

    # check backend support before sending any more requests (because they will fail and crash the Task)
    try:
        session.verify_feature_set('advanced')
    except ValueError:
        # ignoring matching rules only supported in higher tiers
        return container_config

    if ignore_match_rules:
        print("INFO: default docker command line override, ignoring default docker container match rules")
        # ignoring matching rules only supported in higher tiers
        return container_config

    result = session.send_request(
        service='tasks',
        action='get_all',
        version='2.14',
        json={'id': [task_id],
              'only_fields': ['script.requirements', 'script.binary',
                              'script.repository', 'script.branch',
                              'project', 'container'],
              'search_hidden': True},
        method=Request.def_method,
        async_enable=False,
    )
    try:
        task_info = result.json()['data']['tasks'][0] if result.ok else {}
    except (ValueError, TypeError):
        return container_config

    from clearml_agent.external.requirements_parser.requirement import Requirement

    # store tasks repository
    repository = task_info.get('script', {}).get('repository') or ''
    branch = task_info.get('script', {}).get('branch') or ''
    binary = task_info.get('script', {}).get('binary') or ''
    requested_container = task_info.get('container', {})

    # get project full path
    project_full_name = ''
    if task_info.get('project', None):
        result = session.send_request(
            service='projects',
            action='get_all',
            version='2.13',
            json={
                'id': [task_info.get('project')],
                'only_fields': ['name'],
            },
            method=Request.def_method,
            async_enable=False,
        )
        try:
            if result.ok:
                project_full_name = result.json()['data']['projects'][0]['name'] or ''
        except (ValueError, TypeError):
            pass

    task_packages_lookup = {}
    for entry in container_lookup:
        match = entry.get('match', None)
        if not match:
            continue
        if match.get('project', None):
            # noinspection PyBroadException
            try:
                if not re.search(match.get('project', None), project_full_name):
                    continue
            except Exception:
                print('Failed parsing regular expression \"{}\" in rule: {}'.format(
                    match.get('project', None), entry))
                continue

        if match.get('script.repository', None):
            # noinspection PyBroadException
            try:
                if not re.search(match.get('script.repository', None), repository):
                    continue
            except Exception:
                print('Failed parsing regular expression \"{}\" in rule: {}'.format(
                    match.get('script.repository', None), entry))
                continue

        if match.get('script.branch', None):
            # noinspection PyBroadException
            try:
                if not re.search(match.get('script.branch', None), branch):
                    continue
            except Exception:
                print('Failed parsing regular expression \"{}\" in rule: {}'.format(
                    match.get('script.branch', None), entry))
                continue

        if match.get('script.binary', None):
            # noinspection PyBroadException
            try:
                if not re.search(match.get('script.binary', None), binary):
                    continue
            except Exception:
                print('Failed parsing regular expression \"{}\" in rule: {}'.format(
                    match.get('script.binary', None), entry))
                continue

        # if match.get('image', None):
        #     # noinspection PyBroadException
        #     try:
        #         if not re.search(match.get('image', None), requested_container.get('image', '')):
        #             continue
        #     except Exception:
        #         print('Failed parsing regular expression \"{}\" in rule: {}'.format(
        #             match.get('image', None), entry))
        #         continue

        matched = True
        for req_section in ['script.requirements.pip', 'script.requirements.conda']:
            if not match.get(req_section, None):
                continue

            match_pip_reqs = [MarkerRequirement(Requirement.parse('{} {}'.format(k, v)))
                              for k, v in match.get(req_section, None).items()]

            if not task_packages_lookup.get(req_section):
                req_section_parts = req_section.split('.')
                task_packages_lookup[req_section] = \
                    RequirementsManager.parse_requirements_section_to_marker_requirements(
                        requirements=task_info.get(req_section_parts[0], {}).get(
                            req_section_parts[1], {}).get(req_section_parts[2], None)
                    )

            matched_all_reqs = True
            for mr in match_pip_reqs:
                matched_req = False
                for pr in task_packages_lookup[req_section]:
                    if mr.req.name != pr.req.name:
                        continue
                    if compare_version_rules(mr.specs, pr.specs):
                        matched_req = True
                        break
                if not matched_req:
                    matched_all_reqs = False
                    break

            # if ew have a match, check second section
            if matched_all_reqs:
                continue
            # no match stop
            matched = False
            break

        if matched:
            if not container_config.get('image'):
                container_config['image'] = entry.get('image', None)
            if not container_config.get('arguments'):
                container_config['arguments'] = entry.get('arguments', None) or ''
                if isinstance(container_config.get('arguments'), str):
                    container_config['arguments'] = shlex.split(str(container_config.get('arguments') or '').strip())
            print('INFO: Matching default container with rule:\n{}'.format(json.dumps(entry)))
            return container_config

    return container_config

