import argparse
from textwrap import dedent

from trains_agent.helper.base import warning, is_windows_platform
from trains_agent.interface.base import foreign_object_id


class DeprecatedFlag(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        warning('argument "{}" is deprecated'.format(option_string))


WORKER_ARGS = {
    '-O': {
        'help': 'Compile optimized pyc code (see python documentation). Repeat for more optimization.',
        'action': 'count',
        'default': 0,
        'dest': 'optimization',
    },
    '--git-user': {
        'help': 'git username for repository access',
    },
    '--git-pass': {
        'help': 'git password for repository access',
    },
    '--log-level': {
        'help': 'SDK log level',
        'choices': ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'CRITICAL'],
        'type': lambda x: x.upper(),
        'default': 'INFO',
    },
    '--gpus': {
        'help': 'Specify active GPUs for the daemon to use (docker / virtual environment), '
                'Equivalent to setting NVIDIA_VISIBLE_DEVICES '
                'Examples: --gpus 0 or --gpu 0,1,2 or --gpus all',
        'group': 'Docker support',
    },
    '--cpu-only': {
        'help': 'Disable GPU access for the daemon, only use CPU in either docker or virtual environment',
        'action': 'store_true',
        'group': 'Docker support',
    },
}

DAEMON_ARGS = dict({
    '--foreground': {
        'help': 'Pipe full log to stdout/stderr, should not be used if running in background',
        'action': 'store_true',
    },
    '--docker': {
        'help': 'Run execution task inside a docker (v19.03 and above). Optional args <image> <arguments> or '
                'specify default docker image in agent.default_docker.image / agent.default_docker.arguments'
                'use --gpus/--cpu-only (or set NVIDIA_VISIBLE_DEVICES) to limit gpu visibility for docker',
        'nargs': '*',
        'default': False,
        'group': 'Docker support',
    },
    '--force-current-version': {
        'help': 'Force trains-agent to use the current trains-agent version when running in the docker',
        'action': 'store_true',
        'group': 'Docker support',
    },
    '--queue': {
        'help': 'Queue ID(s)/Name(s) to pull tasks from (\'default\' queue)',
        'nargs': '+',
        'default': tuple(),
        'dest': 'queues',
        'type': foreign_object_id('queues'),
    },
    '--standalone-mode': {
        'help': 'Do not use any network connects, assume everything is pre-installed',
        'action': 'store_true',
    },
    '--services-mode': {
        'help': 'Launch multiple long-term docker services. Implies docker & cpu-only flags.',
        'action': 'store_true',
    },
    '--create-queue': {
        'help': 'Create requested queue if it does not exist already.',
        'action': 'store_true',
    },
    '--detached': {
        'help': 'Detached mode, run agent in the background',
        'action': 'store_true',
        'aliases': ['-d'],
    },

}, **WORKER_ARGS)

COMMANDS = {
    'execute': {
        'help': 'Build & Execute a selected experiment',
        'args': dict({
            '--id': {
                'help': 'Task ID to run',
                'required': True,
                'dest': 'task_id',
                'type': foreign_object_id('tasks'),
            },
            '--log-file': {
                'help': 'Output task execution (stdout/stderr) into text file',
            },
            '--disable-monitoring': {
                'help': 'Disable logging & monitoring (stdout is still visible)',
                'action': 'store_true',
            },
            '--full-monitoring': {
                'help': 'Full environment setup log & task logging & monitoring (stdout is still visible)',
                'action': 'store_true',
            },
            '--require-queue': {
                'help': 'If the specified task is not queued (in any Queue), the execution will fail. '
                        '(Used for 3rd party scheduler integration, e.g. K8s, SLURM, etc.)',
                'action': 'store_true',
            },
            '--standalone-mode': {
                'help': 'Do not use any network connects, assume everything is pre-installed',
                'action': 'store_true',
            },
            '--docker': {
                'help': 'Run execution task inside a docker (v19.03 and above). Optional args <image> <arguments> or '
                        'specify default docker image in agent.default_docker.image / agent.default_docker.arguments'
                        'use --gpus/--cpu-only (or set NVIDIA_VISIBLE_DEVICES) to limit gpu visibility for docker',
                'nargs': '*',
                'default': False,
            },
            '--clone': {
                'help': 'Clone the experiment before execution, and execute the cloned experiment',
                'action': 'store_true',
            },
        }, **WORKER_ARGS),
    },
    'build': {
        'help': 'Build selected experiment environment '
                '(including pip packages, cloned code and git diff)\n'
                'Used mostly for debugging purposes',
        'args': dict({
            '--id': {
                'help': 'Task ID to build',
                'required': True,
                'dest': 'task_id',
                'type': foreign_object_id('tasks'),
            },
            '--target': {
                'help': 'Where to build the task\'s virtual environment and source code. '
                        'When used with --docker, target docker image name to create',
            },
            '--install-globally': {
                'help': 'Install required python packages before creating the virtual environment used to execute an '
                        'experiment, and use the \'agent.package_manager.system_site_packages\' virtual env option. '
                        'Note: when --docker is used, install-globally is always true',
                'action': 'store_true',
            },
            '--docker': {
                'help': 'Build the experiment inside a docker (v19.03 and above). Optional args <image> <arguments> or '
                'specify default docker image in agent.default_docker.image / agent.default_docker.arguments'
                'use --gpus/--cpu-only (or set NVIDIA_VISIBLE_DEVICES) to limit gpu visibility for docker',
                'nargs': '*',
                'default': False,
            },
            '--python-version': {
                'help': 'Virtual environment python version to use',
            },
            '--entry-point': {
                'help': 'Run the task in the new docker. There are two options:\nEither add "reuse_task" to run the '
                'given task in the docker, or "clone_task" to first clone the given task and then run it in the docker',
                'default': False,
                'choices': ['reuse_task', 'clone_task'],
            }
        }, **WORKER_ARGS),
    },
    'list': {
        'help': 'List all worker machines and status',
    },
    'daemon': {
        'help': 'Start Trains-Agent daemon worker',
        'args': DAEMON_ARGS,
    },
    'config': {
        'help': 'Check daemon configuration and print it',
    },
    'init': {
        'help': 'Trains-Agent configuration wizard',
    }
}
