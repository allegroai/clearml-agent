import os
from time import sleep

import requests
import json
from threading import Thread
from .package.requirements import SimpleVersion
from ..version import __version__

__check_update_thread = None


def start_check_update_daemon():
    global __check_update_thread
    if __check_update_thread:
        return
    __check_update_thread = Thread(target=_check_update_daemon)
    __check_update_thread.daemon = True
    __check_update_thread.start()


def _check_new_version_available():
    cur_version = __version__
    update_server_releases = requests.get('https://updates.trains.allegro.ai/updates',
                                          data=json.dumps({"versions": {"trains-agent": str(cur_version)}}),
                                          timeout=3.0)
    if update_server_releases.ok:
        update_server_releases = update_server_releases.json()
    else:
        return None
    trains_answer = update_server_releases.get("trains-agent", {})
    latest_version = trains_answer.get("version")
    cur_version = cur_version
    latest_version = latest_version or ''
    if SimpleVersion.compare_versions(cur_version, '>=', latest_version):
        return None
    patch_upgrade = True  # latest_version.major == cur_version.major and latest_version.minor == cur_version.minor
    return str(latest_version), patch_upgrade, trains_answer.get("description").split("\r\n")


def _check_update_daemon():
    counter = 0
    while True:
        # noinspection PyBroadException
        try:
            latest_version = _check_new_version_available()
            # only print when we begin
            if latest_version:
                if latest_version[1]:
                    sep = os.linesep
                    print('TRAINS-AGENT new package available: UPGRADE to v{} is recommended!\nRelease Notes:\n{}'.format(
                        latest_version[0], sep.join(latest_version[2])))
                else:
                    print('TRAINS-SERVER new version available: upgrade to v{} is recommended!'.format(
                        latest_version[0]))
        except Exception:
            pass
        # sleep until the next day
        sleep(60 * 60 * 24)
        counter += 1
