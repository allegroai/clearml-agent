import os
import psutil
from time import sleep
from glob import glob
from tempfile import gettempdir, NamedTemporaryFile

from typing import List, Tuple, Optional

from clearml_agent.definitions import ENV_DOCKER_HOST_MOUNT
from clearml_agent.helper.base import warning, is_windows_platform, safe_remove_file


class Singleton(object):
    prefix = '.clearmlagent'
    sep = '_'
    ext = '.tmp'
    worker_id = None
    worker_name_sep = ':'
    instance_slot = None
    _pid_file = None
    _lock_file_name = sep+prefix+sep+'global.lock'
    _lock_timeout = 10
    _pid = None

    @classmethod
    def close_pid_file(cls):
        if cls._pid_file:
            cls._pid_file.close()
            safe_remove_file(cls._pid_file.name)
        cls._pid_file = None

    @classmethod
    def update_pid_file(cls):
        new_pid = str(os.getpid())
        if not cls._pid_file or cls._pid == new_pid:
            return
        old_name = cls._pid_file.name
        parts = cls._pid_file.name.split(os.path.sep)
        parts[-1] = parts[-1].replace(cls.sep + cls._pid + cls.sep, cls.sep + new_pid + cls.sep)
        new_pid_file = os.path.sep.join(parts)
        cls._pid = new_pid
        cls._pid_file.name = new_pid_file
        # we need to rename to match new pid
        try:
            os.rename(old_name, new_pid_file)
        except:
            pass

    @classmethod
    def get_lock_filename(cls):
        return os.path.join(cls._get_temp_folder(), cls._lock_file_name)

    @classmethod
    def register_instance(cls, unique_worker_id=None, worker_name=None, api_client=None, allow_double=False):
        """
        # Exit the process if another instance of us is using the same worker_id

        :param unique_worker_id: if already exists, return negative
        :param worker_name: slot number will be added to worker name, based on the available instance slot
        :return: (str worker_id, int slot_number) Return None value on instance already running
        """
        # try to lock file
        lock_file = cls.get_lock_filename()
        timeout = 0
        while os.path.exists(lock_file):
            if timeout > cls._lock_timeout:
                warning('lock file timed out {}sec - clearing lock'.format(cls._lock_timeout))
                try:
                    os.remove(lock_file)
                except Exception:
                    pass
                break

            sleep(1)
            timeout += 1

        with open(lock_file, 'wb') as f:
            f.write(bytes(os.getpid()))
            f.flush()
            try:
                ret = cls._register_instance(
                    unique_worker_id=unique_worker_id, worker_name=worker_name,
                    api_client=api_client, allow_double=allow_double)
            except:
                ret = None, None

        try:
            os.remove(lock_file)
        except Exception:
            pass

        return ret

    @classmethod
    def get_running_pids(cls):
        # type: () -> List[Tuple[int, Optional[str], Optional[int], str]]
        temp_folder = cls._get_temp_folder()
        files = glob(os.path.join(temp_folder, cls.prefix + cls.sep + '*' + cls.ext))
        pids = []
        for file in files:
            parts = os.path.basename(file).split(cls.sep)
            # noinspection PyBroadException
            try:
                pid = int(parts[1])
                if not psutil.pid_exists(pid):
                    pid = -1
            except Exception:
                # something is wrong, use non existing pid and delete the file
                pid = -1

            uid, slot = None, None
            # noinspection PyBroadException
            try:
                with open(file, 'r') as f:
                    uid, slot = str(f.read()).split('\n')
                    slot = int(slot)
            except Exception:
                pass
            pids.append((pid, uid, slot, file))

        return pids

    @classmethod
    def _register_instance(cls, unique_worker_id=None, worker_name=None, api_client=None, allow_double=False):
        if cls.worker_id and cls.instance_slot is not None:
            return cls.worker_id, cls.instance_slot
        # make sure we have a unique name
        instance_num = 0
        slots = {}
        for pid, uid, slot, file in cls.get_running_pids():
            worker = None
            if api_client and ENV_DOCKER_HOST_MOUNT.get() and uid:
                try:
                    worker = [w for w in api_client.workers.get_all() if w.id == uid]
                except Exception:
                    worker = None

            # count active instances and delete dead files
            if not worker and pid < 0:
                # delete the file
                try:
                    os.remove(os.path.join(file))
                except Exception:
                    pass
                continue

            instance_num += 1
            if slot is None:
                continue

            if uid == unique_worker_id:
                if allow_double:
                    warning('Instance with the same WORKER_ID [{}] was found on this machine. '
                            'We are ignoring it, make sure this not a mistake.'.format(unique_worker_id))
                else:
                    return None, None

            slots[slot] = uid

        # get a new slot
        if not slots:
            cls.instance_slot = 0
        else:
            # guarantee we have the minimal slot possible
            for i in range(max(slots.keys())+2):
                if i not in slots:
                    cls.instance_slot = i
                    break

        # build worker id based on slot
        if not unique_worker_id:
            unique_worker_id = worker_name + cls.worker_name_sep + str(cls.instance_slot)

        # create lock
        cls._pid = str(os.getpid())
        cls._pid_file = NamedTemporaryFile(
            dir=cls._get_temp_folder(), prefix=cls.prefix + cls.sep + cls._pid + cls.sep, suffix=cls.ext,
            delete=False if is_windows_platform() else True
        )
        cls._pid_file.write(('{}\n{}'.format(unique_worker_id, cls.instance_slot)).encode())
        cls._pid_file.flush()
        cls.worker_id = unique_worker_id

        return cls.worker_id, cls.instance_slot

    @classmethod
    def _get_temp_folder(cls):
        if ENV_DOCKER_HOST_MOUNT.get():
            return ENV_DOCKER_HOST_MOUNT.get().split(':')[-1]
        return gettempdir()

    @classmethod
    def get_slot(cls):
        return cls.instance_slot or 0

    @classmethod
    def get_pid_file(cls):
        if not cls._pid_file:
            return None
        return cls._pid_file.name
