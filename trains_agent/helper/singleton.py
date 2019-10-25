import os
import psutil
from time import sleep
from glob import glob
from tempfile import gettempdir, NamedTemporaryFile

from trains_agent.helper.base import warning


class Singleton(object):
    prefix = 'trainsagent'
    sep = '_'
    ext = '.tmp'
    worker_id = None
    worker_name_sep = ':'
    instance_slot = None
    _pid_file = None
    _lock_file_name = sep+prefix+sep+'global.lock'
    _lock_timeout = 10

    @classmethod
    def register_instance(cls, unique_worker_id=None, worker_name=None):
        """
        # Exit the process if another instance of us is using the same worker_id

        :param unique_worker_id: if already exists, return negative
        :param worker_name: slot number will be added to worker name, based on the available instance slot
        :return: (str worker_id, int slot_number) Return None value on instance already running
        """
        # try to lock file
        lock_file = os.path.join(gettempdir(), cls._lock_file_name)
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
                ret = cls._register_instance(unique_worker_id=unique_worker_id, worker_name=worker_name)
            except:
                ret = None, None

        try:
            os.remove(lock_file)
        except Exception:
            pass

        return ret

    @classmethod
    def _register_instance(cls, unique_worker_id=None, worker_name=None):
        if cls.worker_id:
            return cls.worker_id, cls.instance_slot
        # make sure we have a unique name
        instance_num = 0
        temp_folder = gettempdir()
        files = glob(os.path.join(temp_folder, cls.prefix + cls.sep + '*' + cls.ext))
        slots = {}
        for file in files:
            parts = file.split(cls.sep)
            try:
                pid = int(parts[1])
            except Exception:
                # something is wrong, use non existing pid and delete the file
                pid = -1
            # count active instances and delete dead files
            if not psutil.pid_exists(pid):
                # delete the file
                try:
                    os.remove(os.path.join(file))
                except Exception:
                    pass
                continue

            instance_num += 1
            try:
                with open(file, 'r') as f:
                    uid, slot = str(f.read()).split('\n')
                    slot = int(slot)
            except Exception:
                continue

            if uid == unique_worker_id:
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
        cls._pid_file = NamedTemporaryFile(dir=gettempdir(), prefix=cls.prefix + cls.sep + str(os.getpid()) + cls.sep,
                                           suffix=cls.ext)
        cls._pid_file.write(('{}\n{}'.format(unique_worker_id, cls.instance_slot)).encode())
        cls._pid_file.flush()
        cls.worker_id = unique_worker_id

        return cls.worker_id, cls.instance_slot
