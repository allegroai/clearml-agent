from __future__ import unicode_literals, division

import logging
import os
import shlex
from collections import deque
from itertools import starmap
from threading import Thread, Event
from time import time
from typing import Text, Sequence

import attr
import psutil
from pathlib2 import Path
from clearml_agent.session import Session
from clearml_agent.definitions import ENV_WORKER_TAGS

try:
    from .gpu import gpustat
except ImportError:
    gpustat = None

log = logging.getLogger(__name__)


class BytesSizes(object):
    @staticmethod
    def kilobytes(x):
        # type: (float) -> float
        return x / 1024

    @staticmethod
    def megabytes(x):
        # type: (float) -> float
        return x / (1024*1024)

    @staticmethod
    def gigabytes(x):
        # type: (float) -> float
        return x / (1024*1024*1024)


class ResourceMonitor(object):
    @attr.s
    class StatusReport(object):
        task = attr.ib(default=None, type=str)
        queue = attr.ib(default=None, type=str)
        queues = attr.ib(default=None, type=Sequence[str])

        def to_dict(self):
            return {
                key: value
                for key, value in attr.asdict(self).items()
                if value is not None
            }

    def __init__(
        self,
        session,  # type: Session
        worker_id,  # type: ResourceMonitor.StatusReport,
        sample_frequency_per_sec=2.0,
        report_frequency_sec=30.0,
        first_report_sec=None,
        worker_tags=None,
    ):
        self.session = session
        self.queue = deque(maxlen=1)
        self.queue.appendleft(self.StatusReport())
        self._worker_id = worker_id
        self._sample_frequency = sample_frequency_per_sec
        self._report_frequency = report_frequency_sec
        self._first_report_sec = first_report_sec or report_frequency_sec
        self._num_readouts = 0
        self._readouts = {}
        self._previous_readouts = {}
        self._previous_readouts_ts = time()
        self._thread = None
        self._exit_event = Event()
        self._gpustat_fail = 0
        self._gpustat = gpustat
        self._active_gpus = None
        if not worker_tags and ENV_WORKER_TAGS.get():
            worker_tags = shlex.split(ENV_WORKER_TAGS.get())
        self._worker_tags = worker_tags
        if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'none':
            # NVIDIA_VISIBLE_DEVICES set to none, marks cpu_only flag
            # active_gpus == False means no GPU reporting
            self._active_gpus = False
        elif not self._gpustat:
            log.warning('ClearML-Agent Resource Monitor: GPU monitoring is not available')
        else:
            # None means no filtering, report all gpus
            self._active_gpus = None
            try:
                active_gpus = os.environ.get('NVIDIA_VISIBLE_DEVICES', '') or \
                              os.environ.get('CUDA_VISIBLE_DEVICES', '')
                if active_gpus:
                    self._active_gpus = [int(g.strip()) for g in active_gpus.split(',')]
            except Exception:
                pass

    def set_report(self, report):
        # type: (ResourceMonitor.StatusReport) -> ()
        if report is not None:
            self.queue.appendleft(report)

    def get_report(self):
        # type: () -> ResourceMonitor.StatusReport
        return self.queue[0]

    def start(self):
        self._exit_event.clear()
        self._thread = Thread(target=self._daemon)
        self._thread.daemon = True
        self._thread.start()
        return self

    def stop(self):
        self._exit_event.set()
        self.send_report()

    def send_report(self, stats=None):
        report = dict(
            machine_stats=stats,
            timestamp=(int(time()) * 1000),
            worker=self._worker_id,
            tags=self._worker_tags,
            **self.get_report().to_dict()
        )
        log.debug("sending report: %s", report)

        try:
            self.session.get(service="workers", action="status_report", **report)
        except Exception:
            log.warning("Failed sending report: %s", report)
            return False
        return True

    def _daemon(self):
        seconds_since_started = 0
        reported = 0
        while True:
            last_report = time()
            current_report_frequency = (
                self._report_frequency if reported != 0 else self._first_report_sec
            )
            while (time() - last_report) < current_report_frequency:
                # wait for self._sample_frequency seconds, if event set quit
                if self._exit_event.wait(1 / self._sample_frequency):
                    return
                # noinspection PyBroadException
                try:
                    self._update_readouts()
                except Exception as ex:
                    log.warning("failed getting machine stats: %s", report_error(ex))
                    self._failure()

            seconds_since_started += int(round(time() - last_report))
            # check if we do not report any metric (so it means the last iteration will not be changed)

            # if we do not have last_iteration, we just use seconds as iteration

            # start reporting only when we figured out, if this is seconds based, or iterations based
            average_readouts = self._get_average_readouts()
            stats = {
                # 3 points after the dot
                key: round(value, 3) if isinstance(value, float) else [round(v, 3) for v in value]
                for key, value in average_readouts.items()
            }

            # send actual report
            if self.send_report(stats):
                # clear readouts if this is update was sent
                self._clear_readouts()

            # count reported iterations
            reported += 1

    def _update_readouts(self):
        readouts = self._machine_stats()
        elapsed = time() - self._previous_readouts_ts
        self._previous_readouts_ts = time()

        def fix(k, v):
            if k.endswith("_mbs"):
                v = (v - self._previous_readouts.get(k, v)) / elapsed

            if v is None:
                v = 0
            return k, self._readouts.get(k, 0) + v

        self._readouts.update(starmap(fix, readouts.items()))
        self._num_readouts += 1
        self._previous_readouts = readouts

    def _get_num_readouts(self):
        return self._num_readouts

    def _get_average_readouts(self):
        def create_general_key(old_key):
            """
            Create key for backend payload
            :param old_key: old stats key
            :type old_key: str
            :return: new key for sending stats
            :rtype: str
            """
            key_parts = old_key.rpartition("_")
            return "{}_*".format(key_parts[0] if old_key.startswith("gpu") else old_key)

        ret = {}
        # make sure the gpu/cpu stats are always ordered in the accumulated values list (general_key)
        ordered_keys = sorted(self._readouts.keys())
        for k in ordered_keys:
            v = self._readouts[k]
            stat_key = self.BACKEND_STAT_MAP.get(k)
            if stat_key:
                ret[stat_key] = v / self._num_readouts
            else:
                general_key = create_general_key(k)
                general_key = self.BACKEND_STAT_MAP.get(general_key)
                if general_key:
                    ret.setdefault(general_key, []).append(v / self._num_readouts)
                else:
                    pass  # log.debug("Cannot find key {}".format(k))
        return ret

    def _clear_readouts(self):
        self._readouts = {}
        self._num_readouts = 0

    def _machine_stats(self):
        """
        :return: machine stats dictionary, all values expressed in megabytes
        """
        cpu_usage = psutil.cpu_percent(percpu=True)
        stats = {"cpu_usage": sum(cpu_usage) / len(cpu_usage)}

        virtual_memory = psutil.virtual_memory()
        stats["memory_used"] = BytesSizes.megabytes(virtual_memory.used)
        stats["memory_free"] = BytesSizes.megabytes(virtual_memory.available)
        disk_use_percentage = psutil.disk_usage(Text(Path.home())).percent
        stats["disk_free_percent"] = 100 - disk_use_percentage
        sensor_stat = (
            psutil.sensors_temperatures()
            if hasattr(psutil, "sensors_temperatures")
            else {}
        )
        if "coretemp" in sensor_stat and len(sensor_stat["coretemp"]):
            stats["cpu_temperature"] = max([t.current for t in sensor_stat["coretemp"]])

        # update cached measurements
        net_stats = psutil.net_io_counters()
        stats["network_tx_mbs"] = BytesSizes.megabytes(net_stats.bytes_sent)
        stats["network_rx_mbs"] = BytesSizes.megabytes(net_stats.bytes_recv)
        io_stats = psutil.disk_io_counters()
        stats["io_read_mbs"] = BytesSizes.megabytes(io_stats.read_bytes)
        stats["io_write_mbs"] = BytesSizes.megabytes(io_stats.write_bytes)

        # check if we need to monitor gpus and if we can access the gpu statistics
        if self._active_gpus is not False and self._gpustat:
            try:
                gpu_stat = self._gpustat.new_query()
                for i, g in enumerate(gpu_stat.gpus):
                    # only monitor the active gpu's, if none were selected, monitor everything
                    if self._active_gpus and i not in self._active_gpus:
                        continue
                    stats["gpu_temperature_{:d}".format(i)] = g["temperature.gpu"]
                    stats["gpu_utilization_{:d}".format(i)] = g["utilization.gpu"]
                    stats["gpu_mem_usage_{:d}".format(i)] = (
                        100.0 * g["memory.used"] / g["memory.total"]
                    )
                    # already in MBs
                    stats["gpu_mem_free_{:d}".format(i)] = (
                        g["memory.total"] - g["memory.used"]
                    )
                    stats["gpu_mem_used_%d" % i] = g["memory.used"]
            except Exception as ex:
                # something happened and we can't use gpu stats,
                log.warning("failed getting machine stats: %s", report_error(ex))
                self._failure()

        return stats

    def _failure(self):
        self._gpustat_fail += 1
        if self._gpustat_fail >= 3:
            log.error(
                "GPU monitoring failed getting GPU reading, switching off GPU monitoring"
            )
            self._gpustat = None

    BACKEND_STAT_MAP = {"cpu_usage_*": "cpu_usage",
                        "cpu_temperature_*": "cpu_temperature",
                        "disk_free_percent": "disk_free_home",
                        "io_read_mbs": "disk_read",
                        "io_write_mbs": "disk_write",
                        "network_tx_mbs": "network_tx",
                        "network_rx_mbs": "network_rx",
                        "memory_free": "memory_free",
                        "memory_used": "memory_used",
                        "gpu_temperature_*": "gpu_temperature",
                        "gpu_mem_used_*": "gpu_memory_used",
                        "gpu_mem_free_*": "gpu_memory_free",
                        "gpu_utilization_*": "gpu_usage"}


def report_error(ex):
    return "{}: {}".format(type(ex).__name__, ex)
