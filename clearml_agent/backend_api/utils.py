import logging
import ssl
import sys

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from urllib3 import PoolManager

from .session.defs import ENV_HOST_VERIFY_CERT


__disable_certificate_verification_warning = 0


class _RetryFilter(logging.Filter):
    last_instance = None

    def __init__(self, total, warning_after=5):
        super(_RetryFilter, self).__init__()
        self.total = total
        self.display_warning_after = warning_after
        _RetryFilter.last_instance = self

    def filter(self, record):
        if record.args and len(record.args) > 0 and isinstance(record.args[0], Retry):
            left = (record.args[0].total, record.args[0].connect, record.args[0].read,
                    record.args[0].redirect, record.args[0].status)
            left = [l for l in left if isinstance(l, int)]
            if left:
                retry_left = max(left) - min(left)
                return retry_left >= self.display_warning_after

        return True


def urllib_log_warning_setup(total_retries=10, display_warning_after=5):
    for l in ('urllib3.connectionpool', 'requests.packages.urllib3.connectionpool'):
        urllib3_log = logging.getLogger(l)
        if urllib3_log:
            urllib3_log.removeFilter(_RetryFilter.last_instance)
            urllib3_log.addFilter(_RetryFilter(total_retries, display_warning_after))


class TLSv1HTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       ssl_version=ssl.PROTOCOL_TLSv1_2)


def get_http_session_with_retry(
        total=0,
        connect=None,
        read=None,
        redirect=None,
        status=None,
        status_forcelist=None,
        backoff_factor=0,
        backoff_max=None,
        pool_connections=None,
        pool_maxsize=None,
        config=None):
    if not config:
        config = {}
    global __disable_certificate_verification_warning
    if not all(isinstance(x, (int, type(None))) for x in (total, connect, read, redirect, status)):
        raise ValueError('Bad configuration. All retry count values must be null or int')

    if status_forcelist and not all(isinstance(x, int) for x in status_forcelist):
        raise ValueError('Bad configuration. Retry status_forcelist must be null or list of ints')

    pool_maxsize = (
        pool_maxsize
        if pool_maxsize is not None
        else config.get('api.http.pool_maxsize', 512)
    )

    pool_connections = (
        pool_connections
        if pool_connections is not None
        else config.get('api.http.pool_connections', 512)
    )

    session = requests.Session()

    if backoff_max is not None:
        Retry.DEFAULT_BACKOFF_MAX = backoff_max

    retry = Retry(
        total=total, connect=connect, read=read, redirect=redirect, status=status,
        status_forcelist=status_forcelist, backoff_factor=backoff_factor)

    adapter = TLSv1HTTPAdapter(max_retries=retry, pool_connections=pool_connections, pool_maxsize=pool_maxsize)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    # update verify host certificate
    session.verify = ENV_HOST_VERIFY_CERT.get(default=config.get('api.verify_certificate', True))
    if not session.verify and __disable_certificate_verification_warning < 2:
        # show warning
        __disable_certificate_verification_warning += 1
        logging.getLogger('ClearML').warning(
            msg='InsecureRequestWarning: Certificate verification is disabled! Adding '
                'certificate verification is strongly advised. See: '
                'https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings')
        # make sure we only do not see the warning
        import urllib3
        # noinspection PyBroadException
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
    return session


def get_response_cls(request_cls):
    """ Extract a request's response class using the mapping found in the module defining the request's service """
    for req_cls in request_cls.mro():
        module = sys.modules[req_cls.__module__]
        if hasattr(module, 'action_mapping'):
            return module.action_mapping[(request_cls._action, request_cls._version)][1]
        elif hasattr(module, 'response_mapping'):
            return module.response_mapping[req_cls]
    raise TypeError('no response class!')
