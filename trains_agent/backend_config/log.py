import logging.config

from pathlib2 import Path


def logger(path=None):
    """
    Returns the logger.

    Args:
        path: (str): write your description
    """
    name = "trains"
    if path:
        p = Path(path)
        module = (p.parent if p.stem.startswith('_') else p).stem
        name = "trains.%s" % module
    return logging.getLogger(name)


def initialize(logging_config=None, extra=None):
    """
    Initialize the logger.

    Args:
        logging_config: (todo): write your description
        extra: (todo): write your description
    """
    if extra is not None:
        from logging import Logger

        class _Logger(Logger):
            __extra = extra.copy()

            def _log(self, level, msg, args, exc_info=None, extra=None, **kwargs):
                """
                Logs a message to the logger.

                Args:
                    self: (todo): write your description
                    level: (int): write your description
                    msg: (str): write your description
                    exc_info: (todo): write your description
                    extra: (dict): write your description
                """
                extra = extra or {}
                extra.update(self.__extra)
                super(_Logger, self)._log(level, msg, args, exc_info=exc_info, extra=extra, **kwargs)

        Logger.manager.loggerClass = _Logger

    if logging_config is not None:
        logging.config.dictConfig(dict(logging_config))
