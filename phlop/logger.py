#
#


import logging
import os

LOG_LEVEL = os.environ.get("PHLOP_LOG_LEVEL", "INFO")
log_levels = {"INFO": 20, "WARNING": 30, "ERROR": 40, "DEBUG": 10}

FORMAT = "[log: %(filename)s:%(lineno)s - %(funcName)30s() ] %(message)s"
logging.basicConfig(format=FORMAT)
# logger.setLevel(log_levels[LOG_LEVEL])


def getLogger(name, level=LOG_LEVEL):
    logger = logging.getLogger(__name__)
    level = log_levels[level] if isinstance(level, str) else level
    logger.setLevel(level)
    return logger
