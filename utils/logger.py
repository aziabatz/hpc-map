import logging

from lightning.pytorch.utilities.rank_zero import rank_zero_only


def get_pylogger(name=__name__) -> logging.Logger:

    logger = logging.getLogger(name)

    logging_levels = [
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
        "info"
    ]

    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
