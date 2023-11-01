import logging
import logging.handlers
import sys

from hardpicks import ANALYSIS_RESULTS_DIR

ANALYSIS_LOGGING_FORMAT = "%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() - %(message)s"


def setup_analysis_logger():
    """This method sets up logging for analysis scripts."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    analysis_log_file = ANALYSIS_RESULTS_DIR.joinpath("analysis.log")
    formatter = logging.Formatter(ANALYSIS_LOGGING_FORMAT)

    file_handler = logging.handlers.WatchedFileHandler(analysis_log_file)
    stream_handler = logging.StreamHandler(stream=sys.stdout)

    list_handlers = [file_handler, stream_handler]

    for handler in list_handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)


def configure_logger_for_console_only(logger):
    """Configure the logger to only output to console. Good for little ad hoc scripts."""
    formatter = logging.Formatter(ANALYSIS_LOGGING_FORMAT)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def configure_logging_for_console_only():
    """Configure the logger to only output to console. Good for little ad hoc scripts."""
    formatter = logging.Formatter(ANALYSIS_LOGGING_FORMAT)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
