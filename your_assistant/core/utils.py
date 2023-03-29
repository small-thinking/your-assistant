"""Utilities.
"""
import inspect
import logging
from typing import Any

from dotenv import load_dotenv


def load_env(env_file_path: str = None):
    load_dotenv(env_file_path)


# Create a logger class that accept level setting.
# The logger should be able to log to stdout and display the datetime, caller, and line of code.
class Logger:
    def __init__(self, logger_name: str, level: Any = logging.INFO):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level=level)
        self.formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s (%(filename)s:%(lineno)d)"
        )

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level=level)
        self.console_handler.setFormatter(self.formatter)

        self.logger.addHandler(self.console_handler)

    def info(self, message):
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame[3]
        caller_line = caller_frame[2]

        self.logger.info(f"({caller_name} L{caller_line}): {message}")
