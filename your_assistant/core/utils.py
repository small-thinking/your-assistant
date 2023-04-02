"""Utilities.
"""
import inspect
import logging
import os
import ssl
import urllib.parse
from typing import Any, Tuple
from urllib.request import Request, urlopen

from dotenv import load_dotenv


def load_env(env_file_path: str = ""):
    if env_file_path:
        load_dotenv(env_file_path)
    else:
        load_dotenv()


# Create a logger class that accept level setting.
# The logger should be able to log to stdout and display the datetime, caller, and line of code.
class Logger:
    def __init__(
        self, logger_name: str, verbose: bool = True, level: Any = logging.INFO
    ):
        self.logger = logging.getLogger(logger_name)
        self.verbose = verbose
        self.logger.setLevel(level=level)
        self.formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s (%(filename)s:%(lineno)d)"
        )

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level=level)
        self.console_handler.setFormatter(self.formatter)

        self.logger.addHandler(self.console_handler)

    def info(self, message):
        if not self.verbose:
            return
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame[3]
        caller_line = caller_frame[2]
        self.logger.info(f"({caller_name} L{caller_line}): {message}")

    def error(self, message):
        if not self.verbose:
            return
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame[3]
        caller_line = caller_frame[2]
        self.logger.error(f"({caller_name} L{caller_line}): {message}")

    def warning(self, message):
        if not self.verbose:
            return
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame[3]
        caller_line = caller_frame[2]
        self.logger.warning(f"({caller_name} L{caller_line}): {message}")


def file_downloader(url: str, retry_with_no_verify: bool = True) -> Tuple[str, str]:
    """Download a file from a given url.

    Args:
        url (str): The url to download the file from.
        retry_with_no_verify (bool, optional): Whether to retry the download with

    Returns:
        Tuple[str, str]: The url and the path to the downloaded file.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_0_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
    }
    logger = Logger("file_downloader")
    filepath = os.path.basename(urllib.parse.urlparse(url).path)
    req = Request(url, headers=headers)
    try:
        response = urlopen(req)
    except urllib.error.URLError:
        if not retry_with_no_verify:
            raise
        logger.warning("SSL certificate verification error. Ignore it.")
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        response = urlopen(req, context=context)

    with open(filepath, "wb") as outfile:
        outfile.write(response.read())

    return url, filepath
