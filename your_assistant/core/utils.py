"""Utilities.
"""
import inspect
import itertools
import logging
import os
import ssl
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Iterator, List, Tuple
from urllib.request import Request, urlopen

from colorama import Fore
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize


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
        self.logger.info(
            Fore.BLUE + f"({caller_name} L{caller_line}): {message}" + Fore.RESET
        )

    def error(self, message):
        if not self.verbose:
            return
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame[3]
        caller_line = caller_frame[2]
        self.logger.error(
            Fore.RED + f"({caller_name} L{caller_line}): {message}" + Fore.RESET
        )

    def warning(self, message):
        if not self.verbose:
            return
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame[3]
        caller_line = caller_frame[2]
        self.logger.warning(
            Fore.YELLOW + f"({caller_name} L{caller_line}): {message}" + Fore.RESET
        )


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


def chunk_list(lst: List[Any], chunk_size: int) -> Iterator:
    """Chunk a list into smaller lists.

    Args:
        lst (List[Any]): The list to be chunked.
        chunk_size (int): The size of each chunk.

    Returns:
        Iterator: The chunked list.
    """
    it = iter(lst)
    return iter(lambda: tuple(itertools.islice(it, chunk_size)), ())


def xml_to_markdown(xml_string: str) -> str:
    """This function is used to convert document annotations in XML format to Markdown.

    Args:
        xml_string (str): The document annotated in XML.

    Returns:
        str: The document annotated in Markdown.
    """
    markdown_text = []
    # Parse the XML
    xml_string = xml_string.replace("<?xml version='1.0' encoding='utf-8'?>", "")
    root = ET.fromstring(xml_string.strip())
    # Iterate through the elements
    for element in root.iter():
        if element.tag.endswith("p"):
            markdown_text.append("\n")

        if element.tag.endswith("a"):
            markdown_text.append(f"[{element.text}]({element.get('href')})")
        elif element.tag.endswith("span"):
            if "Body-Italics" in element.get("class", ""):
                markdown_text.append(f"*{element.text}*")
            elif "Body-Superscript" in element.get("class", ""):
                markdown_text.append(f"<sup>{element.text}</sup>")
            else:
                markdown_text.append(element.text or "")
        elif element.tag.endswith("hr"):
            markdown_text.append("\n---\n")
        elif element.tag.endswith("ul"):
            markdown_text.append("\n")
        elif element.tag.endswith("li"):
            markdown_text.append(f"\n- ")
        else:
            markdown_text.append(element.text or "")

    return "".join(markdown_text).strip()
