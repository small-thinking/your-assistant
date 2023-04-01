"""Utilities.
"""
import inspect
import logging
import os
from typing import Any, List
from urllib.parse import urlparse

import nltk
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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

    def error(self, message):
        caller_frame = inspect.stack()[1]
        caller_name = caller_frame[3]
        caller_line = caller_frame[2]

        self.logger.error(f"({caller_name} L{caller_line}): {message}")


class PDFIndexer:
    """Index a PDF file."""

    def __init__(self):
        nltk.download("averaged_perceptron_tagger")
        self.logger = Logger("PDFIndexer")

    def index(self, file_path: str):
        loader = self._init_loader(file_path)
        documents = self._extract_data(loader)

    def _init_loader(self, file_path: str) -> Any:
        """Index a PDF file.

        Args:
            file_path (str): The path to the file. Can be a url.
        """
        try:
            result = urlparse(file_path)
            if all([result.scheme, result.netloc]):
                self.logger.info("Load online pdf loader.")
                loader = OnlinePDFLoader(file_path)
            elif os.path.exists(file_path):
                self.logger.info("Load local pdf loader.")
                loader = PyPDFLoader(file_path)
            else:
                raise ValueError(f"File not found: {file_path}")
        except ValueError:
            raise ValueError(f"Error happens when initialize the pdf loader.")
        return loader

    def _extract_data(self, loader: Any):
        """Index a PDF file.

        Args:
            loader (Any): The loader to load the file.
        """
        # OpenAI embeddings are limited to 8191 tokens.
        # See: https://platform.openai.com/docs/guides/embeddings/what-are-embeddings.
        documents = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=6000, chunk_overlap=500
            )
        )
        return documents

    def _get_embeddings(self, documents: List[Document]):
        """Index a PDF file.

        Args:
            documents (Any): The documents to index.
        """
        pass


# # file_path = "https://arxiv.org/pdf/2103.00020.pdf"
# # file_path = "/Users/yjiang/Downloads/foundation-model.pdf"
# file_path = "/Users/yjiang/Downloads/test-pdf.pdf"
# indexer = PDFIndexer()
# data = indexer.index(file_path)
# print(len(data))

# print(max([len(d.page_content) for d in data]))
# print(data)
# print(data[0])
