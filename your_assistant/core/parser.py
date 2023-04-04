"""Implementation for the data parsers.
"""
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import ebooklib
import html2text
import mobi
from ebooklib import epub
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import *


class MobiLoader(BaseLoader):
    """Loader for e-books."""

    _DEFAULT_CHUNK_SIZE = 700

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def load(self) -> List[Document]:
        """Parse a .mobi file and extract the authors, title, and content per page.

        Returns:
            A dictionary containing authors, title, and chunked contents.
        """
        file_extension = os.path.splitext(self.path)[1]

        if file_extension != ".mobi":
            raise ValueError(f"Unsupported file format: {file_extension}")
        tempdir, filepath = mobi.extract(self.path)
        with open(filepath, "r") as file:
            content = file.read()
        # Put everything into a single document. Text splitter will chunk it.
        content = html2text.html2text(content)
        document = Document(page_content=content, metadata={"source": self.path})
        shutil.rmtree(tempdir)
        return [document]


# def run():
#     # file_path = "/Users/yjiang/Downloads/docs/First, Break All the Rules_ Wha - Marcus Buckingham.epub"
#     file_path = "/Users/yjiang/Downloads/docs/Philosophical Investigations - Ludwig Wittgenstein.mobi"
#     mobi_loader = MobiLoader(file_path)
#     chunk_size, chunk_overlap = 500, 50
#     documents = mobi_loader.load_and_split(
#         text_splitter=TokenTextSplitter(
#             chunk_size=chunk_size, chunk_overlap=chunk_overlap
#         )
#     )
#     print(len(documents))
#     for i in range(5):
#         print(f"Document {i}")
#         print(documents[i])
#         print("\n")


# run()
