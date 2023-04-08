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

from your_assistant.core.utils import xml_to_markdown


class MobiLoader(BaseLoader):
    """Loader for e-books."""

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


class EpubLoader(BaseLoader):
    """Loader for epub documents."""

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def load(self) -> List[Document]:
        """Parse a .epub file and extract the authors, title, and content per page.

        Returns:
            A dictionary containing authors, title, and chunked contents.
        """
        file_extension = os.path.splitext(self.path)[1]

        if file_extension != ".epub":
            raise ValueError(f"Unsupported file format: {file_extension}")

        book = epub.read_epub(self.path)
        title = book.get_metadata("DC", "title")[0][0]
        authors = [author[0] for author in book.get_metadata("DC", "creator")]

        documents: List[Document] = []
        page_number = 0
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                page_number += 1
                content = item.get_content().decode("utf-8")
                if "<?xml version='1.0' encoding='utf-8'?>" in content:
                    content = xml_to_markdown(content)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": self.path,
                            "title": title,
                            "authors": authors,
                            "page": page_number,
                        },
                    )
                )
        return documents
