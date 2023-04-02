"""Core logic of the indexers.

"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import nltk
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

import your_assistant.core.utils as utils


class KnowledgeIndexer:
    """Index a PDF file into a vector DB."""

    def __init__(self, db_name: str = "faiss.db", verbose: bool = False):
        nltk.download("averaged_perceptron_tagger")
        self.logger = utils.Logger("KnowledgeIndexer", verbose=verbose)
        self.db_name = db_name
        self.db_record_name = os.path.join(db_name, "index_record.json")
        self.db_index_name = os.path.join(db_name, "index")
        self.verbose = verbose
        self.embeddings_tool = OpenAIEmbeddings()  # type: ignore
        self.index_record_path = Path(self.db_record_name)
        # Load the index record. Create one if not exist.
        if not self.index_record_path.exists():
            self.logger.info(
                f"Index record file {self.index_record_path} does not exist. Create one."
            )
            if not os.path.exists(os.path.dirname(self.index_record_path)):
                os.makedirs(os.path.dirname(self.index_record_path))
            self.index_record_path.touch()
        self.index_record: Dict[str, Any] = {}
        with self.index_record_path.open("r+") as f:
            try:
                self.index_record = json.load(f)
            except json.decoder.JSONDecodeError:
                self.index_record["indexed_doc"] = {}
        self.index_record["indexed_doc"] = set(self.index_record["indexed_doc"])

    def index(self, path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Index a given file into the vector DB according to the name.

        Args:
            path (str): The path to the file. Can be a url.
        """
        self.logger.info(f"Indexing {path}...")
        loader, source, filepath = self._init_loader(path=path)
        documents = self._extract_data(
            loader=loader, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._index_embeddings(documents=documents, source=source)
        # Remove the downloaded file.
        if os.path.exists(filepath):
            os.remove(filepath)
        return documents

    def _init_loader(self, path: str) -> Tuple[UnstructuredFileLoader, str, str]:
        """Index the data according to the path.

        Args:
            path (str): The path to the data. Can be a url or a directory.

        Returns:
            Tuple[UnstructuredFileLoader, str, str]: The loader, the source, and the downloaded filepath.
        """
        try:
            # If the path is a url, download the file.
            result = urlparse(path)
            filepath = ""
            if all([result.scheme, result.netloc]):
                self.logger.info("Download online file.")
                source, filepath = utils.file_downloader(url=path)
                loader = UnstructuredFileLoader(filepath)
            elif os.path.exists(path):
                self.logger.info("Load local loader.")
                loader = UnstructuredFileLoader(path)
                source = path
            else:
                raise ValueError(f"File not found: {path}")
        except ValueError:
            raise ValueError(f"Error happens when initialize the data loader.")
        return loader, source, filepath

    def _extract_data(
        self, loader: Any, chunk_size: int = 1000, chunk_overlap: int = 100
    ):
        """Index a PDF file.

        Args:
            loader (Any): The loader to load the file.
        """
        if chunk_size <= chunk_overlap:
            raise ValueError(
                f"Chunk size [{chunk_size}] must be larger than chunk overlap [{chunk_overlap}]."
            )
        # OpenAI embeddings are limited to 8191 tokens.
        # See: https://platform.openai.com/docs/guides/embeddings/what-are-embeddings.
        documents = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        )
        return documents

    def _index_embeddings(self, documents: List[Document], source: str):
        # type: ignore
        """Index a PDF file.

        Args:
            documents (Any): The documents to index.
            source (str): The source of the documents.
        """
        self.index_record["indexed_doc"] = set(self.index_record["indexed_doc"])
        if source in self.index_record["indexed_doc"]:
            self.logger.info(f"File {source} already indexed. Skip.")
            return
        # Update the source of each document.
        for doc in documents:
            doc.metadata["source"] = source
            doc.page_content = re.sub(r"[^\w\s]|['\"]", "", doc.page_content)
        db = None
        if os.path.exists(self.db_index_name):
            self.logger.info(f"DB [{self.db_index_name}] exists, load it.")
            db = FAISS.load_local(self.db_index_name, self.embeddings_tool)
        new_db = FAISS.from_documents(documents, self.embeddings_tool)
        if db:
            db.merge_from(new_db)
        else:
            db = new_db
        self.logger.info(f"Indexing done. {len(documents)} documents indexed.")
        db.save_local(self.db_index_name)
        # Record the newly indexed documents. Delete the old index first.
        self.index_record["indexed_doc"].add(source)
        self.index_record["indexed_doc"] = list(self.index_record["indexed_doc"])
        index_size = len(self.index_record["indexed_doc"])
        self.index_record_path.unlink()
        with self.index_record_path.open("w") as f:
            json.dump(self.index_record, f)
            self.logger.info(f"Updated index record with [{index_size}] records.")
        self.logger.info(f"DB saved to {self.db_name}.")
