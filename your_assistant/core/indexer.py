"""Core logic of the indexers.

"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union
from urllib.parse import urlparse

import nltk
from langchain.docstore.document import Document
from langchain.document_loaders import OnlinePDFLoader, UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

import your_assistant.core.utils as utils


class KnowledgeIndexer:
    """Index a PDF file into a vector DB."""

    def __init__(self, db_name: str = "faiss.db"):
        nltk.download("averaged_perceptron_tagger")
        self.logger = utils.Logger("PDFIndexer")
        self.db_name = db_name
        self.embeddings_tool = OpenAIEmbeddings()  # type: ignore
        self.index_record_path = Path(os.path.join(db_name, "index_record.json"))
        # Load the index record. Create one if not exist.
        if not self.index_record_path.exists():
            self.logger.info(
                f"Index record file {self.index_record_path} does not exist. Create one."
            )
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

        loader = self._init_loader(path=path)
        documents = self._extract_data(
            loader=loader, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._index_embeddings(documents=documents)
        return documents

    def _init_loader(self, path: str) -> Union[UnstructuredFileLoader, OnlinePDFLoader]:
        """Index the data according to the path.

        Args:
            path (str): The path to the data. Can be a url or a directory.
        """
        loader: Union[UnstructuredFileLoader, OnlinePDFLoader]
        try:
            result = urlparse(path)
            if all([result.scheme, result.netloc]):
                self.logger.info("Load online pdf loader.")
                loader = OnlinePDFLoader(path)
            elif os.path.exists(path):
                self.logger.info("Load local pdf loader.")
                loader = UnstructuredFileLoader(path)
            else:
                raise ValueError(f"File not found: {path}")
        except ValueError:
            raise ValueError(f"Error happens when initialize the pdf loader.")
        return loader

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

    def _index_embeddings(self, documents: List[Document]):
        """Index a PDF file.

        Args:
            documents (Any): The documents to index.
        """
        # Drop the documents that are already indexed.
        documents = list(
            filter(
                lambda d: "source" not in d.metadata
                or d.metadata["source"] not in self.index_record,
                documents,
            )
        )
        db = None
        if os.path.exists(self.db_name):
            self.logger.info(f"DB [{self.db_name}] exists, load it.")
            db = FAISS.load_local(self.db_name, self.embeddings_tool)
        new_db = FAISS.from_documents(documents, self.embeddings_tool)
        if db:
            db.merge_from(new_db)
        else:
            db = new_db
        self.logger.info(f"Indexing done. {len(documents)} documents indexed.")
        db.save_local(self.db_name)
        # Record the newly indexed documents. Delete the old index first.
        map(
            lambda d: self.index_record["indexed_doc"].add(d.metadata["source"] or ""),
            documents,
        )
        self.index_record["indexed_doc"] = list(self.index_record["indexed_doc"])
        self.index_record_path.unlink()
        with self.index_record_path.open("w") as f:
            json.dump(self.index_record, f)
            self.logger.info("Updated index record.")
        self.logger.info(f"DB saved to {self.db_name}.")
