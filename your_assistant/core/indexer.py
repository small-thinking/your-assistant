"""Core logic of the indexers.

"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlparse

import nltk
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS, VectorStore

import your_assistant.core.loader as loader_lib
import your_assistant.core.utils as utils


class KnowledgeIndexer:
    def __init__(self, args: argparse.Namespace):
        """Initialize the knowledge indexer.
        Needed arguments:
            verbose: Whether to print out the verbose logs. (Default: False)
            db_path: The path to the vector database.
            embeddings_tool_name: The name of the embedding tool to use, e.g. openai.
        """
        self.verbose = False if not args.verbose else args.verbose
        nltk.download("averaged_perceptron_tagger")
        self.logger = utils.Logger("KnowledgeIndexer", verbose=self.verbose)
        self.supported_file_types: Set[str] = self._init_supported_file_types()
        self.embeddings_tool = self._init_embeddings_tool(args=args)
        # Initialize the db index engine (e.g. FAISS) and db index record.
        if not args.db_path:
            raise ValueError("db_path is not specified.")
        self.db_path = args.db_path
        self._init_index_db(args=args, embeddings_tool=self.embeddings_tool)
        self._init_index_recorder(args=args)

    def _init_supported_file_types(self) -> Set[str]:
        """Initialize the supported file types.

        Returns:
            A dictionary of supported file types.
        """
        return set([".pdf", ".mobi", ".epub", ".txt", ".html"])

    def _init_embeddings_tool(self, args: argparse.Namespace) -> Embeddings:
        """Initialize the embedding tool.

        Args:
            args (argparse.Namespace): The arguments passed in.
        """
        if not args.embeddings_tool_name:
            raise ValueError("embeddings_tool_name is not specified.")
        if args.embeddings_tool_name == "openai":
            return OpenAIEmbeddings()
        raise ValueError(f"Unsupported embeddings tool: {args.embeddings_tool_name}.")

    def _init_index_db(
        self, args: argparse.Namespace, embeddings_tool: Embeddings
    ) -> None:
        """Initialize the index engine.
        If the db already exists, use the negine to load it in memory.

        Args:
            args (argparse.Namespace): The arguments passed in.
            embeddings_tool (Embeddings): The embeddings tool to use.
        """
        if not args.db_path:
            raise ValueError("db_path is not specified.")
        self.db_index_path = os.path.join(args.db_path, "index")
        self.embeddings_db_engine = FAISS
        self.embeddings_db: VectorStore = None
        if os.path.exists(self.db_index_path):
            self.logger.info(f"DB [{self.db_index_path}] exists, load it.")
            self.embeddings_db = self.embeddings_db_engine.load_local(
                self.db_index_path, embeddings_tool
            )

    def _init_index_recorder(self, args: argparse.Namespace) -> None:
        """Initialize the index recorder. The index recorder stores the information
        on which document has been indexed.

        Args:
            args (argparse.Namespace): The arguments passed in.
        """
        self.db_record_path = os.path.join(args.db_path, "index_record.json")
        self.index_record_path = Path(self.db_record_path)
        # Create the index record if not exist.
        if not self.index_record_path.exists():
            self.logger.info(
                f"Index record file {self.index_record_path} does not exist. Create one."
            )
            if not os.path.exists(os.path.dirname(self.index_record_path)):
                os.makedirs(os.path.dirname(self.index_record_path))
            self.index_record_path.touch()
        # Load the index record.
        self.index_record: Dict[str, Any] = {}
        with self.index_record_path.open("r+") as f:
            try:
                self.index_record = json.load(f)
            except json.decoder.JSONDecodeError:
                self.index_record["indexed_doc"] = {}
        self.index_record["indexed_doc"] = set(self.index_record["indexed_doc"])

    def index(
        self,
        path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 50,
    ) -> str:
        """Index a given file into the vector DB according to the name.

        Args:
            path (str): The path to the file. Can be a url.
            chunk_size (int, optional): The chunk size to split the text. Defaults to 500.
            chunk_overlap (int, optional): The chunk overlap to split the text. Defaults to 50.
            batch_size (int, optional): The batch size to index the embeddings. Defaults to 50.

        Returns:
            str: The status of the indexing.
        """
        self.logger.info(f"Indexing {path}...")
        loader, source, downloaded_path = self._init_loader(path=path)
        documents = self._extract_data(
            loader=loader, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        is_indexed = self._index_embeddings(
            documents=documents, source=source, batch_size=batch_size
        )
        # Remove the downloaded file.
        if os.path.exists(downloaded_path):
            os.remove(downloaded_path)
        return f"Index {source} finished." if is_indexed else ""

    def _init_loader(self, path: str) -> Tuple[BaseLoader, str, str]:
        """Initialize the loader based on the file path and type.

        Args:
            path (str): The path to the file.

        Returns:
            Tuple[BaseLoader, str, str]: The loader, the source, and the downloaded file path.
        """
        loader: BaseLoader
        try:
            # If the path is a url, download the file.
            result = urlparse(path)
            downloaded_path = ""
            if all([result.scheme, result.netloc]):
                self.logger.info("Download online file.")
                source, downloaded_path = utils.file_downloader(url=path)
            if os.path.exists(path):
                self.logger.info("Load local loader.")
                extension = os.path.splitext(path)[1]
                if extension not in self.supported_file_types:
                    raise ValueError(
                        f"File extension not supported: {os.path.basename(path)}. "
                        + f"Only support {list(sorted(self.supported_file_types))}."
                    )
                if extension == ".mobi":
                    loader = loader_lib.MobiLoader(path=path)
                elif extension == ".epub":
                    loader = loader_lib.EpubLoader(path=path)
                elif extension == ".pdf":
                    loader = loader_lib.PdfLoader(path=path)
                elif extension in self.supported_file_types:
                    loader = UnstructuredFileLoader(path)
                source = path
            else:
                raise ValueError(f"File not found: {os.path.basename(path)}")
        except ValueError as e:
            raise e
        return loader, source, downloaded_path

    def _extract_data(
        self, loader: BaseLoader, chunk_size: int = 500, chunk_overlap: int = 50
    ):
        """Index a PDF file.

        Args:
            loader (Any): The loader to load the file.
        """
        if chunk_size <= chunk_overlap:
            raise ValueError(
                f"Chunk size [{chunk_size}] must be larger than chunk overlap [{chunk_overlap}]."
            )
        documents = loader.load_and_split(
            text_splitter=TokenTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        )
        return documents

    def _index_embeddings(
        self, documents: List[Document], source: str, batch_size: int = 100
    ) -> bool:
        """Index a file.

        Args:
            documents (Any): The documents to index.
            source (str): The source of the documents.
        """
        self.index_record["indexed_doc"] = set(self.index_record["indexed_doc"])
        if source in self.index_record["indexed_doc"]:
            self.logger.info(f"File {source} already indexed. Skip.")
            return False
        # Update the source of each document.
        for doc in documents:
            doc.metadata["source"] = source
            doc.page_content = re.sub(r"[^\w\s]|['\"]", "", doc.page_content)
        # Index the new documents in batches.
        for idx, document_batch in enumerate(utils.chunk_list(documents, batch_size)):
            if self.verbose:
                self.logger.info(
                    f"Indexing {len(document_batch)} documents (batch {idx})."
                )
            new_db = self.embeddings_db_engine.from_documents(
                document_batch, self.embeddings_tool
            )
            if self.embeddings_db:
                self.embeddings_db.merge_from(new_db)  # type: ignore
            else:
                self.embeddings_db = new_db
        self.logger.info(f"Indexing done. {len(documents)} documents indexed.")
        self.embeddings_db.save_local(self.db_index_path)  # type: ignore
        # Record the newly indexed documents. Delete the old index first.
        self.index_record["indexed_doc"].add(source)
        self.index_record["indexed_doc"] = list(self.index_record["indexed_doc"])
        index_size = len(self.index_record["indexed_doc"])
        self.index_record_path.unlink()
        with self.index_record_path.open("w") as f:
            json.dump(self.index_record, f, indent=2)
            self.logger.info(f"Updated index record with [{index_size}] records.")
        self.logger.info(f"DB saved to {self.db_path}.")
        return True
