"""Core logic of the tools.

"""
import os
from typing import Any, List, Union
from urllib.parse import urlparse

import nltk
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.document_loaders import OnlinePDFLoader, PyPDFLoader
from langchain.embeddings import FakeEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

import your_assistant.core.llms as llms
import your_assistant.core.utils as utils


class PDFIndexer:
    """Index a PDF file into a vector DB."""

    def __init__(self, db_name: str = "faiss.db"):
        nltk.download("averaged_perceptron_tagger")
        self.logger = utils.Logger("PDFIndexer")
        self.db_name = db_name
        self.embeddings_tool = OpenAIEmbeddings()

    def index(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Index a given PDF file into the vector DB according to the name.

        Args:
            file_path (str): The path to the file. Can be a url.
        """
        loader = self._init_loader(file_path=file_path)
        documents = self._extract_data(
            loader=loader, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self._index_embeddings(documents=documents)
        return documents

    def _init_loader(self, file_path: str) -> Union[PyPDFLoader, OnlinePDFLoader]:
        """Index a PDF file.

        Args:
            file_path (str): The path to the file. Can be a url.
        """
        loader: Union[PyPDFLoader, OnlinePDFLoader]
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

    def _index_embeddings(self, documents: List[Document], db: Any = None):
        """Index a PDF file.

        Args:
            documents (Any): The documents to index.
        """
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
        self.logger.info(f"DB saved to {self.db_name}.")


class DocumentQA:
    """Answer a question based on a given vector store."""

    def __init__(
        self,
        db_name: str = "faiss.db",
        llm_type: str = "RevChatGPT",
        test_mode: bool = False,
        verbose: bool = False,
    ):
        self.logger = utils.Logger("DocumentQA")
        self.db_name = db_name
        # Init the LLM.
        if llm_type == "RevBard":
            self.llm = llms.RevBard()
        else:
            self.llm = llms.RevChatGPT()
        if test_mode:
            self.embeddings_tool = FakeEmbeddings()
        else:
            self.embeddings_tool = OpenAIEmbeddings()
        self.verbose = verbose
        prompt_template = """
            Please answer the following question based on the retrieved document snippets. Do not use your own context knowledge.
            Please also provide the source if there exists, in the format of content, document name, page number, at the end of the answer.
            Question: {question}
            Document snippets: {doc_snippets}
        """
        self.prompt_template = PromptTemplate(
            input_variables=["question", "doc_snippets"],
            template=prompt_template,
        )

    def answer(self, question: str) -> str:
        """Answer a given question.

        Args:
            question (str): The question to answer.
        """
        loaded_db = FAISS.load_local(self.db_name, self.embeddings_tool)
        retriever = loaded_db.as_retriever()
        docs = retriever.get_relevant_documents(question)
        self.logger.info(f"Retrieved {len(docs)} documents.")
        if self.verbose:
            for idx, doc in enumerate(docs):
                self.logger.info(f"Doc {idx + 1}:\n {doc}")
        doc_snippets = self._concate_docs(docs)
        prompt = self.prompt_template.format(
            question=question, doc_snippets=doc_snippets
        )
        answers = self.llm(prompt=prompt)
        return answers

    def _concate_docs(self, docs: List[Document]) -> str:
        """Concatenate a list of documents into a single string.

        Args:
            docs (List[Document]): The documents to concatenate.
        """
        doc_snippets = [str(doc) for doc in docs]
        return "\n".join(doc_snippets)
