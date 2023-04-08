"""Core logic of the responders.
"""

import os
from typing import Any, List, Union

from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import FakeEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

import your_assistant.core.llm as llm
import your_assistant.core.utils as utils


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
        self.db_index_name = os.path.join(db_name, "index")
        self.llm: Any = None
        # Init the LLM.
        if llm_type == "ChatGPT":
            self.llm = llm.ChatGPT()
        elif llm_type == "RevBard":
            self.llm = llm.RevBard()
        else:
            self.llm = llm.RevChatGPT()
        if test_mode:
            self.embeddings_tool = FakeEmbeddings()  # type: ignore
        else:
            self.embeddings_tool = OpenAIEmbeddings()  # type: ignore
        self.verbose = verbose
        prompt_template = """
            Please provide an informative ANSWER to the following question based on the retrieved document snippets.
            DO NOT use your own context knowledge. The answer should be in the same language as the question.

            After the answer, PLEASE PROVIDE THE context of the sources, in the following format:
            <The word 'Source' translated in question language>: <original words>, <document path>, <page number>.

            Question: {question}
            Document snippets: {doc_snippets}

        """
        self.prompt_template = PromptTemplate(
            input_variables=["question", "doc_snippets"],
            template=prompt_template,
        )

    def answer(self, question: str, k: int = 5) -> str:
        """Answer a given question.

        Args:
            question (str): The question to answer.
        """
        loaded_db = FAISS.load_local(self.db_index_name, self.embeddings_tool)
        retriever = VectorStoreRetriever(vectorstore=loaded_db, search_type="mmr", k=k)  # type: ignore
        docs = retriever.get_relevant_documents(question)
        if self.verbose:
            self.logger.info(f"Retrieved {len(docs)} documents.")
        if self.verbose:
            for idx, doc in enumerate(docs):
                self.logger.info(f"Doc {idx + 1}:\n {doc}")
        doc_snippets = self._concate_docs(docs)
        prompt = self.prompt_template.format(
            question=question, doc_snippets=doc_snippets
        )
        answer = self.llm(prompt=prompt)
        answer = f"Question: {question}.\nThe answer: {answer}."
        return answer

    def _concate_docs(self, docs: List[Document]) -> str:
        """Concatenate a list of documents into a single string.

        Args:
            docs (List[Document]): The documents to concatenate.
        """
        doc_snippets = [str(doc) for doc in docs]
        return "\n".join(doc_snippets)
