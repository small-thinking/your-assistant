"""Core logic of the responders.
"""

from typing import Any, List, Union

from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import FakeEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS

import your_assistant.core.llms as llms
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
        self.db_name = db_name
        self.llm: Any = None
        # Init the LLM.
        if llm_type == "RevBard":
            self.llm = llms.RevBard()
        else:
            self.llm = llms.RevChatGPT()
        if test_mode:
            self.embeddings_tool = FakeEmbeddings()  # type: ignore
        else:
            self.embeddings_tool = OpenAIEmbeddings()  # type: ignore
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
