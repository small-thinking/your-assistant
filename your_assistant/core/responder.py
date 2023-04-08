"""Core logic of the responders.
"""

import os
import textwrap
from typing import Any, Dict, List

from colorama import Fore
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import FakeEmbeddings, OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

import your_assistant.core.llm as llm_lib
import your_assistant.core.utils as utils


class DocumentQA:
    """Answer a question based on a given vector store."""

    def __init__(
        self,
        db_name: str = "faiss.db",
        llm_type: str = "RevChatGPT",
        use_memory: bool = True,
        memory_token_size: int = 300,
        test_mode: bool = False,
        verbose: bool = False,
        max_token_size: int = 1000,
    ):
        self.logger = utils.Logger("DocumentQA")
        self.db_index_name = os.path.join(db_name, "index")
        self.llm: Any = None
        # Init the LLM.
        if llm_type == "ChatGPT":
            self.llm = llm_lib.ChatGPT()
        elif llm_type == "RevBard":
            self.llm = llm_lib.RevBard()
        else:
            self.llm = llm_lib.RevChatGPT()
        self.use_memory = use_memory
        if self.use_memory:
            self.memory: ConversationSummaryBufferMemory = (
                ConversationSummaryBufferMemory(
                    llm=self.llm, max_token_limit=memory_token_size
                )
            )
        if test_mode:
            self.embeddings_tool = FakeEmbeddings()  # type: ignore
        else:
            self.embeddings_tool = OpenAIEmbeddings()  # type: ignore
        self.verbose = verbose
        self.max_token_size = max_token_size
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
        if self.use_memory:
            history: Dict[str, Any] = self.memory.load_memory_variables({})
            if self.verbose:
                self.logger.info(f"History: {history}\n\n")
            prompt_with_hist = textwrap.dedent(
                f"""
                Past conversations for references:
                {history["history"]}

                The current round of the conversation:
                {prompt}
            """
            )
        else:
            prompt_with_hist = prompt
        truncated_prompt = utils.truncate_text_by_tokens(
            text=prompt_with_hist, max_token_size=self.max_token_size
        )
        if self.verbose:
            self.logger.info(
                Fore.GREEN + f"Prompt: {truncated_prompt}\n\n" + Fore.RESET
            )
        answer = self.llm(prompt=truncated_prompt)
        if self.use_memory:
            # Only save the user original prompt without history augmentation.
            self.memory.save_context(inputs={"user": prompt}, outputs={"AI": answer})
        answer = f"{answer}."
        return answer

    def _concate_docs(self, docs: List[Document]) -> str:
        """Concatenate a list of documents into a single string.
        Args:
            docs (List[Document]): The documents to concatenate.
        """
        doc_snippets = [str(doc) for doc in docs]
        return "\n".join(doc_snippets)
