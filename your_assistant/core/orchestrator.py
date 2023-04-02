"""The orchestrator that uses the agents to orchestrate the conversation.
"""
import os
from abc import ABC, abstractmethod

from your_assistant.core.indexer import KnowledgeIndexer
from your_assistant.core.llms import RevBard, RevChatGPT
from your_assistant.core.responder import DocumentQA
from your_assistant.core.utils import load_env


class Orchestrator(ABC):
    """The abstract orchestrator."""

    def __init__(self, verbose: bool = False):
        load_env()
        self.verbose = verbose


class ReadOrchestrator(Orchestrator):
    """The abstract reade orchestrator."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)

    @abstractmethod
    def process(self, prompt: str):
        """Process the prompt.

        Args:
            prompt (str): The prompt to the agent.
        """
        raise NotImplementedError


class RevChatGPTOrchestrator(ReadOrchestrator):
    """The orchestrator that uses the RevChatGPT."""

    def __init__(self, verbose: bool = False):
        """Initialize the orchestrator."""
        super().__init__(verbose=verbose)
        self.llm = RevChatGPT()

    def process(self, prompt: str):
        """Process the prompt.

        Args:
            prompt (str): The prompt to the agent.
        """
        if len(prompt) == 0:
            return ""
        response = self.llm(prompt)
        return response


class RevBardOrchestrator(ReadOrchestrator):
    """The orchestrator that uses the RevBard."""

    def __init__(self, verbose: bool = False):
        """Initialize the orchestrator."""
        super().__init__(verbose=verbose)
        self.llm = RevBard()

    def process(self, prompt: str) -> str:
        """Process the prompt.

        Args:
            prompt (str): The prompt to the agent.
        """
        if len(prompt) == 0:
            return ""
        response = self.llm(prompt)
        return response


class KnowledgeIndexOrchestrator(Orchestrator):
    def __init__(self, db_name: str = "faiss.db", verbose: bool = False):
        """Initialize the orchestrator."""
        super().__init__(verbose=verbose)
        self.indexer = KnowledgeIndexer(db_name=db_name)

    def process(
        self, path: str, chunk_size: int = 1000, chunk_overlap: int = 100
    ) -> str:
        """Process the index according to the path.

        Args:
            path (str): The path to the data to be indexed.
        """
        response = self.indexer.index(
            path=path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return response


class QAOrchestrator(ReadOrchestrator):
    """The orchestrator that uses the QA agent."""

    def __init__(
        self,
        db_name: str = "faiss.db",
        llm_type: str = "RevChatGPT",
        test_mode: bool = False,
        verbose: bool = False,
    ):
        """Initialize the orchestrator."""
        super().__init__(verbose=verbose)
        self.qa = DocumentQA(db_name=db_name, llm_type=llm_type, test_mode=test_mode)

    def process(self, prompt: str) -> str:
        """Process the prompt.

        Args:
            prompt (str): The prompt to the agent.
        """
        if len(prompt) == 0:
            return ""
        response = self.qa.answer(prompt)
        return response


qa = QAOrchestrator()
res = qa.process("What is the definition of OKR?")
print(res)
