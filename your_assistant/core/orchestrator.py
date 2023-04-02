"""The orchestrator that uses the agents to orchestrate the conversation.
"""
import os
from abc import ABC, abstractmethod

from your_assistant.core.llms import *
from your_assistant.core.tools import DocumentQA
from your_assistant.core.utils import load_env


class Orchestrator(ABC):
    """The abstract orchestrator."""

    def __init__(self, verbose: bool = False):
        load_env()
        self.verbose = verbose

    @abstractmethod
    def process(self, prompt: str):
        """Process the prompt.

        Args:
            prompt (str): The prompt to the agent.
        """
        raise NotImplementedError


class RevChatGPTOrchestrator(Orchestrator):
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


class RevBardOrchestrator(Orchestrator):
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


class QAOrchestrator(Orchestrator):
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
