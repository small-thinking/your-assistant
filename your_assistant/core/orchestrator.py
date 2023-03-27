"""The orchestrator that uses the agents to orchestrate the conversation.
"""
import os
from abc import ABC, abstractmethod

from your_assistant.core.llms import RevChatGPT
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
