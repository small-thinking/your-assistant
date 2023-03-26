"""Core logic of custom LLMs.
"""
from langchain.llms.base import LLM
from langchain import PromptTemplate
import os
from revChatGPT.V1 import Chatbot
from typing import Optional, List


class RevChatGPT(LLM):
    test_mode: bool = False

    @property
    def _llm_type(self) -> str:
        return "RevChatGPT"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the LLM. In test mode, return a test response.

        Args:
            prompt (str): The prompt to the LLM.
            stop (Optional[List[str]]): The stop tokens.

        Returns:
            str: The response from the LLM.
        """
        if stop is not None:
            raise ValueError("The stop tokens are not supported by RevChatGPT.")

        response = ""
        if self.test_mode:
            response = "This is a test response."
            return response

        chatgpt = Chatbot(
            config={
                "session_token": os.getenv("CHATGPT_SESSION_TOKEN"),
            }
        )
        for data in chatgpt.ask(prompt):
            response += data["text"]
        return response
