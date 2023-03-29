"""Test the LLMs.
Run this test with command: pytest your_assistant/tests/core/test_llms.py
"""
import os

import pytest
from langchain import PromptTemplate

import your_assistant.core.llms as llms
from your_assistant.core.utils import load_env


@pytest.fixture()
def setup():
    test_folder_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(test_folder_path)))
    config = load_env(env_file_path=os.path.join(root_path, ".env.template"))
    assert "abc" == os.getenv("CHATGPT_SESSION_TOKEN")
    assert "bard" == os.getenv("BARD_SESSION_TOKEN")


class TestLLMs:
    def test_revchatgpt(self, setup):
        """Test the RevChatGPT LLM."""
        llm = llms.RevChatGPT(test_mode=True)
        assert llm("This is a test prompt.") == "This is a test response."

    def test_revbard(self, setup):
        """Test the RevBard LLM."""
        llm = llms.RevBard(test_mode=True)
        assert llm("This is a test prompt.") == "This is a test response."
