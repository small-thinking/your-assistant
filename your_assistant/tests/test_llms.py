"""Test the LLMs.
Run this test with command: pytest your_assistant/tests/test_llms.py
"""
from langchain import PromptTemplate
import os
import pytest
from your_assistant.core.utils import Config
import your_assistant.core.llms as llms


@pytest.fixture()
def setup():
    test_folder_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(test_folder_path))
    config = Config(os.path.join(root_path, ".env.template"))
    assert "abc" == os.getenv("CHATGPT_SESSION_TOKEN")


class TestLLMs:
    def test_revchatgpt(self, setup):
        """Test the RevChatGPT LLM."""
        llm = llms.RevChatGPT(test_mode=True)
        assert llm("This is a test prompt.") == "This is a test response."
