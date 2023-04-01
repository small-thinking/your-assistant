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
    return root_path


class TestLLMs:
    @pytest.mark.parametrize(
        "config_file, test_mode, expected",
        [
            (".env.template", True, "This is a test response."),
            (
                ".env.not-exist",
                False,
                "Please set CHATGPT_ACCESS_TOKEN before chatting with ChatGPT.",
            ),
        ],
    )
    def test_revchatgpt(self, setup, config_file, test_mode, expected):
        """Test the RevChatGPT LLM."""
        root_path = setup
        for key in os.environ:
            del os.environ[key]
        load_env(env_file_path=os.path.join(root_path, config_file))
        llm = llms.RevChatGPT(test_mode=test_mode)
        assert llm("This is a test prompt.") == expected

    @pytest.mark.parametrize(
        "config_file, test_mode, expected",
        [
            (".env.template", True, "This is a test response."),
            (
                ".env.not-exist",
                False,
                "Please set BARD_SESSION_TOKEN before chatting with Bard.",
            ),
        ],
    )
    def test_revbard(self, setup, config_file, test_mode, expected):
        """Test the RevBard LLM."""
        root_path = setup
        for key in os.environ:
            del os.environ[key]
        load_env(env_file_path=os.path.join(root_path, config_file))
        llm = llms.RevBard(test_mode=test_mode)
        assert llm("This is a test prompt.") == expected
