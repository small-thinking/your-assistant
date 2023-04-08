"""Test the LLMs.
Run this test with command: pytest your_assistant/tests/core/test_llm.py
"""
import os

import pytest
from langchain import PromptTemplate

import your_assistant.core.llm as llm_lib
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
            (".env.template", True, "This is a test chatgpt response."),
            (
                ".env.not-exist",
                False,
                "Please set OPENAI_API_KEY before chatting with ChatGPT.",
            ),
        ],
    )
    def test_chatgpt(self, setup, config_file, test_mode, expected):
        """Test the ChatGPT LLM."""
        root_path = setup
        for key in os.environ:
            del os.environ[key]
        load_env(env_file_path=os.path.join(root_path, config_file))
        llm = llm_lib.ChatGPT(test_mode=test_mode)
        assert llm("This is a test chatgpt prompt.") == expected

    @pytest.mark.parametrize(
        "config_file, test_mode, expected",
        [
            (".env.template", True, "This is a test revchatgpt response."),
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
        llm = llm_lib.RevChatGPT(test_mode=test_mode)
        assert llm("This is a test prompt.") == expected

    @pytest.mark.parametrize(
        "config_file, test_mode, expected",
        [
            (".env.template", True, "This is a test revbard response."),
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
        llm = llm_lib.RevBard(test_mode=test_mode)
        assert llm("This is a test prompt.") == expected
