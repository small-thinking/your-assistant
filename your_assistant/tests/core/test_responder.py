"""Test the utils.
Run this test with command: pytest your_assistant/tests/core/test_indexer.py
"""
import os

import pytest

from your_assistant.core.utils import load_env


@pytest.fixture()
def setup() -> str:
    test_folder_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(test_folder_path)))
    return root_path


class TestResponder:
    pass
