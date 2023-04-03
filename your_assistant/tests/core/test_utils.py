"""Test the utils.
Run this test with command: pytest your_assistant/tests/core/test_utils.py
"""
import os

import pytest
from langchain.document_loaders import PyPDFLoader

import your_assistant.core.utils as utils


@pytest.mark.parametrize(
    "input, chunk_size, expected",
    [
        (
            list("Hello world"),
            3,
            [("H", "e", "l"), ("l", "o", " "), ("w", "o", "r"), ("l", "d")],
        ),
        (range(14), 3, [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13)]),
    ],
)
def test_chunk_iterator(input, chunk_size, expected):
    chunk_iterator = utils.chunk_iterator(input, chunk_size)
    assert list(chunk_iterator) == expected
