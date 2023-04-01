"""Test the utils.
Run this test with command: pytest your_assistant/tests/core/test_utils.py
"""
import os

import pytest
from langchain.document_loaders import PyPDFLoader

import your_assistant.core.utils as utils


class TestUtils:
    @pytest.mark.parametrize(
        "file_path, expected",
        [
            (
                "testdata/void.pdf",
                ValueError("Error happens when initialize the pdf loader."),
            ),
            ("testdata/test-pdf.pdf", PyPDFLoader),
        ],
    )
    def test_pdf_indexer_init_loader(self, file_path, expected):
        indexer = utils.PDFIndexer()
        if isinstance(expected, ValueError):
            with pytest.raises(ValueError) as e:
                indexer._init_loader(file_path=file_path)
            assert str(e.value) == "Error happens when initialize the pdf loader."
        else:
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            loader = indexer._init_loader(file_path=file_path)
            assert type(loader) == expected

    @pytest.mark.parametrize(
        "file_path, expected", [("testdata/test-pdf.pdf", "This is a test pdf file.")]
    )
    def test_pdf_indexer_extract_data(self, file_path, expected):
        indexer = utils.PDFIndexer()
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        loader = indexer._init_loader(file_path=file_path)
        data = indexer._extract_data(loader=loader)
        assert len(data) == 1
        assert data[0].page_content == expected
