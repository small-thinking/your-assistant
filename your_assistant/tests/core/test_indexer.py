"""Test the utils.
Run this test with command: pytest your_assistant/tests/core/test_indexer.py
"""
import os

import pytest
from langchain.document_loaders import UnstructuredFileLoader

import your_assistant.core.indexer as indexer
from your_assistant.core.utils import load_env


@pytest.fixture()
def setup():
    test_folder_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(test_folder_path)))
    return root_path


class TestTools:
    @pytest.mark.parametrize(
        "config_file, file_path, expected",
        [
            (
                ".env.template",
                "testdata/void.pdf",
                ValueError("Error happens when initialize the pdf loader."),
            ),
            (".env.template", "testdata/test-pdf.pdf", UnstructuredFileLoader),
        ],
    )
    def test_pdf_indexer_init_loader(self, setup, config_file, file_path, expected):
        root_path = setup
        for key in os.environ:
            del os.environ[key]
        load_env(env_file_path=os.path.join(root_path, config_file))
        knowledge_indexer = indexer.KnowledgeIndexer()
        if isinstance(expected, ValueError):
            with pytest.raises(ValueError) as e:
                knowledge_indexer._init_loader(file_path=file_path)
            assert str(e.value) == expected.args[0]
        else:
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            loader = knowledge_indexer._init_loader(file_path=file_path)
            assert type(loader) == expected

    @pytest.mark.parametrize(
        "config_file, file_path, chunk_size, chunk_overlap, expected",
        [
            (
                ".env.template",
                "testdata/test-pdf.pdf",
                1000,
                200,
                "This is a test pdf file.",
            ),
            (
                ".env.template",
                "testdata/test-pdf.pdf",
                100,
                200,
                ValueError("Chunk size [100] must be larger than chunk overlap [200]."),
            ),
        ],
    )
    def test_pdf_indexer_extract_data(
        self, setup, config_file, file_path, chunk_size, chunk_overlap, expected
    ):
        root_path = setup
        for key in os.environ:
            del os.environ[key]
        load_env(env_file_path=os.path.join(root_path, config_file))
        knowledge_indexer = indexer.KnowledgeIndexer()
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        loader = knowledge_indexer._init_loader(file_path=file_path)
        if isinstance(expected, ValueError):
            with pytest.raises(ValueError) as e:
                knowledge_indexer._extract_data(
                    loader=loader, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            assert str(e.value) == expected.args[0]
        else:
            data = knowledge_indexer._extract_data(
                loader=loader, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            assert len(data) == 1
            assert data[0].page_content == expected
