"""Test the utils.
Run this test with command: pytest your_assistant/tests/core/test_indexer.py
"""
import argparse
import os
from unittest.mock import MagicMock

import pytest
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import your_assistant.core.indexer as indexer
import your_assistant.core.loader as loader
from your_assistant.core.utils import load_env


@pytest.fixture()
def setup():
    test_folder_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(test_folder_path)))
    args = argparse.Namespace()
    args.verbose = False
    args.embedding_tool_name = "openai"
    return root_path, args


class TestIndexer:
    @pytest.mark.parametrize(
        "db_path, config_file, expected",
        [
            (None, ".env.template", ValueError("db_path is not specified.")),
            ("test-faiss.db", ".env.template", type(None)),
        ],
    )
    def test_init_index_db(self, setup, db_path, config_file, expected):
        root_path, args = setup
        args.db_path = (
            os.path.join(os.path.dirname(__file__), db_path) if db_path else None
        )
        for key in os.environ:
            del os.environ[key]
        load_env(env_file_path=os.path.join(root_path, config_file))
        embeddings_tool = OpenAIEmbeddings()
        if isinstance(expected, ValueError):
            with pytest.raises(ValueError) as e:
                knowledge_indexer = indexer.KnowledgeIndexer(args=args)
                knowledge_indexer._init_index_db(
                    args=args, embeddings_tool=embeddings_tool
                )
            assert str(e.value) == expected.args[0]
        else:
            knowledge_indexer = indexer.KnowledgeIndexer(args=args)
            knowledge_indexer._init_index_db(args=args, embeddings_tool=embeddings_tool)
            assert type(knowledge_indexer.embeddings_db) == expected

    @pytest.mark.parametrize(
        "config_file, path, expected",
        [
            (
                ".env.template",
                "testdata/void.pdf",
                ValueError("File not found: void.pdf"),
            ),
            (
                ".env.template",
                "testdata/test-pdf.pdf",
                loader.PdfLoader,
            ),
            (
                ".env.template",
                "testdata/test.xyz",
                ValueError(
                    "File extension not supported: test.xyz. "
                    + "Only support ['.epub', '.html', '.mobi', '.pdf', '.txt'].",
                ),
            ),
        ],
    )
    def test_indexer_init_loader(self, setup, config_file, path, expected):
        root_path, args = setup
        args.db_path = "faiss.db"
        for key in os.environ:
            del os.environ[key]
        load_env(env_file_path=os.path.join(root_path, config_file))
        knowledge_indexer = indexer.KnowledgeIndexer(args=args)
        path = os.path.join(os.path.dirname(__file__), path)
        if isinstance(expected, ValueError):
            with pytest.raises(ValueError) as e:
                knowledge_indexer._init_loader(path=path)
            assert str(e.value) == expected.args[0]
        else:
            loader, source, _ = knowledge_indexer._init_loader(path=path)
            assert type(loader) == expected
            assert source == path

    @pytest.mark.parametrize(
        "config_file, path, chunk_size, chunk_overlap, expected",
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
        self, setup, config_file, path, chunk_size, chunk_overlap, expected
    ):
        root_path, args = setup
        args.db_path = "faiss.db"
        for key in os.environ:
            del os.environ[key]
        load_env(env_file_path=os.path.join(root_path, config_file))
        knowledge_indexer = indexer.KnowledgeIndexer(args=args)
        path = os.path.join(os.path.dirname(__file__), path)
        loader, source, _ = knowledge_indexer._init_loader(path=path)
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
            assert data[0].page_content.strip() == expected.strip()
            assert source == path
