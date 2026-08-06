"""Unit tests for rag/ — loader, splitter, vector store."""

import pytest
from pathlib import Path
from rag.loader import load_file
from rag.splitter import recursive_split


class TestLoader:
    def test_load_txt(self):
        path = Path("logs/_test_pytest.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Hello world.", encoding="utf-8")

        docs = load_file(str(path))
        assert len(docs) == 1
        assert docs[0]["text"] == "Hello world."
        assert docs[0]["metadata"]["source"] == str(path)

        path.unlink()

    def test_load_md(self):
        path = Path("logs/_test_pytest.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Title\n\nBody text.", encoding="utf-8")

        docs = load_file(str(path))
        assert len(docs) == 1
        assert "Title" in docs[0]["text"]

        path.unlink()

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file("test.xyz")


class TestSplitter:
    def test_basic_split(self):
        text = "This is a test. " * 50
        chunks = recursive_split(text, chunk_size=200, chunk_overlap=30)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 200 + 30  # allow overlap

    def test_short_text_no_split(self):
        text = "Short text."
        chunks = recursive_split(text, chunk_size=800, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_paragraph_split(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = recursive_split(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) >= 1

    def test_chinese_text(self):
        text = "第一段内容。\n\n第二段内容。" * 20
        chunks = recursive_split(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_overlap(self):
        text = "ABCDEFGHIJ" * 20
        chunks = recursive_split(text, chunk_size=50, chunk_overlap=10)
        if len(chunks) >= 2:
            # Check that consecutive chunks have overlap
            end_of_first = chunks[0][-10:]
            start_of_second = chunks[1][:10]
            assert end_of_first == start_of_second
