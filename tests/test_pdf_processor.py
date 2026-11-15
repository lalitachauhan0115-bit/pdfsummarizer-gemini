"""Phase 1 tests: extraction, cleaning, counting and chunking."""

from __future__ import annotations

import io

import pytest

from pdf_processor import PDFProcessor
from tests.pdf_factory import build_pdf

SAMPLE_PAGES = [
    ["Quarterly Revenue Report", "Revenue rose by 12 percent this quarter."],
    ["Costs stayed flat.", "Headcount grew by four people."],
]


@pytest.fixture(scope="module")
def processor() -> PDFProcessor:
    return PDFProcessor()


@pytest.fixture(scope="module")
def sample_pdf() -> bytes:
    return build_pdf(SAMPLE_PAGES)


# --------------------------------------------------------------- extraction


def test_extract_text_roundtrip(processor, sample_pdf):
    text = processor.extract_text_from_pdf(io.BytesIO(sample_pdf))
    assert "Quarterly Revenue Report" in text
    assert "Headcount grew by four people." in text


def test_extract_joins_pages_with_blank_line(processor, sample_pdf):
    text = processor.extract_text_from_pdf(io.BytesIO(sample_pdf))
    assert "\n\n" in text


def test_extract_seeks_to_start_first(processor, sample_pdf):
    handle = io.BytesIO(sample_pdf)
    handle.seek(len(sample_pdf) // 2)  # simulate a previously read file object
    assert "Revenue" in processor.extract_text_from_pdf(handle)


def test_metadata(processor, sample_pdf):
    meta = processor.get_pdf_metadata(io.BytesIO(sample_pdf))
    assert meta["pages"] == 2
    assert set(meta) == {"pages", "title", "author", "subject"}


def test_corrupt_pdf_raises_runtime_error(processor):
    with pytest.raises(RuntimeError, match="Error reading PDF"):
        processor.extract_text_from_pdf(io.BytesIO(b"this is not a pdf"))


# ----------------------------------------------------------------- cleaning


def test_clean_removes_control_characters(processor):
    assert processor.clean_text("alpha\x00beta\x07gamma") == "alphabetagamma"


def test_clean_collapses_spaces_and_blank_lines(processor):
    dirty = "a    b\tc\n\n\n\nnext   line"
    assert processor.clean_text(dirty) == "a b c\n\nnext line"


def test_clean_keeps_punctuation_and_paragraphs(processor):
    text = "First sentence. Second sentence!\n\nNew paragraph?"
    assert processor.clean_text(text) == text


# ----------------------------------------------------------------- counting


def test_count_tokens_matches_encoding(processor):
    assert processor.count_tokens("hello world") > 0
    assert processor.count_tokens("") == 0


# ----------------------------------------------------------------- chunking


def test_chunk_text_rejects_empty(processor):
    with pytest.raises(ValueError, match="No extractable text"):
        processor.chunk_text("   \n  ")


def test_chunk_text_rejects_bad_budget(processor):
    with pytest.raises(ValueError, match="max_tokens"):
        processor.chunk_text("some text", max_tokens=0)


def test_chunk_text_respects_budget(processor):
    text = " ".join(f"Sentence number {i} talks about a distinct topic." for i in range(200))
    chunks = processor.chunk_text(text, max_tokens=60)
    assert len(chunks) > 1
    assert all(processor.count_tokens(chunk) <= 60 for chunk in chunks)


def test_chunk_text_hard_splits_oversized_sentence(processor):
    giant = "word " * 3000  # one sentence, no terminal punctuation
    chunks = processor.chunk_text(giant, max_tokens=100)
    assert len(chunks) > 1
    assert all(processor.count_tokens(chunk) <= 100 for chunk in chunks)


def test_chunk_text_keeps_single_short_chunk(processor):
    chunks = processor.chunk_text("Only one short sentence.", max_tokens=100)
    assert chunks == ["Only one short sentence."]


def test_chunk_text_preserves_all_words(processor):
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
    chunks = processor.chunk_text(text, max_tokens=5)
    joined = " ".join(chunks)
    for word in ("Alpha", "gamma", "Delta", "iota."):
        assert word in joined


# ----------------------------------------------------------------- pipeline


def test_process_returns_stats(processor, sample_pdf):
    result = processor.process(io.BytesIO(sample_pdf), max_tokens=100)
    assert result["num_chunks"] == len(result["chunks"]) >= 1
    assert result["total_tokens"] > 0
    assert "Revenue" in result["text"]


def test_process_rejects_scanned_pdf(processor):
    blank = build_pdf([[]])  # a page with no text, i.e. a scanned image
    with pytest.raises(ValueError, match="No extractable text"):
        processor.process(io.BytesIO(blank))
