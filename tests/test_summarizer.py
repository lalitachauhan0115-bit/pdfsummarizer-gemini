"""Phase 3 tests: chains, map-reduce and error handling, with a fake LLM.

No network calls: PDFSummarizer accepts an injected chat model.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from summarizer import PDFSummarizer, message_to_text


class FakeLLM(Runnable):
    """Echoes a deterministic summary; raises when the prompt contains FAIL."""

    def invoke(self, input, config=None, **kwargs):  # noqa: A002 - LCEL signature
        text = input.to_string() if hasattr(input, "to_string") else str(input)
        if "FAIL" in text:
            raise RuntimeError("simulated provider error")
        return AIMessage(content=f"SUMMARY[{len(text)}]")


class ListLLM(Runnable):
    """Returns a scripted string, for testing parsing and quote splitting."""

    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, input, config=None, **kwargs):  # noqa: A002
        return AIMessage(content=self.response)


@pytest.fixture
def summarizer() -> PDFSummarizer:
    return PDFSummarizer(llm=FakeLLM())


# --------------------------------------------------------------- parsing


def test_message_to_text_plain_string():
    assert message_to_text(AIMessage(content="hello")) == "hello"


def test_message_to_text_block_list():
    message = AIMessage(content=[{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}])
    assert message_to_text(message) == "part one part two"


def test_message_to_text_falls_back_to_str():
    assert "abc" in message_to_text("abc")


# ---------------------------------------------------------------- chains


def test_summarize_text_returns_parsed_string(summarizer):
    result = summarizer.summarize_text("some document text", "brief")
    assert result.startswith("SUMMARY[")


def test_summarize_chunks_records_every_chunk(summarizer):
    result = summarizer.summarize_chunks(["alpha", "beta", "gamma"], "brief")
    assert result["total_chunks"] == 3
    assert len(result["individual_summaries"]) == 3
    assert all(item["ok"] for item in result["individual_summaries"])
    assert [item["chunk_number"] for item in result["individual_summaries"]] == [1, 2, 3]
    assert result["combined_summary"]


def test_summarize_chunks_isolates_a_failure():
    result = PDFSummarizer(llm=FakeLLM()).summarize_chunks(["good", "FAIL this one"], "brief")
    failed = [item for item in result["individual_summaries"] if not item["ok"]]
    assert len(failed) == 1
    assert failed[0]["summary_length"] == 0
    assert "Error processing chunk" in failed[0]["summary"]
    # The surviving chunk must still drive the final summary.
    assert result["combined_summary"]


def test_summarize_chunks_rejects_empty_input(summarizer):
    with pytest.raises(ValueError, match="No chunks"):
        summarizer.summarize_chunks([], "brief")


def test_combine_returns_single_summary_unchanged(summarizer):
    items = [{"summary": "only one", "ok": True}]
    assert summarizer.combine_summaries(items, "brief") == "only one"


def test_combine_reports_when_all_chunks_failed(summarizer):
    items = [{"summary": "err", "ok": False}, {"summary": "err", "ok": False}]
    assert summarizer.combine_summaries(items, "brief") == "No valid summaries were generated."


def test_combine_reduces_hierarchically_beyond_one_batch(summarizer):
    items = [{"summary": f"summary {i}", "ok": True} for i in range(20)]
    result = summarizer.combine_summaries(items, "brief")
    assert result.startswith("SUMMARY[")  # went through at least one combine call


def test_extract_quotes_strips_markers():
    summarizer = PDFSummarizer(llm=ListLLM('"First quote"\n- Second quote\n\n* Third quote'))
    quotes = summarizer.extract_quotes("document")
    assert quotes == ["First quote", "Second quote", "Third quote"]


def test_extract_quotes_drops_blank_lines():
    summarizer = PDFSummarizer(llm=ListLLM("one\n\n\n two \n"))
    assert summarizer.extract_quotes("document") == ["one", "two"]


def test_analyze_document_returns_text():
    summarizer = PDFSummarizer(llm=ListLLM("analysis body"))
    assert summarizer.analyze_document("document") == "analysis body"


def test_analysis_input_is_truncated(monkeypatch):
    import summarizer as module

    monkeypatch.setattr(module, "ANALYSIS_CHAR_LIMIT", 10)
    fake = PDFSummarizer(llm=FakeLLM())
    result = fake.analyze_document("x" * 5000)
    prompt_length = int(result.removeprefix("SUMMARY[").rstrip("]"))
    # Template boilerplate is a few hundred chars; 5000 chars of document is not.
    assert prompt_length < 1000
