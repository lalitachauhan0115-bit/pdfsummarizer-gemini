"""Phase 2 tests: the prompt layer."""

from __future__ import annotations

import pytest

from prompts import (
    ANALYSIS,
    CHUNK_SUMMARY,
    COMBINE,
    QUOTES,
    STYLES,
    SYSTEM,
    style_instruction,
)


def test_all_styles_have_instructions():
    assert set(STYLES) == {"brief", "detailed", "bullet_points", "executive"}
    assert all(isinstance(text, str) and text.strip() for text in STYLES.values())


def test_style_instruction_falls_back_to_detailed():
    assert style_instruction("nonsense") == STYLES["detailed"]


def test_style_instruction_returns_requested_style():
    assert style_instruction("brief") == STYLES["brief"]


def test_chunk_summary_renders_with_system_and_vars():
    messages = CHUNK_SUMMARY.format_messages(style="brief", text="Body text here")
    assert messages[0].content == SYSTEM
    assert "brief" in messages[1].content
    assert "Body text here" in messages[1].content


def test_combine_renders_summaries():
    messages = COMBINE.format_messages(style="brief", summaries="one\ntwo")
    assert "one\ntwo" in messages[1].content


def test_analysis_renders_and_asks_for_five_parts():
    text = ANALYSIS.format_messages(text="doc")[1].content
    for expected in ("Document Type", "Main Topics", "Target Audience", "Overall Purpose"):
        assert expected in text


def test_quotes_forbids_numbering():
    text = QUOTES.format_messages(text="doc")[1].content
    assert "verbatim" in text
    assert "no numbering" in text.lower()


@pytest.mark.parametrize("template", [CHUNK_SUMMARY, COMBINE, ANALYSIS, QUOTES])
def test_templates_have_no_unescaped_braces(template):
    """A stray brace would make .format_messages raise on every call."""
    variables = set(template.input_variables)
    assert variables <= {"style", "text", "summaries"}
