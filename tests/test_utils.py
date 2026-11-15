"""Tests for the display/export helpers and the secrets bridge."""

from __future__ import annotations

import os

import pandas as pd
import pytest

import utils


@pytest.fixture
def summary_data() -> dict:
    return {
        "individual_summaries": [
            {
                "chunk_number": 1,
                "summary": "first summary",
                "ok": True,
                "original_length": 200,
                "summary_length": 50,
            },
            {
                "chunk_number": 2,
                "summary": "Error processing chunk: boom",
                "ok": False,
                "original_length": 100,
                "summary_length": 0,
            },
        ],
        "combined_summary": "the final summary",
        "total_chunks": 2,
        "summary_type": "bullet_points",
        "model": "gemini-3.1-flash-lite",
    }


# ----------------------------------------------------------------- exports


def test_text_export_contains_report_sections(summary_data):
    report = utils.export_summary_to_text(summary_data, "report.pdf")
    assert "PDF Summary Report" in report
    assert "File: report.pdf" in report
    assert "the final summary" in report
    assert "Chunk 1:" in report
    assert "Chunk 2 (FAILED):" in report


def test_text_export_omits_filename_when_blank(summary_data):
    assert "File:" not in utils.export_summary_to_text(summary_data)


def test_dataframe_has_one_row_per_chunk(summary_data):
    frame = utils.create_summary_dataframe(summary_data)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 2
    assert list(frame["Status"]) == ["Success", "Error"]


def test_dataframe_compression_ratio(summary_data):
    frame = utils.create_summary_dataframe(summary_data)
    assert frame.loc[0, "Compression Ratio"] == 0.25
    # A failed chunk has zero summary length and must not divide by zero.
    assert frame.loc[1, "Compression Ratio"] == 0


def test_time_estimate_scales_with_chunks():
    assert "seconds" in utils.estimate_processing_time(1)
    assert "minute" in utils.estimate_processing_time(30)


# ------------------------------------------------------ secrets bridge


def test_bridge_copies_streamlit_secrets(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(
        utils.st,
        "secrets",
        {"GEMINI_API_KEY": "from-secrets", "GEMINI_MODEL": "gemini-x"},
    )

    utils.bridge_streamlit_secrets()

    assert os.environ["GEMINI_API_KEY"] == "from-secrets"
    assert os.environ["GEMINI_MODEL"] == "gemini-x"


def test_bridge_does_not_override_existing_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    monkeypatch.setattr(utils.st, "secrets", {"GEMINI_API_KEY": "from-secrets"})

    utils.bridge_streamlit_secrets()

    assert os.environ["GEMINI_API_KEY"] == "from-env"


def test_bridge_survives_missing_secrets(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class Exploding:
        def get(self, key):
            raise FileNotFoundError("no secrets.toml here")

    monkeypatch.setattr(utils.st, "secrets", Exploding())

    utils.bridge_streamlit_secrets()  # must not raise

    assert not os.getenv("GEMINI_API_KEY")
