"""Presentation and export helpers for the Streamlit UI."""

from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st


def validate_api_key() -> bool:
    """Stop the app early if the key is missing, with instructions."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return True
    st.error(
        "**GEMINI_API_KEY not found.**\n\n"
        "Create a `.env` file next to `app.py` containing:\n\n"
        "```\nGEMINI_API_KEY=your_key_here\n```\n\n"
        "Get a free key at https://aistudio.google.com/apikey"
    )
    return False


def format_summary_display(summary_data: dict) -> None:
    """Render statistics, the final summary and the per-chunk summaries."""
    st.subheader("📊 Summary Statistics")

    successful = sum(1 for item in summary_data["individual_summaries"] if item["ok"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", summary_data["total_chunks"])
    col2.metric("Successful Summaries", successful)
    col3.metric("Summary Type", summary_data["summary_type"].replace("_", " ").title())

    st.subheader("📝 Final Summary")
    st.write(summary_data["combined_summary"])

    with st.expander("View individual chunk summaries"):
        for item in summary_data["individual_summaries"]:
            label = f"**Chunk {item['chunk_number']}**"
            if not item["ok"]:
                label += " ⚠ failed"
            st.write(label)
            st.write(item["summary"])
            st.write(
                f"*Original length: {item['original_length']} chars, "
                f"summary length: {item['summary_length']} chars*"
            )
            st.divider()


def export_summary_to_text(summary_data: dict, filename: str = "") -> str:
    """Build the downloadable plain-text report."""
    lines = ["PDF Summary Report", "==================", ""]
    if filename:
        lines.append(f"File: {filename}")
    lines += [
        f"Model: {summary_data.get('model', '-')}",
        f"Summary Type: {summary_data['summary_type'].replace('_', ' ').title()}",
        f"Total Chunks Processed: {summary_data['total_chunks']}",
        f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "FINAL SUMMARY",
        "=============",
        "",
        summary_data["combined_summary"],
        "",
        "INDIVIDUAL CHUNK SUMMARIES",
        "==========================",
        "",
    ]
    for item in summary_data["individual_summaries"]:
        suffix = "" if item["ok"] else " (FAILED)"
        lines += [f"Chunk {item['chunk_number']}{suffix}:", item["summary"], ""]
    return "\n".join(lines)


def create_summary_dataframe(summary_data: dict) -> pd.DataFrame:
    """One row per chunk with lengths, compression ratio and status."""
    rows = []
    for item in summary_data["individual_summaries"]:
        original = item["original_length"]
        rows.append(
            {
                "Chunk": item["chunk_number"],
                "Original Length": original,
                "Summary Length": item["summary_length"],
                "Compression Ratio": (
                    round(item["summary_length"] / original, 2) if original else 0
                ),
                "Status": "Success" if item["ok"] else "Error",
            }
        )
    return pd.DataFrame(rows)


def estimate_processing_time(num_chunks: int) -> str:
    """Rough wall-clock estimate: parallel chunks plus one combine call."""
    seconds = num_chunks * 4 + 5
    if seconds < 60:
        return f"~{seconds} seconds"
    return f"~{max(1, round(seconds / 60))} minute(s)"
