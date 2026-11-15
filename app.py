"""Streamlit UI: settings, upload, pipeline run, results and downloads."""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from pdf_processor import PDFProcessor
from summarizer import PDFSummarizer
from utils import (
    create_summary_dataframe,
    estimate_processing_time,
    export_summary_to_text,
    format_summary_display,
    validate_api_key,
)

load_dotenv()

MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
SUMMARY_TYPES = ["brief", "detailed", "bullet_points", "executive"]
MAX_UPLOAD_MB = 50

st.set_page_config(page_title="PDF AI Summarizer", page_icon="📄", layout="wide")


def sidebar_settings() -> dict:
    """Everything the user can tune, collected in one place."""
    st.sidebar.header("⚙️ Settings")

    default_model = os.getenv("GEMINI_MODEL", MODELS[0])
    models = MODELS if default_model in MODELS else [default_model, *MODELS]
    model = st.sidebar.selectbox(
        "Model",
        models,
        index=models.index(default_model),
        help="Gemini model used for every LLM call",
    )

    summary_type = st.sidebar.selectbox(
        "Summary type",
        SUMMARY_TYPES,
        index=SUMMARY_TYPES.index("detailed"),
        format_func=lambda name: name.replace("_", " ").title(),
        help="Shape of the final summary",
    )

    max_tokens = st.sidebar.slider(
        "Max tokens per chunk",
        4000,
        10000,
        8000,
        step=500,
        help="Smaller chunks mean more calls but better fidelity on long documents",
    )

    show_analysis = st.sidebar.checkbox(
        "Show document analysis", value=True, help="Document type, themes, audience, purpose"
    )
    show_quotes = st.sidebar.checkbox(
        "Extract key quotes", value=False, help="5-10 verbatim quotes from the document"
    )

    return {
        "model": model,
        "summary_type": summary_type,
        "max_tokens": max_tokens,
        "show_analysis": show_analysis,
        "show_quotes": show_quotes,
    }


def get_summarizer(model: str) -> PDFSummarizer:
    """Cache the summarizer in session state, rebuilding it when the model changes."""
    if st.session_state.get("summarizer_model") != model:
        st.session_state.summarizer = PDFSummarizer(model)
        st.session_state.summarizer_model = model
    return st.session_state.summarizer


def show_metadata(processor: PDFProcessor, uploaded_file) -> None:
    try:
        meta = processor.get_pdf_metadata(uploaded_file)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read PDF metadata: {exc}")
        return
    cols = st.columns(4)
    cols[0].metric("Pages", meta["pages"])
    cols[1].metric("Title", meta["title"] or "—")
    cols[2].metric("Author", meta["author"] or "—")
    cols[3].metric("Subject", meta["subject"] or "—")


def run_pipeline(processor: PDFProcessor, uploaded_file, settings: dict) -> dict:
    """Extract, chunk, summarize and optionally analyse. Returns everything for display."""
    with st.spinner("Extracting and chunking text…"):
        processed = processor.process(uploaded_file, max_tokens=settings["max_tokens"])

    st.info(
        f"{processed['num_chunks']} chunk(s), {processed['total_tokens']:,} tokens — "
        f"{estimate_processing_time(processed['num_chunks'])}"
    )

    summarizer = get_summarizer(settings["model"])
    with st.spinner(
        f"Summarizing {processed['num_chunks']} chunk(s) with {settings['model']}…"
    ):
        result = summarizer.summarize_chunks(processed["chunks"], settings["summary_type"])
        result["analysis"] = (
            summarizer.analyze_document(processed["text"]) if settings["show_analysis"] else None
        )
        result["quotes"] = (
            summarizer.extract_quotes(processed["text"]) if settings["show_quotes"] else None
        )

    result["filename"] = uploaded_file.name
    return result


def show_results(results: dict, settings: dict) -> None:
    format_summary_display(results)

    if results.get("analysis"):
        st.subheader("🔍 Document Analysis")
        st.write(results["analysis"])

    if results.get("quotes"):
        st.subheader("💬 Key Quotes")
        for quote in results["quotes"]:
            st.markdown(f"> {quote}")

    st.subheader("⬇️ Downloads")
    col1, col2 = st.columns(2)
    stem = os.path.splitext(results.get("filename", "summary"))[0]

    col1.download_button(
        "Download summary (.txt)",
        data=export_summary_to_text(results, results.get("filename", "")),
        file_name=f"{stem}_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )
    col2.download_button(
        "Download chunk stats (.csv)",
        data=create_summary_dataframe(results).to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_chunks.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Per-chunk statistics"):
        st.dataframe(create_summary_dataframe(results), use_container_width=True)

    if not settings["show_analysis"] and not settings["show_quotes"]:
        st.caption("Tip: enable analysis or quotes in the sidebar for more detail.")


def main() -> None:
    st.title("📄 PDF Reader + AI Summarizer")
    st.markdown(
        "Upload a PDF and get a structured summary powered by **Google Gemini** via LangChain."
    )

    if not validate_api_key():
        st.stop()

    settings = sidebar_settings()

    if "pdf_processor" not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    processor: PDFProcessor = st.session_state.pdf_processor

    uploaded_file = st.file_uploader(
        "Choose a PDF file", type="pdf", help=f"Up to {MAX_UPLOAD_MB} MB"
    )

    # Drop stale results whenever the uploaded file changes or is removed.
    file_id = uploaded_file.file_id if uploaded_file else None
    if st.session_state.get("file_id") != file_id:
        st.session_state.pop("results", None)
        st.session_state.file_id = file_id

    if uploaded_file is None:
        st.info("Upload a PDF to begin.")
        return

    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"That file is larger than {MAX_UPLOAD_MB} MB. Try a smaller document.")
        return

    show_metadata(processor, uploaded_file)

    if st.button("🚀 Process PDF", type="primary"):
        try:
            st.session_state.results = run_pipeline(processor, uploaded_file, settings)
        except ValueError as exc:
            st.error(f"{exc}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong: {exc}")
            return

    if "results" in st.session_state:
        st.divider()
        show_results(st.session_state.results, settings)


if __name__ == "__main__":
    main()
