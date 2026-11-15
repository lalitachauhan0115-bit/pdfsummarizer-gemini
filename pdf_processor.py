"""PDF text extraction, cleaning, token counting and chunking.

The chunker splits on sentence boundaries so that each chunk handed to the LLM
is self-contained English prose rather than a mid-sentence fragment. A single
sentence longer than the token budget is hard-split as a last resort.
"""

from __future__ import annotations

import logging
import re
from typing import BinaryIO

import tiktoken
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Control characters that carry no meaning in extracted prose.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


class PDFProcessor:
    """Turns an uploaded PDF into a list of token-bounded chunks."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        # cl100k_base is an approximation for Gemini's own tokenizer. It is only
        # used for chunk sizing, so a close-enough count is fine.
        self.encoding = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------------------------ read

    def extract_text_from_pdf(self, pdf_file: BinaryIO) -> str:
        """Return the text of every page, joined by blank lines.

        A page that fails to extract is skipped and logged rather than
        aborting the whole document.
        """
        try:
            pdf_file.seek(0)
            reader = PdfReader(pdf_file)
            pages: list[str] = []
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    pages.append(page.extract_text() or "")
                except Exception as exc:  # noqa: BLE001 - a bad page must not kill the run
                    logger.warning("Skipping page %d: %s", page_num, exc)
            return "\n\n".join(pages)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Error reading PDF: {exc}") from exc

    def get_pdf_metadata(self, pdf_file: BinaryIO) -> dict:
        """Page count and document info, without touching the LLM."""
        pdf_file.seek(0)
        reader = PdfReader(pdf_file)
        info = reader.metadata or {}

        def _field(name: str) -> str:
            value = info.get(name) or ""
            return str(value).strip()

        return {
            "pages": len(reader.pages),
            "title": _field("/Title"),
            "author": _field("/Author"),
            "subject": _field("/Subject"),
        }

    # ----------------------------------------------------------------- clean

    def clean_text(self, text: str) -> str:
        """Strip control characters and normalise whitespace, keeping paragraphs."""
        text = _CONTROL_CHARS.sub("", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ----------------------------------------------------------------- count

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    # ----------------------------------------------------------------- chunk

    def chunk_text(self, text: str, max_tokens: int = 8000) -> list[str]:
        """Split text into chunks of at most ``max_tokens`` tokens."""
        if not text.strip():
            raise ValueError("No extractable text - the PDF may be scanned images")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        # (piece, token_count) pairs, split on sentence boundaries.
        pieces: list[tuple[str, int]] = []
        for sentence in _SENTENCE_BREAK.split(text.strip()):
            tokens = self.encoding.encode(sentence, disallowed_special=())
            if len(tokens) > max_tokens:
                for start in range(0, len(tokens), max_tokens):
                    window = tokens[start : start + max_tokens]
                    pieces.append((self.encoding.decode(window), len(window)))
            else:
                pieces.append((sentence, len(tokens)))

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for piece, token_count in pieces:
            # +1 accounts for the single space used to rejoin pieces.
            if current and current_tokens + token_count + 1 > max_tokens:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            current.append(piece)
            current_tokens += token_count + 1
        if current:
            chunks.append(" ".join(current))
        return chunks

    # ------------------------------------------------------------- pipeline

    def process(self, pdf_file: BinaryIO, max_tokens: int = 8000) -> dict:
        """Extract, clean and chunk in one call; returns text plus stats."""
        raw = self.extract_text_from_pdf(pdf_file)
        cleaned = self.clean_text(raw)
        chunks = self.chunk_text(cleaned, max_tokens=max_tokens)
        return {
            "text": cleaned,
            "chunks": chunks,
            "total_tokens": self.count_tokens(cleaned),
            "num_chunks": len(chunks),
        }


def _main() -> int:
    """CLI: dump chunk statistics for a PDF without calling any LLM."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Inspect how a PDF gets chunked")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--max-tokens", type=int, default=8000)
    args = parser.parse_args()

    processor = PDFProcessor()
    with args.pdf.open("rb") as handle:
        meta = processor.get_pdf_metadata(handle)
        result = processor.process(handle, max_tokens=args.max_tokens)

    print(f"file      : {args.pdf.name}")
    print(f"pages     : {meta['pages']}")
    print(f"title     : {meta['title'] or '-'}")
    print(f"chars     : {len(result['text'])}")
    print(f"tokens    : {result['total_tokens']}")
    print(f"chunks    : {result['num_chunks']}")
    for i, chunk in enumerate(result["chunks"], start=1):
        print(f"  chunk {i}: {len(chunk):>6} chars, {processor.count_tokens(chunk):>5} tokens")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
