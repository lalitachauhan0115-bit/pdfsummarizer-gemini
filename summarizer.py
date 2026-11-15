"""Gemini calls: chunk summaries, combine (reduce), analysis and quotes.

Chains are plain LCEL compositions: ``prompt | llm | parser``. The parser turns
an ``AIMessage`` into plain text, because Gemini 3 returns content as a list of
blocks and ``.content`` is not always a string.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, style_instruction

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-flash-lite"
# How many chunk summaries are merged per reduce step. Fixed size guarantees the
# hierarchy shrinks on every pass, so the recursion always terminates.
REDUCE_FACTOR = 8
# Characters sent to a single-shot analysis/quote call. Well under Gemini's
# context window, and keeps a stray 900-page PDF from blowing up a request.
ANALYSIS_CHAR_LIMIT = 300_000

# Bullets and numbering that models add despite being told not to.
_QUOTE_MARKER = re.compile(r"^\s*(?:[-*•]\s*|\d+[.)]\s*)")
_QUOTE_CHARS = "\"'“”‘’"


def message_to_text(message: Any) -> str:
    """Best-effort extraction of text from a chat message."""
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):  # older langchain exposed .text() as a method
        try:
            value = text()
            if isinstance(value, str):
                return value
        except Exception:  # noqa: BLE001
            pass

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        return "".join(parts)
    return str(content)


_PARSER = RunnableLambda(message_to_text)


class PDFSummarizer:
    """Wraps a Gemini chat model in the four chains the app needs."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.3,
        max_retries: int = 3,
        max_concurrency: int = 4,
        llm: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self.max_concurrency = max_concurrency

        if llm is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            llm = ChatGoogleGenerativeAI(
                model=self.model,
                temperature=temperature,
                google_api_key=api_key,
                max_retries=max_retries,
            )
        # Injectable so tests can swap in a fake chat model.
        self.llm = llm
        self.chunk_chain = CHUNK_SUMMARY | self.llm | _PARSER
        self.combine_chain = COMBINE | self.llm | _PARSER
        self.analysis_chain = ANALYSIS | self.llm | _PARSER
        self.quotes_chain = QUOTES | self.llm | _PARSER

    # ------------------------------------------------------------ single call

    def summarize_text(self, text: str, summary_type: str = "detailed") -> str:
        return self.chunk_chain.invoke(
            {"style": style_instruction(summary_type), "text": text}
        )

    # ---------------------------------------------------------- map (chunks)

    def summarize_chunks(self, chunks: list[str], summary_type: str = "detailed") -> dict:
        """Summarize every chunk, then reduce those into one summary.

        A failing chunk is recorded as an error instead of killing the run, so a
        rate-limited or malformed chunk still leaves a usable document summary.
        """
        if not chunks:
            raise ValueError("No chunks to summarize")

        style = style_instruction(summary_type)
        inputs = [{"style": style, "text": chunk} for chunk in chunks]
        results = self.chunk_chain.batch(
            inputs,
            config={"max_concurrency": self.max_concurrency},
            return_exceptions=True,
        )

        summaries: list[dict] = []
        for number, (chunk, result) in enumerate(zip(chunks, results, strict=False), start=1):
            ok = not isinstance(result, Exception)
            if not ok:
                logger.warning("Chunk %d failed: %s", number, result)
            text = result if ok else f"Error processing chunk: {result}"
            summaries.append(
                {
                    "chunk_number": number,
                    "summary": text,
                    "ok": ok,
                    "original_length": len(chunk),
                    "summary_length": len(text) if ok else 0,
                }
            )

        return {
            "individual_summaries": summaries,
            "combined_summary": self.combine_summaries(summaries, summary_type),
            "total_chunks": len(chunks),
            "summary_type": summary_type,
            "model": self.model,
        }

    # -------------------------------------------------------- reduce (combine)

    def combine_summaries(self, chunk_summaries: list[dict], summary_type: str) -> str:
        valid = [item["summary"] for item in chunk_summaries if item.get("ok")]
        if not valid:
            return "No valid summaries were generated."
        if len(valid) == 1:
            return valid[0]
        return self._reduce(valid, style_instruction(summary_type))

    def _reduce(self, summaries: list[str], style: str) -> str:
        if len(summaries) == 1:
            return summaries[0]
        merged = [
            self.combine_chain.invoke(
                {"style": style, "summaries": "\n\n".join(summaries[i : i + REDUCE_FACTOR])}
            )
            for i in range(0, len(summaries), REDUCE_FACTOR)
        ]
        return self._reduce(merged, style)

    # ------------------------------------------------------------- extras

    def analyze_document(self, text: str) -> str:
        return self.analysis_chain.invoke({"text": text[:ANALYSIS_CHAR_LIMIT]})

    def extract_quotes(self, text: str) -> list[str]:
        raw = self.quotes_chain.invoke({"text": text[:ANALYSIS_CHAR_LIMIT]})
        quotes: list[str] = []
        for line in raw.splitlines():
            cleaned = _QUOTE_MARKER.sub("", line).strip()
            cleaned = cleaned.strip(_QUOTE_CHARS).strip()
            if cleaned:
                quotes.append(cleaned)
        return quotes


def _main() -> int:
    """CLI: summarize a PDF end to end with no UI."""
    import argparse
    from pathlib import Path

    from pdf_processor import PDFProcessor

    parser = argparse.ArgumentParser(description="Summarize a PDF with Gemini")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--style", default="detailed", choices=["brief", "detailed", "bullet_points", "executive"])
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--quotes", action="store_true", help="also extract key quotes")
    parser.add_argument("--analysis", action="store_true", help="also analyze structure")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    processor = PDFProcessor()
    with args.pdf.open("rb") as handle:
        processed = processor.process(handle, max_tokens=args.max_tokens)

    print(f"{args.pdf.name}: {processed['num_chunks']} chunk(s), {processed['total_tokens']} tokens")

    summarizer = PDFSummarizer()
    result = summarizer.summarize_chunks(processed["chunks"], args.style)

    print("\n=== FINAL SUMMARY ===\n")
    print(result["combined_summary"])

    if args.analysis:
        print("\n=== ANALYSIS ===\n")
        print(summarizer.analyze_document(processed["text"]))
    if args.quotes:
        print("\n=== QUOTES ===\n")
        for quote in summarizer.extract_quotes(processed["text"]):
            print(f"- {quote}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
