# PDF AI Summarizer — Walkthrough

A guided tour of the code, for anyone reading it for the first time.

## 1. What the app does

You upload a PDF. The app pulls out the text, splits it into chunks that fit a
model's context window, summarizes each chunk, and then merges those chunk
summaries into one coherent summary. Along the way it can also describe the
document's structure and pull out verbatim quotes.

That two-stage pattern is called **map-reduce**:

```
PDF → extract → clean → chunk (by tokens)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         summary 1       summary 2   …   summary N        ← map
              └───────────────┼───────────────┘
                              ▼
                     combine → final summary             ← reduce
```

Why not just send the whole PDF to Gemini? Three reasons:

1. **Context limits.** A 300-page PDF does not fit, and even when it does, very
   long single requests degrade summary quality.
2. **Cost and latency.** You pay per token. Chunking lets the app summarize
   chunks **in parallel** (4 at a time by default) rather than serially.
3. **Fault tolerance.** If one chunk fails, the other 19 still produce a usable
   summary. The failure is reported per chunk instead of losing the whole run.

## 2. Where things live

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI: sidebar settings, upload, pipeline run, results, downloads |
| `pdf_processor.py` | Reading the PDF, cleaning text, counting tokens, splitting into chunks |
| `prompts.py` | Every LLM instruction: system persona, summary styles, four templates |
| `summarizer.py` | The Gemini calls and the map-reduce chains |
| `utils.py` | Display helpers, text/CSV export, time estimate, API key check |
| `scripts/check_env.py` | Environment gate: validates `.env`, the key, and the LangChain path |
| `scripts/install_xxhash_shim.py` | Windows-only workaround for Smart App Control (see §7) |
| `tests/` | pytest suite, including a tiny PDF builder (`pdf_factory.py`) |
| `samples/sample.pdf` | A three-page report used for manual smoke tests |

## 3. How a PDF flows through the code

1. **Upload** — `st.file_uploader` in `app.py` returns a file-like object.
2. **Metadata** — `PDFProcessor.get_pdf_metadata()` reads page count, title,
   author and subject with `pypdf`. No LLM involved.
3. **Extract** — `extract_text_from_pdf()` calls `page.extract_text()` per page.
   A page that raises is logged and skipped. A file that cannot be opened at all
   raises `RuntimeError`.
4. **Clean** — `clean_text()` strips control characters, collapses runs of
   spaces and tabs, and limits blank lines to one. Punctuation and paragraph
   breaks survive because they carry meaning.
5. **Count** — `count_tokens()` uses tiktoken's `cl100k_base`. Gemini has its own
   tokenizer, so this is an approximation; it is only used for chunk sizing.
6. **Chunk** — `chunk_text()` splits on sentence boundaries, then greedily packs
   sentences into chunks up to `max_tokens`. A single sentence longer than the
   budget is hard-split as a last resort.
7. **Map** — `PDFSummarizer.summarize_chunks()` builds one prompt per chunk and
   calls `chunk_chain.batch(...)` with `max_concurrency=4` and
   `return_exceptions=True`. Each result is recorded with its own `ok` flag.
8. **Reduce** — `combine_summaries()` merges the successful summaries. If there
   are more than `REDUCE_FACTOR` (8) of them, it merges in batches and recurses,
   so the hierarchy always shrinks and always terminates.
9. **Extras** — `analyze_document()` and `extract_quotes()` run single-shot over
   the cleaned text, truncated to `ANALYSIS_CHAR_LIMIT` characters.
10. **Display** — `utils.format_summary_display()` renders metrics, the final
    summary and a per-chunk expander. Downloads produce a `.txt` report and a
    `.csv` of per-chunk statistics.

## 4. Why chains are written as `prompt | llm | parser`

LangChain's LCEL pipe operator composes `Runnable`s. `CHUNK_SUMMARY | llm` gives
a runnable that takes a dict of variables and returns an `AIMessage`. That
message is awkward to work with: Gemini 3 returns content as a **list of blocks**,
so `.content` is not always a plain string.

`message_to_text()` normalizes all of it — string content, block lists, dicts,
and older `.text()` method variants — into a `str`. Wrapping it in
`RunnableLambda` makes it a chain step, so every chain ends with plain text.

## 5. Failure handling

| Failure | Where it is handled | What the user sees |
|---|---|---|
| Corrupt PDF | `extract_text_from_pdf` raises `RuntimeError` | Red error, no crash |
| Scanned PDF, no text | `chunk_text` raises `ValueError` | "No extractable text" |
| One chunk fails (429, 500) | `return_exceptions=True` in `batch` | Other chunks still summarized |
| Transient 503 from Gemini | `max_retries=3` on the chat model | Transparent retry |
| Missing API key | `validate_api_key()` in `utils.py` | Setup instructions |
| Every chunk fails | `combine_summaries` returns a sentinel string | "No valid summaries were generated." |

## 6. Tests

```powershell
pytest          # 42 tests, ~1s, no network
ruff check .
```

`tests/pdf_factory.py` builds real PDFs from raw PDF syntax, so extraction is
tested against actual bytes rather than a mocked `PdfReader`.
`PDFSummarizer(llm=FakeLLM())` injects a fake chat model, so chain logic is
tested without spending tokens.

## 7. The xxhash shim (Windows only)

On Windows with **Smart App Control** enabled, the unsigned `_xxhash.pyd` inside
the `xxhash` wheel is blocked by policy:

```
ImportError: DLL load failed while importing _xxhash:
An Application Control policy has blocked this file.
```

LangChain imports xxhash transitively
(`langchain_core → langsmith → _uuid`), and langsmith uses only `xxh3_128`, for
deterministic tracing IDs. There is no tracing in this app, so
`scripts/install_xxhash_shim.py` swaps in the pure-Python `shim/xxhash.py`.

If you are on macOS, Linux, or Windows with the policy off, you never need this
and `python scripts/check_env.py` will show the LangChain check passing directly.
