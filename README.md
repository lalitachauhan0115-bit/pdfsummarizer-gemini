# PDF AI Summarizer

A Streamlit app that summarizes PDF documents with **Google Gemini** via
**LangChain**. Upload a PDF, pick a summary style, and get a map-reduce summary
plus optional structure analysis and key quotes.

- Map-reduce summarization: every chunk is summarized, then merged hierarchically
- Four summary styles: `brief`, `detailed`, `bullet_points`, `executive`
- Optional document analysis and verbatim key quotes
- Exports to `.txt` (full report) and `.csv` (per-chunk statistics)
- Runs without a UI too: `python summarizer.py report.pdf`

## How it works

```
PDF -> extract text -> clean -> chunk by tokens
                                      |
                    +-----------------+-----------------+
                    v                 v                 v
               summary 1         summary 2   ...   summary N     (map)
                    +-----------------+-----------------+
                                      v
                            combine -> final summary           (reduce)
```

Chunks are summarized four at a time. A chunk that fails is reported
individually instead of failing the whole document. When there are more chunk
summaries than fit a single reduce step, the merge recurses in batches.

## Project layout

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit UI: sidebar settings, upload, pipeline, results, downloads |
| `pdf_processor.py` | PDF text extraction, cleaning, token counting, chunking |
| `summarizer.py` | Gemini calls: chunk summaries, combine, analysis, quotes |
| `prompts.py` | All `ChatPromptTemplate`s and summary styles |
| `utils.py` | Display helpers, text/CSV export, statistics, API key check |
| `scripts/check_env.py` | Validates `.env`, the API key, the model and the LangChain path |
| `tests/` | pytest suite (no network required) |
| `samples/sample.pdf` | Three-page report for manual smoke tests |
| `.env.example` | Template for `GEMINI_API_KEY` and `GEMINI_MODEL` |

## Setup

Requires Python 3.11+ (developed on 3.12).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env    # then put your real key in .env
```

Get a free API key at <https://aistudio.google.com/apikey>, then verify
everything before running the app:

```powershell
python scripts/check_env.py
```

It checks that `.env` is readable, that the key is accepted, that the configured
model is available to that key, that a live `generateContent` call succeeds, and
that the LangChain client can complete a request.

## Run

```powershell
streamlit run app.py                       # web UI
python summarizer.py samples/sample.pdf --style detailed --analysis --quotes
python pdf_processor.py samples/sample.pdf --max-tokens 2000   # chunk stats only
```

## Tests

```powershell
pip install -r requirements-dev.txt
pytest
ruff check .
```

The suite runs in about a second with no network access: the PDF builder writes
real PDF bytes, and the summarizer accepts an injected fake chat model.

## Troubleshooting

- **No extractable text** — the PDF is probably scanned images; run OCR first.
- **429 / rate limit** — wait a minute, use a smaller document, or switch models.
  `max_retries=3` is already set on the client.
- **Model not available / 404** — set `GEMINI_MODEL` to a model your key can
  access; `python scripts/check_env.py` lists what is available.
- **`DLL load failed while importing _xxhash`** (Windows) — Smart App Control is
  blocking an unsigned dependency. Run
  `python scripts/install_xxhash_shim.py`. See `WALKTHROUGH.md` §7.
- **API key not found** — `.env` must sit next to `app.py`.

## License

MIT — see `LICENSE`.
