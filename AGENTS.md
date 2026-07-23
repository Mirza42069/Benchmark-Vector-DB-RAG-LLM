# AGENTS.md
## What this repo is
- Streamlit RAG chatbot + benchmark UI comparing Pinecone, PostgreSQL+pgvector, and ChromaDB.
- Entrypoints: `Benchmark.py`, `chatbot_rag.py`.
- Ingestion scripts: `ingestion_PN.py`, `ingestion_PG.py`, `ingestion_CH.py`.
- Shared code: `utils/document_processor.py`, `utils/security.py`, `utils/ground_truth.py`.
## Cursor/Copilot rules
- Cursor rules: none found in `.cursor/rules/` or `.cursorrules`.
- Copilot rules: none found in `.github/copilot-instructions.md`.
## Setup
- Python: 3.12.x
- Commands:
```bash
python -m venv venv
# Windows (PowerShell)
./venv/Scripts/Activate.ps1
# Windows (cmd)
./venv/Scripts/activate.bat
# macOS/Linux
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements.txt --upgrade
```

## Run ("build")
No compile/build step beyond import/syntax checks.
Run Streamlit apps:
```bash
streamlit run Benchmark.py
streamlit run chatbot_rag.py
python -m compileall .
python -m pip check
```

## Lint/format (optional, not configured in-repo)
If you need fast agent feedback loops, install Ruff locally:
```bash
python -m pip install ruff
python -m ruff check .
python -m ruff format .
```
Guideline: avoid repo-wide formatting unless the change requires it.

## Tests
No automated tests are present.
If/when pytest tests exist, key commands (single-test first):
```bash
pytest
pytest tests/test_file.py
pytest tests/test_file.py::test_name
pytest -k "keyword"
pytest --lf
```

## PostgreSQL + pgvector (docker)
Compose file: `docker-compose.yml` (requires `DB_PASSWORD`).
Start / logs / stop:
```bash
# macOS/Linux
DB_PASSWORD=your_password docker compose up -d
# Windows (PowerShell)
$env:DB_PASSWORD="your_password"; docker compose up -d
docker compose logs -f postgres
docker compose down
```

## Ingestion (DANGER: deletes data)
These scripts intentionally clear existing vectors/collections for a fresh run.
Do not point them at production resources.

- Pinecone: `python ingestion_PN.py`
  - Clears vectors in the index; may delete/recreate the index if dimension mismatches.
- PostgreSQL: `python ingestion_PG.py`
  - Deletes the `PGVector` collection and recreates it.
- ChromaDB: `python ingestion_CH.py`
  - Deletes the Chroma collection and recreates it.

## Environment variables

Loaded via `python-dotenv` (`load_dotenv()`); `.env` is gitignored.

Common:
- `EMBEDDING_MODEL` (keep consistent across databases when benchmarking)
- `CHAT_MODEL`
- `SCORE_THRESHOLD`
- `TOP_K`
- `COLLECTION_NAME`

Pinecone:
- `PINECONE_API_KEY` (required)
- `PINECONE_INDEX_NAME`

PostgreSQL:
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`
- `DB_PASSWORD` (required; never hardcode)

## Code style and conventions

### Layout
- Script-first repo: keep top-level entrypoints runnable via `python file.py` / `streamlit run file.py`.
- Entrypoints currently use `sys.path.append(...)` to import `utils.*`; keep this unless packaging the repo.

### Imports
- Order: standard library -> third-party -> local (`utils.*`).
- Avoid wildcard imports.

### Formatting
- 4-space indentation.
- Prefer f-strings.
- Keep diffs tight; do not reformat unrelated code.

### Types
- Add type hints for new/edited functions (especially in `utils/`).
- Prefer Python 3.12 style in new code (`str | None`, `list[str]`); do not mass-refactor existing code.

### Naming
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

### Error handling and security
- At boundaries (DB connect, retrieval, LLM calls, ingestion loops):
  - Catch `Exception as e`.
  - Log details with `logger.exception(...)` (preferred) or `logger.error(...)`.
  - Show users a safe message (avoid leaking secrets/connection strings).
- Use `utils/security.py` helpers where applicable:
  - `sanitize_query(...)` before retrieval/LLM.
  - `sanitize_error_message(e)` for user-visible failures.
  - `escape_html(...)` before rendering untrusted text with `unsafe_allow_html=True`.
  - `build_pg_connection_string(...)` for Postgres URLs.
  - `require_env(...)` for required env vars in scripts.

### Streamlit
- Put heavyweight initializations in `st.cache_resource`.
- Avoid network calls at import time.
- Use `st.session_state` for UI state.

### Benchmark invariants
- Keep chunking consistent for fair comparisons:
  - `utils/document_processor.py` uses `chunk_size=500`, `chunk_overlap=100`.
- Preserve metadata keys used downstream:
  - `source_file`, `detected_language`, `chunk_language`, `chunk_id`, `chunk_length`.
- Keep NUL-byte stripping (`\x00`) in ingestion scripts.

## Repo hygiene
- Never commit `.env` or credentials (see `.gitignore`).
- Keep generated data untracked: `chroma_db/`, `venv/`, `__pycache__/`, Postgres docker volume.
