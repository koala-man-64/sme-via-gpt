# sme-via-gpt

Local RAG chatbot (FastAPI) that indexes documents from Azure Blob Storage (ADLS Gen2 via Blob endpoint) into a FAISS vector index and answers questions using OpenAI.

## Requirements
- Python 3.10+
- An Azure Storage container with documents
- An OpenAI API key

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your values.

## Run
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000`.

## Environment variables
Required:
- `OPENAI_API_KEY`
- Azure storage: either `AZURE_STORAGE_CONNECTION_STRING` **or** (`AZURE_STORAGE_ACCOUNT` + `AZURE_STORAGE_CONTAINER`)

Optional:
- `AZURE_STORAGE_PREFIX` (default: empty)
- `OPENAI_MODEL` (default: `gpt-5-nano`)
- `OPENAI_EMBED_MODEL` (default: `text-embedding-3-small`)
- `APP_PASSWORD` (if set, UI prompts for password and API requires `X-App-Password` header)
- `SYSTEM_INSTRUCTIONS_PATH` (default: `system_instructions.txt`)
- `RAG_CACHE_DIR` (default: `./rag_cache`, not committed)

## Indexing
Use the UI “Reindex ADLS” button (or call `POST /api/reindex`) to build/update the local FAISS index from Azure.
