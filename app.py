import os
import io
import re
import json
import time
import uuid
import hashlib
import pathlib
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from openai import OpenAI

DOTENV_LOADED = load_dotenv()
logger = logging.getLogger("rag_app")

# Optional loaders
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None


# -----------------------------
# Configuration (env vars)
# -----------------------------
# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# System instructions
SYSTEM_INSTRUCTIONS_PATH = os.getenv("SYSTEM_INSTRUCTIONS_PATH", "system_instructions.txt")
DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant. Answer the user's question using the provided sources when relevant. "
    "If the sources don't contain the answer, say you don't know and suggest what document to add or where to look. "
    "When you use a source, cite it by SOURCE name and chunk number."
)
SYSTEM_INSTRUCTIONS = DEFAULT_SYSTEM_INSTRUCTIONS
SYSTEM_INSTRUCTIONS_SOURCE = "default"
SYSTEM_INSTRUCTIONS_PATH_RESOLVED: Optional[str] = None

# App access
APP_PASSWORD = os.getenv("APP_PASSWORD")
AUTH_REQUIRED = bool(APP_PASSWORD)

# Azure Storage / ADLS Gen2 (uses Blob endpoint)
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")  # e.g. "mystorageacct"
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER")  # e.g. "docs"
AZURE_STORAGE_PREFIX = os.getenv("AZURE_STORAGE_PREFIX", "")  # e.g. "knowledgebase/"
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")  # optional

# Index/cache paths
RAG_CACHE_DIR = pathlib.Path(os.getenv("RAG_CACHE_DIR", "./rag_cache"))
RAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = RAG_CACHE_DIR / "index.faiss"
META_PATH = RAG_CACHE_DIR / "metadata.jsonl"
STATE_PATH = RAG_CACHE_DIR / "state.json"  # stores etags/last_modified for incremental refresh
DOC_CACHE_DIR = RAG_CACHE_DIR / "doc_cache"
DOC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Chunking / Retrieval
CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
MAX_CONTEXT_CHUNKS = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "6"))

# Server-side conversation memory (very simple)
MAX_TURNS = int(os.getenv("CHAT_MAX_TURNS", "12"))

ALLOWED_EXTS = {".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm"}


# -----------------------------
# Helpers
# -----------------------------
def _format_env_value(key: str, value: Optional[str]) -> str:
    if value is None:
        return "<unset>"
    if not isinstance(value, str):
        return str(value)
    if key in {"OPENAI_API_KEY", "AZURE_STORAGE_CONNECTION_STRING"}:
        if value == "":
            return "<unset>"
        return f"****{value[-4:]}" if len(value) > 4 else "****"
    if value == "":
        return "<empty>"
    return value


def _resolve_instructions_path(path_value: str) -> pathlib.Path:
    path = pathlib.Path(path_value)
    if not path.is_absolute():
        path = (pathlib.Path(__file__).resolve().parent / path).resolve()
    return path


def load_system_instructions() -> Tuple[str, str, Optional[str]]:
    if not SYSTEM_INSTRUCTIONS_PATH:
        return DEFAULT_SYSTEM_INSTRUCTIONS, "default", None

    path = _resolve_instructions_path(SYSTEM_INSTRUCTIONS_PATH)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("System instructions file not found: %s. Falling back to default.", path)
        return DEFAULT_SYSTEM_INSTRUCTIONS, "default", str(path)
    except Exception as exc:
        logger.warning(
            "Failed to read system instructions file %s: %s. Falling back to default.",
            path,
            exc,
        )
        return DEFAULT_SYSTEM_INSTRUCTIONS, "default", str(path)

    text = text.strip()
    if text == "":
        logger.warning("System instructions file %s is empty; using empty instructions.", path)
    return text, "file", str(path)


def reload_system_instructions() -> None:
    global SYSTEM_INSTRUCTIONS, SYSTEM_INSTRUCTIONS_SOURCE, SYSTEM_INSTRUCTIONS_PATH_RESOLVED
    (
        SYSTEM_INSTRUCTIONS,
        SYSTEM_INSTRUCTIONS_SOURCE,
        SYSTEM_INSTRUCTIONS_PATH_RESOLVED,
    ) = load_system_instructions()


def log_env_config() -> None:
    values = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_MODEL": OPENAI_MODEL,
        "OPENAI_EMBED_MODEL": OPENAI_EMBED_MODEL,
        "SYSTEM_INSTRUCTIONS_PATH": SYSTEM_INSTRUCTIONS_PATH,
        "SYSTEM_INSTRUCTIONS_PATH_RESOLVED": SYSTEM_INSTRUCTIONS_PATH_RESOLVED,
        "SYSTEM_INSTRUCTIONS_SOURCE": SYSTEM_INSTRUCTIONS_SOURCE,
        "SYSTEM_INSTRUCTIONS_LENGTH": len(SYSTEM_INSTRUCTIONS or ""),
        "APP_PASSWORD_SET": bool(APP_PASSWORD),
        "AUTH_REQUIRED": AUTH_REQUIRED,
        "AZURE_STORAGE_ACCOUNT": AZURE_STORAGE_ACCOUNT,
        "AZURE_STORAGE_CONTAINER": AZURE_STORAGE_CONTAINER,
        "AZURE_STORAGE_PREFIX": AZURE_STORAGE_PREFIX,
        "AZURE_STORAGE_CONNECTION_STRING": AZURE_STORAGE_CONNECTION_STRING,
        "RAG_CACHE_DIR": str(RAG_CACHE_DIR),
        "RAG_CHUNK_TOKENS": CHUNK_TOKENS,
        "RAG_CHUNK_OVERLAP": CHUNK_OVERLAP,
        "RAG_TOP_K": TOP_K,
        "RAG_MAX_CONTEXT_CHUNKS": MAX_CONTEXT_CHUNKS,
        "CHAT_MAX_TURNS": MAX_TURNS,
    }

    logger.info("dotenv loaded: %s", DOTENV_LOADED)
    logger.info("Environment configuration:")
    for key, value in values.items():
        logger.info("  %s=%s", key, _format_env_value(key, value))


def enforce_auth(request: Request) -> None:
    if not AUTH_REQUIRED:
        return
    password = request.headers.get("x-app-password", "")
    if not password or password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def get_tokenizer():
    # cl100k_base works well for modern OpenAI text models
    return tiktoken.get_encoding("cl100k_base")


def compute_corpus_stats(meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not meta:
        return {
            "blobs": 0,
            "chunks": 0,
            "tokens": 0,
            "total_chars": 0,
            "avg_tokens_per_chunk": 0,
            "last_modified_max": "",
        }

    blobs = set()
    enc = None
    tokens = 0
    total_chars = 0
    last_modified_max = ""

    for m in meta:
        text = m.get("text") or ""
        total_chars += len(text)
        chunk_tokens = m.get("tokens")
        if isinstance(chunk_tokens, int):
            tokens += chunk_tokens
        else:
            if enc is None:
                enc = get_tokenizer()
            tokens += len(enc.encode(text))
        blob_name = m.get("blob_name")
        if blob_name:
            blobs.add(blob_name)
        last_modified = m.get("last_modified") or ""
        if last_modified and (not last_modified_max or last_modified > last_modified_max):
            last_modified_max = last_modified

    chunks = len(meta)
    return {
        "blobs": len(blobs),
        "chunks": chunks,
        "tokens": tokens,
        "total_chars": total_chars,
        "avg_tokens_per_chunk": round(tokens / chunks, 2) if chunks else 0,
        "last_modified_max": last_modified_max,
    }


def chunk_by_tokens(text: str, chunk_tokens: int, overlap: int) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    enc = get_tokenizer()
    tokens = enc.encode(text)

    chunks = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_tokens, n)
        chunk_tokens_slice = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens_slice).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def extract_text_from_bytes(blob_name: str, data: bytes) -> str:
    ext = pathlib.Path(blob_name).suffix.lower()

    # Plain text / markdown / html
    if ext in {".txt", ".md", ".markdown"}:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="ignore")

    if ext in {".html", ".htm"}:
        # Minimal HTML -> text (no extra dependency)
        try:
            html = data.decode("utf-8", errors="ignore")
        except Exception:
            html = str(data)
        # strip script/style and tags
        html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        html = re.sub(r"(?is)<br\s*/?>", "\n", html)
        html = re.sub(r"(?is)</p\s*>", "\n\n", html)
        html = re.sub(r"(?is)<.*?>", " ", html)
        return clean_text(html)

    if ext == ".pdf":
        if PdfReader is None:
            raise RuntimeError("PDF support not available. Install pypdf.")
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return clean_text("\n\n".join(parts))

    if ext == ".docx":
        if docx is None:
            raise RuntimeError("DOCX support not available. Install python-docx.")
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs if p.text and p.text.strip()]
        return clean_text("\n".join(parts))

    # Fallback: try decoding
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def azure_blob_service_client() -> BlobServiceClient:
    # ADLS Gen2 is built on Blob storage; use the blob endpoint for listing/downloading.
    if AZURE_STORAGE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

    if not AZURE_STORAGE_ACCOUNT:
        raise RuntimeError("Missing AZURE_STORAGE_ACCOUNT (or AZURE_STORAGE_CONNECTION_STRING).")

    account_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
    cred = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    return BlobServiceClient(account_url=account_url, credential=cred)


def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
        data.setdefault("blobs", {})
        data.setdefault("selected_blobs", [])
        data.setdefault("selection_initialized", False)
        return data
    return {"blobs": {}, "selected_blobs": [], "selection_initialized": False}


def save_state(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    # Batches embeddings; returns float32 matrix [n, dim]
    vectors: List[List[float]] = []
    batch_size = 64

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=batch)  # :contentReference[oaicite:3]{index=3}
        # resp.data is ordered by input index
        for item in resp.data:
            vectors.append(item.embedding)

    arr = np.array(vectors, dtype="float32")
    # Normalize for cosine similarity using inner product
    faiss.normalize_L2(arr)
    return arr


def doc_cache_key(blob_name: str) -> str:
    return hashlib.sha256(blob_name.encode("utf-8")).hexdigest()


def doc_cache_paths(blob_name: str) -> Tuple[pathlib.Path, pathlib.Path]:
    key = doc_cache_key(blob_name)
    return (DOC_CACHE_DIR / f"{key}.json", DOC_CACHE_DIR / f"{key}.npy")


def _cache_is_compatible(meta: Dict[str, Any], etag: str, last_modified: str) -> bool:
    return (
        meta.get("etag") == etag
        and meta.get("last_modified") == last_modified
        and meta.get("chunk_tokens") == CHUNK_TOKENS
        and meta.get("chunk_overlap") == CHUNK_OVERLAP
        and meta.get("embed_model") == OPENAI_EMBED_MODEL
    )


def load_doc_cache(blob_name: str, etag: str, last_modified: str) -> Optional[Dict[str, Any]]:
    meta_path, emb_path = doc_cache_paths(blob_name)
    if not meta_path.exists() or not emb_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(meta, dict) or not _cache_is_compatible(meta, etag, last_modified):
        return None
    chunks = meta.get("chunks") or []
    try:
        vectors = np.load(emb_path, allow_pickle=False)
    except Exception:
        return None
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        return None
    dim = meta.get("dim")
    if isinstance(dim, int) and vectors.shape[1] != dim:
        return None
    return {"meta": meta, "chunks": chunks, "vectors": vectors}


def cache_status_for_blob(blob_name: str, etag: str, last_modified: str) -> str:
    meta_path, emb_path = doc_cache_paths(blob_name)
    if not meta_path.exists() or not emb_path.exists():
        return "missing"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return "stale"
    return "fresh" if _cache_is_compatible(meta, etag, last_modified) else "stale"


def save_doc_cache(
    blob_name: str,
    etag: str,
    last_modified: str,
    chunk_meta: List[Dict[str, Any]],
    vectors: np.ndarray,
    content_hash: Optional[str] = None,
) -> None:
    meta_path, emb_path = doc_cache_paths(blob_name)
    meta = {
        "blob_name": blob_name,
        "etag": etag,
        "last_modified": last_modified,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model": OPENAI_EMBED_MODEL,
        "dim": int(vectors.shape[1]),
        "content_hash": content_hash or "",
        "chunks": chunk_meta,
        "saved_at": now_iso(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    np.save(emb_path, vectors)


def normalize_chunk_meta(
    blob_name: str,
    etag: str,
    last_modified: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    enc = None
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        item = dict(chunk)
        item.setdefault("blob_name", blob_name)
        item.setdefault("chunk_id", i)
        item.setdefault("etag", etag)
        item.setdefault("last_modified", last_modified)
        if "text" not in item:
            item["text"] = ""
        if not isinstance(item.get("tokens"), int):
            if enc is None:
                enc = get_tokenizer()
            item["tokens"] = len(enc.encode(item["text"])) if item["text"] else 0
        normalized.append(item)
    return normalized


def extract_answer_text(response_obj: Any) -> str:
    # SDKs may support output_text for convenience :contentReference[oaicite:4]{index=4}
    if hasattr(response_obj, "output_text") and response_obj.output_text:
        return response_obj.output_text

    # Fallback: traverse response.output
    try:
        out = getattr(response_obj, "output", None) or response_obj.get("output", [])
        parts = []
        for item in out:
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
        return "\n".join(parts).strip()
    except Exception:
        return str(response_obj)


# -----------------------------
# RAG Index
# -----------------------------
class RagIndex:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.meta: List[Dict[str, Any]] = []
        self.dim: Optional[int] = None
        self.stats: Dict[str, Any] = compute_corpus_stats([])

    def is_ready(self) -> bool:
        return self.index is not None and self.dim is not None and len(self.meta) > 0

    def load(self) -> bool:
        if FAISS_INDEX_PATH.exists() and META_PATH.exists():
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            self.meta = []
            with META_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.meta.append(json.loads(line))
            self.dim = self.index.d
            self.stats = compute_corpus_stats(self.meta)
            return True
        return False

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("No index to save")
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with META_PATH.open("w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def build_or_update_from_adls(self, selected_blobs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Incremental build:
        - Lists blobs in container/prefix
        - Downloads and chunks supported documents
        - Embeds new/changed chunks (cached per document)
        - Rebuilds FAISS index from scratch for selected documents
        """
        t0 = time.time()

        bsc = azure_blob_service_client()
        container = bsc.get_container_client(AZURE_STORAGE_CONTAINER)
        state = load_state()
        seen_blob_names = set()
        available_blob_names: List[str] = []

        selection_from_request = selected_blobs is not None
        if selected_blobs is None:
            selected_set = set(state.get("selected_blobs", []))
        else:
            selected_set = set(selected_blobs)
        select_all = len(selected_set) == 0 and not selection_from_request

        all_vectors: List[np.ndarray] = []
        all_chunk_meta: List[Dict[str, Any]] = []

        blobs_iter = container.list_blobs(name_starts_with=AZURE_STORAGE_PREFIX or None)  # :contentReference[oaicite:5]{index=5}

        included_files = 0
        skipped_files = 0
        cached_files = 0
        embedded_files = 0

        for blob in blobs_iter:
            name = blob.name
            ext = pathlib.Path(name).suffix.lower()
            if ext not in ALLOWED_EXTS:
                skipped_files += 1
                continue

            seen_blob_names.add(name)
            available_blob_names.append(name)

            etag = str(getattr(blob, "etag", "") or "")
            last_modified = getattr(blob, "last_modified", None)
            last_modified_iso = last_modified.isoformat() if last_modified else ""

            prev = state["blobs"].get(name, {})
            if not (select_all or name in selected_set):
                state["blobs"][name] = {
                    "etag": etag,
                    "last_modified": last_modified_iso,
                    "num_chunks": prev.get("num_chunks", 0),
                    "content_hash": prev.get("content_hash", ""),
                    "indexed_at": prev.get("indexed_at", ""),
                }
                continue

            included_files += 1

            cached = load_doc_cache(name, etag, last_modified_iso)
            if cached:
                chunk_meta = normalize_chunk_meta(name, etag, last_modified_iso, cached["chunks"])
                all_chunk_meta.extend(chunk_meta)
                all_vectors.append(cached["vectors"])
                cached_files += 1
                state["blobs"][name] = {
                    "etag": etag,
                    "last_modified": last_modified_iso,
                    "num_chunks": len(chunk_meta),
                    "content_hash": cached["meta"].get("content_hash", prev.get("content_hash", "")),
                    "indexed_at": now_iso(),
                }
                continue

            blob_client = container.get_blob_client(name)
            data = blob_client.download_blob().readall()  # :contentReference[oaicite:6]{index=6}
            text = extract_text_from_bytes(name, data)

            chunks = chunk_by_tokens(text, CHUNK_TOKENS, CHUNK_OVERLAP)
            enc = get_tokenizer()
            chunk_texts: List[str] = []
            chunk_meta: List[Dict[str, Any]] = []
            for j, chunk in enumerate(chunks):
                chunk_texts.append(chunk)
                chunk_meta.append(
                    {
                        "blob_name": name,
                        "chunk_id": j,
                        "etag": etag,
                        "last_modified": last_modified_iso,
                        "text": chunk,
                        "tokens": len(enc.encode(chunk)),
                    }
                )

            content_hash = sha256_bytes(data)
            if chunk_texts:
                client = openai_client()
                vectors = embed_texts(client, chunk_texts)
                save_doc_cache(name, etag, last_modified_iso, chunk_meta, vectors, content_hash)
                all_vectors.append(vectors)
                all_chunk_meta.extend(chunk_meta)

            embedded_files += 1
            state["blobs"][name] = {
                "etag": etag,
                "last_modified": last_modified_iso,
                "num_chunks": len(chunk_meta),
                "content_hash": content_hash,
                "indexed_at": now_iso(),
            }

        if select_all:
            selected_set = set(available_blob_names)
        else:
            selected_set = selected_set.intersection(available_blob_names)
        state["selected_blobs"] = sorted(selected_set)
        state["selection_initialized"] = True

        # Remove blobs that no longer exist
        removed = []
        for k in list(state["blobs"].keys()):
            if k not in seen_blob_names:
                removed.append(k)
                del state["blobs"][k]

        save_state(state)

        if not all_vectors or not all_chunk_meta:
            # Create an empty-ish state
            self.index = None
            self.meta = []
            self.dim = None
            self.stats = compute_corpus_stats([])
            if FAISS_INDEX_PATH.exists():
                FAISS_INDEX_PATH.unlink()
            if META_PATH.exists():
                META_PATH.unlink()
            return {
                "ok": True,
                "indexed_files": included_files,
                "skipped_files": skipped_files,
                "cached_files": cached_files,
                "embedded_files": embedded_files,
                "selected_files": len(selected_set),
                "chunks": 0,
                "removed_blobs": removed,
                "seconds": round(time.time() - t0, 2),
            }

        # Build FAISS from cached + newly embedded chunks
        vectors = np.vstack(all_vectors)
        dim = vectors.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self.index = index
        self.dim = dim
        self.meta = all_chunk_meta
        self.stats = compute_corpus_stats(self.meta)
        self.save()

        return {
            "ok": True,
            "indexed_files": included_files,
            "skipped_files": skipped_files,
            "cached_files": cached_files,
            "embedded_files": embedded_files,
            "selected_files": len(selected_set),
            "chunks": len(all_chunk_meta),
            "dim": dim,
            "removed_blobs": removed,
            "seconds": round(time.time() - t0, 2),
        }

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not self.is_ready():
            return []

        client = openai_client()
        qv = embed_texts(client, [query])  # shape [1, dim]
        D, I = self.index.search(qv, k)
        hits = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or idx >= len(self.meta):
                continue
            m = dict(self.meta[idx])
            m["score"] = float(score)
            hits.append(m)
        return hits


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Local RAG Chatbot (OpenAI + ADLS)")

rag = RagIndex()
conversations: Dict[str, List[Dict[str, str]]] = {}  # {conversation_id: [{"role": "...", "content": "..."}]}


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: List[Dict[str, Any]]


class AuthRequest(BaseModel):
    password: str


class ReindexRequest(BaseModel):
    selected_blobs: Optional[List[str]] = None


@app.on_event("startup")
def startup_event():
    # Try load an existing index; if missing, we don't auto-build (often better to control explicitly).
    # You can hit POST /api/reindex to build.
    reload_system_instructions()
    log_env_config()
    rag.load()


@app.get("/", response_class=HTMLResponse)
def root():
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Local RAG Chatbot</title>
  <style>
    :root {
      --bg: #f7f5ef;
      --panel: #ffffff;
      --ink: #1a1a1a;
      --muted: #5d5d5d;
      --line: #1d1d1d;
      --line-soft: #b8b8b8;
      --accent: #0f766e;
    }
    * { box-sizing: border-box; }
    body {
      font-family: "JetBrains Mono", "IBM Plex Mono", "Fira Mono", "Source Code Pro", "Menlo", "Consolas", monospace;
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      background-image:
        repeating-linear-gradient(0deg, transparent, transparent 23px, rgba(0, 0, 0, 0.04) 24px),
        repeating-linear-gradient(90deg, transparent, transparent 23px, rgba(0, 0, 0, 0.04) 24px);
    }
    header {
      padding: 16px 18px;
      background: var(--panel);
      border-bottom: 2px solid var(--line);
      display:flex;
      gap:12px;
      align-items:center;
      flex-wrap: wrap;
    }
    .spacer { flex: 1; }
    header .actions { display:flex; gap:10px; align-items:center; }
    header b {
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    header .pill {
      font-size: 11px;
      padding: 3px 8px;
      border: 2px solid var(--line);
      border-radius: 0;
      color: var(--ink);
      background: #f2f2f2;
      min-width: 0;
      flex: 1 1 420px;
      white-space: normal;
      line-height: 1.25;
    }
    #wrap { max-width: 980px; margin: 0 auto; padding: 16px; animation: rise 0.22s ease-out; }
    #chat {
      height: 70vh;
      overflow: auto;
      background: var(--panel);
      border: 2px solid var(--line);
      border-radius: 0;
      padding: 12px;
    }
    .msg { margin: 12px 0; display: flex; }
    .msg .bubble {
      padding: 10px 12px;
      border-radius: 0;
      max-width: 78%;
      white-space: pre-wrap;
      line-height: 1.35;
      border: 2px solid var(--line);
    }
    .user { justify-content: flex-end; }
    .user .bubble { background: #efefef; border-style: solid; }
    .assistant { justify-content: flex-start; }
    .assistant .bubble { background: #ffffff; border-style: dashed; }
    #bar { display:flex; gap: 10px; margin-top: 12px; }
    #input {
      flex: 1;
      padding: 10px 12px;
      border-radius: 0;
      border: 2px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      outline: none;
    }
    #input::placeholder { color: #7a7a7a; }
    #input:focus { border-color: var(--accent); box-shadow: 0 0 0 2px #d8f2ef; }
    button {
      padding: 10px 12px;
      border-radius: 0;
      border: 2px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      cursor:pointer;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-size: 11px;
    }
    button:hover { background: #f5f5f5; }
    .row { display:flex; gap: 10px; margin-top: 10px; align-items:center; }
    .muted { color: var(--muted); font-size:12px; }
    .panel {
      margin-top: 12px;
      border: 2px solid var(--line);
      background: var(--panel);
      padding: 10px;
    }
    .panel-header {
      display:flex;
      gap:10px;
      align-items:center;
      border-bottom: 2px solid var(--line);
      padding-bottom: 8px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }
    .panel-title {
      font-size: 11px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .panel-actions { display:flex; gap: 8px; align-items:center; }
    #authMask {
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(247, 245, 239, 0.96);
      z-index: 1000;
    }
    #authMask.active { display: flex; }
    #authPanel {
      width: min(420px, 92vw);
      border: 2px solid var(--line);
      background: var(--panel);
      padding: 16px;
    }
    #authPanel .row { margin-top: 10px; }
    #authInput {
      flex: 1;
      padding: 10px 12px;
      border-radius: 0;
      border: 2px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      outline: none;
    }
    #authInput::placeholder { color: #7a7a7a; }
    #authInput:focus { border-color: var(--accent); box-shadow: 0 0 0 2px #d8f2ef; }
    #authError { margin-top: 8px; color: #8a1f1f; font-size: 12px; }
    .doc-list {
      display: grid;
      gap: 6px;
      max-height: 240px;
      overflow: auto;
      padding-right: 4px;
    }
    .doc-list.collapsed { display: none; }
    .doc-item {
      display: grid;
      grid-template-columns: 16px 1fr auto;
      gap: 8px;
      align-items: center;
      border: 2px dashed var(--line-soft);
      padding: 6px 8px;
    }
    .doc-item input[type="checkbox"] { margin: 0; }
    .doc-name { font-size: 12px; word-break: break-all; }
    .doc-extra { display:flex; gap: 8px; align-items:center; }
    .doc-meta { font-size: 11px; color: var(--muted); white-space: nowrap; }
    .doc-badge {
      border: 2px solid var(--line);
      padding: 2px 6px;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .doc-badge.fresh { background: #e6f4f1; border-color: var(--accent); color: #0a4943; }
    .doc-badge.stale { background: #fdf1d6; border-color: #d08b00; color: #6d4a00; }
    .doc-badge.missing { background: #f5f5f5; border-color: var(--line-soft); color: var(--muted); }
    .doc-empty { font-size: 12px; color: var(--muted); padding: 6px 4px; }
    input[type="checkbox"] {
      appearance: none;
      width: 14px;
      height: 14px;
      border: 2px solid var(--line);
      background: #ffffff;
    }
    input[type="checkbox"]:checked {
      background: var(--accent);
      box-shadow: inset 0 0 0 2px #ffffff;
    }
    .thinking { display:flex; align-items:center; gap:8px; color: var(--muted); font-size:12px; }
    .spinner {
      width:12px;
      height:12px;
      border:2px solid var(--line);
      border-top-color: var(--accent);
      border-radius:0;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  </style>
</head>
<body>
  <div id="authMask">
    <div id="authPanel">
      <span class="panel-title">Unlock</span>
      <div class="muted" id="authHint">Enter the password to use this app.</div>
      <div class="row">
        <input id="authInput" type="password" placeholder="Password" />
        <button id="authBtn">Unlock</button>
      </div>
      <div id="authError"></div>
    </div>
  </div>
  <header>
    <b>Local RAG Chatbot</b>
    <span class="pill" id="status">Index: unknown</span>
    <span class="muted">OpenAI model + ADLS docs (RAG)</span>
    <span class="spacer"></span>
    <span class="muted" id="reindexOut"></span>
    <div class="actions">
      <button id="reindexBtn">Reindex ADLS</button>
    </div>
  </header>

  <div id="wrap">
    <div id="docsPanel" class="panel">
      <div class="panel-header">
        <span class="panel-title">Documents</span>
        <span class="muted" id="docsSummary"></span>
        <span class="spacer"></span>
        <div class="panel-actions">
          <button id="toggleDocsBtn">Collapse</button>
          <button id="selectAllBtn">Select All</button>
          <button id="selectNoneBtn">Select None</button>
        </div>
      </div>
      <div id="docsList" class="doc-list"></div>
    </div>

    <div id="chat"></div>

    <div id="bar">
      <input id="input" placeholder="Ask a question..." />
      <button id="send">Send</button>
    </div>
  </div>

<script>
  const AUTH_REQUIRED = __AUTH_REQUIRED__;
  let conversationId = null;
  let docs = [];
  let selectedDocs = new Set();
  let docsCollapsed = false;
  let appPassword = null;

  function addMsg(role, text) {
    const chat = document.getElementById('chat');
    const div = document.createElement('div');
    div.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    div.appendChild(bubble);
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  function setSources(sources) {
    const el = document.getElementById('sources');
    if (!el) return;
    if (!sources || sources.length === 0) {
      el.textContent = '';
      return;
    }
    const lines = sources.map(s => `• ${s.blob_name} (chunk ${s.chunk_id}, score ${s.score.toFixed(3)})`);
    el.textContent = "Sources:\\n" + lines.join("\\n");
  }

  function showAuthMask(message) {
    const mask = document.getElementById('authMask');
    if (!mask) return;
    mask.classList.add('active');
    const error = document.getElementById('authError');
    if (error) {
      error.textContent = message || '';
    }
    const input = document.getElementById('authInput');
    if (input) input.focus();
  }

  function hideAuthMask() {
    const mask = document.getElementById('authMask');
    if (mask) mask.classList.remove('active');
    const error = document.getElementById('authError');
    if (error) error.textContent = '';
  }

  function ensureAuth() {
    if (!AUTH_REQUIRED) return true;
    if (appPassword) return true;
    showAuthMask('Password required.');
    return false;
  }

  async function fetchWithAuth(url, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    if (appPassword) {
      headers['X-App-Password'] = appPassword;
    }
    const r = await fetch(url, { ...options, headers });
    if (r.status === 401) {
      appPassword = null;
      showAuthMask('Password required.');
      throw new Error('Unauthorized');
    }
    return r;
  }

  async function attemptAuth() {
    const input = document.getElementById('authInput');
    const btn = document.getElementById('authBtn');
    const password = input ? input.value.trim() : '';
    if (!password) {
      showAuthMask('Enter a password to continue.');
      return;
    }
    if (btn) btn.disabled = true;
    try {
      const r = await fetch('/api/auth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password })
      });
      if (!r.ok) {
        showAuthMask('Incorrect password.');
        return;
      }
      appPassword = password;
      if (input) input.value = '';
      hideAuthMask();
      await refreshStatus();
      await loadDocs();
    } catch (err) {
      showAuthMask('Unable to authenticate.');
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function formatBytes(value) {
    if (!value && value !== 0) return '';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = Number(value);
    let i = 0;
    while (size >= 1024 && i < units.length - 1) {
      size /= 1024;
      i += 1;
    }
    return `${size.toFixed(size >= 10 || i === 0 ? 0 : 1)}${units[i]}`;
  }

  function updateDocsSummary() {
    const el = document.getElementById('docsSummary');
    if (!el) return;
    el.textContent = `${selectedDocs.size} selected / ${docs.length} docs`;
  }

  function updateDocsCollapseState() {
    const list = document.getElementById('docsList');
    const btn = document.getElementById('toggleDocsBtn');
    if (!list || !btn) return;
    if (docsCollapsed) {
      list.classList.add('collapsed');
      btn.textContent = 'Expand';
    } else {
      list.classList.remove('collapsed');
      btn.textContent = 'Collapse';
    }
  }

  function renderDocs() {
    const list = document.getElementById('docsList');
    if (!list) return;
    list.innerHTML = '';

    if (!docs.length) {
      const empty = document.createElement('div');
      empty.className = 'doc-empty';
      empty.textContent = 'No supported documents found.';
      list.appendChild(empty);
      updateDocsSummary();
      return;
    }

    docs.forEach((doc) => {
      const row = document.createElement('label');
      row.className = 'doc-item';

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.checked = selectedDocs.has(doc.name);
      checkbox.addEventListener('change', () => {
        if (checkbox.checked) {
          selectedDocs.add(doc.name);
        } else {
          selectedDocs.delete(doc.name);
        }
        updateDocsSummary();
      });

      const name = document.createElement('span');
      name.className = 'doc-name';
      name.textContent = doc.name;

      const extra = document.createElement('div');
      extra.className = 'doc-extra';

      const badge = document.createElement('span');
      badge.className = `doc-badge ${doc.cache_status || 'missing'}`;
      badge.textContent = doc.cache_status || 'missing';

      const meta = document.createElement('span');
      meta.className = 'doc-meta';
      const metaParts = [];
      if (doc.last_modified) metaParts.push(doc.last_modified);
      const size = formatBytes(doc.size);
      if (size) metaParts.push(size);
      meta.textContent = metaParts.join(' | ');

      extra.appendChild(badge);
      if (meta.textContent) {
        extra.appendChild(meta);
      }

      row.appendChild(checkbox);
      row.appendChild(name);
      row.appendChild(extra);
      list.appendChild(row);
    });

    updateDocsSummary();
    updateDocsCollapseState();
  }

  async function loadDocs() {
    if (!ensureAuth()) return;
    const list = document.getElementById('docsList');
    if (list) {
      list.innerHTML = '';
    }
    try {
      const r = await fetchWithAuth('/api/docs');
      const j = await r.json();
      docs = j.docs || [];
      selectedDocs = new Set(j.selected || docs.filter(d => d.selected).map(d => d.name));
      renderDocs();
    } catch (err) {
      docs = [];
      selectedDocs = new Set();
      renderDocs();
    }
  }

  function setAllDocs(selected) {
    if (selected) {
      selectedDocs = new Set(docs.map(d => d.name));
    } else {
      selectedDocs = new Set();
    }
    renderDocs();
  }

  function formatNumber(value) {
    if (value === null || value === undefined) return '0';
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value.toLocaleString() : '0';
    }
    return String(value);
  }

  function formatAverage(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '0';
    if (value % 1 === 0) return value.toLocaleString();
    return value.toFixed(2);
  }

  async function refreshStatus() {
    if (!ensureAuth()) return;
    const r = await fetchWithAuth('/api/status');
    const j = await r.json();
    const parts = [];
    parts.push(j.ready ? 'Index: ready' : 'Index: not built');
    parts.push(`blobs ${formatNumber(j.blobs)}`);
    parts.push(`chunks ${formatNumber(j.chunks)}`);
    parts.push(`tokens ${formatNumber(j.tokens)}`);
    parts.push(`avg/chunk ${formatAverage(j.avg_tokens_per_chunk)}`);
    parts.push(`chars ${formatNumber(j.total_chars)}`);
    parts.push(`dim ${j.dim ?? '-'}`);
    parts.push(`model ${j.model || '-'}`);
    parts.push(`embed ${j.embed_model || '-'}`);
    if (j.last_modified_max) {
      parts.push(`latest ${j.last_modified_max}`);
    }
    document.getElementById('status').textContent = parts.join(' | ');
  }

  function addThinking() {
    removeThinking();
    const chat = document.getElementById('chat');
    const div = document.createElement('div');
    div.className = 'msg assistant';
    div.id = 'thinkingMsg';
    const bubble = document.createElement('div');
    bubble.className = 'bubble thinking';
    const spin = document.createElement('span');
    spin.className = 'spinner';
    const text = document.createElement('span');
    text.textContent = 'Thinking...';
    bubble.appendChild(spin);
    bubble.appendChild(text);
    div.appendChild(bubble);
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
  }

  function removeThinking() {
    const existing = document.getElementById('thinkingMsg');
    if (existing) existing.remove();
  }

  async function send() {
    const inp = document.getElementById('input');
    const text = inp.value.trim();
    if (!text) return;
    if (!ensureAuth()) return;
    inp.value = '';
    addMsg('user', text);
    setSources([]);
    addThinking();

    try {
      const r = await fetchWithAuth('/api/chat', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ message: text, conversation_id: conversationId })
      });
      const j = await r.json();
      conversationId = j.conversation_id;
      removeThinking();
      addMsg('assistant', j.answer);
      setSources(j.sources || []);
    } catch (err) {
      removeThinking();
      addMsg('assistant', 'Sorry, something went wrong while contacting the server.');
    } finally {
      removeThinking();
    }
  }

  document.getElementById('send').addEventListener('click', send);
  document.getElementById('input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') send();
  });

  document.getElementById('authBtn').addEventListener('click', attemptAuth);
  document.getElementById('authInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') attemptAuth();
  });

  document.getElementById('toggleDocsBtn').addEventListener('click', () => {
    docsCollapsed = !docsCollapsed;
    updateDocsCollapseState();
  });

  document.getElementById('selectAllBtn').addEventListener('click', () => {
    setAllDocs(true);
  });

  document.getElementById('selectNoneBtn').addEventListener('click', () => {
    setAllDocs(false);
  });

  document.getElementById('reindexBtn').addEventListener('click', async () => {
    if (!ensureAuth()) return;
    const btn = document.getElementById('reindexBtn');
    btn.disabled = true;
    document.getElementById('reindexOut').textContent = 'Running...';
    try {
      const payload = { selected_blobs: Array.from(selectedDocs) };
      const r = await fetchWithAuth('/api/reindex', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const j = await r.json();
      const out = j.ok
        ? `Indexed ${j.indexed_files} files (cached ${j.cached_files}, embedded ${j.embedded_files}, chunks ${j.chunks}) in ${j.seconds}s`
        : JSON.stringify(j);
      document.getElementById('reindexOut').textContent = out;
      await refreshStatus();
      await loadDocs();
    } catch (err) {
      document.getElementById('reindexOut').textContent = 'Reindex failed.';
    } finally {
      btn.disabled = false;
    }
  });

  if (AUTH_REQUIRED) {
    showAuthMask('');
  } else {
    refreshStatus();
    loadDocs();
  }
</script>
</body>
</html>
        """
    return HTMLResponse(html.replace("__AUTH_REQUIRED__", "true" if AUTH_REQUIRED else "false"))


@app.post("/api/auth")
def auth(req: AuthRequest):
    if not AUTH_REQUIRED:
        return {"ok": True, "required": False}
    if req.password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"ok": True, "required": True}


@app.get("/api/docs")
def list_docs(request: Request):
    enforce_auth(request)
    if not (AZURE_STORAGE_CONNECTION_STRING or (AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_CONTAINER)):
        raise HTTPException(
            status_code=400,
            detail="Missing Azure storage config. Set AZURE_STORAGE_CONNECTION_STRING OR (AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_CONTAINER).",
        )
    bsc = azure_blob_service_client()
    container = bsc.get_container_client(AZURE_STORAGE_CONTAINER)
    state = load_state()
    selected_set = set(state.get("selected_blobs", []))
    selection_initialized = bool(state.get("selection_initialized"))

    docs: List[Dict[str, Any]] = []
    blobs_iter = container.list_blobs(name_starts_with=AZURE_STORAGE_PREFIX or None)
    for blob in blobs_iter:
        name = blob.name
        ext = pathlib.Path(name).suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue
        etag = str(getattr(blob, "etag", "") or "")
        last_modified = getattr(blob, "last_modified", None)
        last_modified_iso = last_modified.isoformat() if last_modified else ""
        docs.append(
            {
                "name": name,
                "etag": etag,
                "last_modified": last_modified_iso,
                "size": getattr(blob, "size", None),
                "cache_status": cache_status_for_blob(name, etag, last_modified_iso),
            }
        )

    if not selection_initialized and docs:
        selected_set = {d["name"] for d in docs}
        state["selected_blobs"] = sorted(selected_set)
        state["selection_initialized"] = True
        save_state(state)

    for d in docs:
        d["selected"] = d["name"] in selected_set

    return {"ok": True, "docs": docs, "selected": sorted(selected_set)}


@app.get("/api/status")
def status(request: Request):
    enforce_auth(request)
    stats = rag.stats if hasattr(rag, "stats") else compute_corpus_stats(rag.meta)
    return {
        "ready": rag.is_ready(),
        "chunks": stats.get("chunks", len(rag.meta)),
        "blobs": stats.get("blobs", 0),
        "tokens": stats.get("tokens", 0),
        "total_chars": stats.get("total_chars", 0),
        "avg_tokens_per_chunk": stats.get("avg_tokens_per_chunk", 0),
        "last_modified_max": stats.get("last_modified_max", ""),
        "dim": rag.dim,
        "model": OPENAI_MODEL,
        "embed_model": OPENAI_EMBED_MODEL,
    }


@app.post("/api/reindex")
def reindex(request: Request, req: Optional[ReindexRequest] = None):
    enforce_auth(request)
    # Builds/refreshes the local index from ADLS
    if not (AZURE_STORAGE_CONNECTION_STRING or (AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_CONTAINER)):
        raise HTTPException(
            status_code=400,
            detail="Missing Azure storage config. Set AZURE_STORAGE_CONNECTION_STRING OR (AZURE_STORAGE_ACCOUNT + AZURE_STORAGE_CONTAINER).",
        )
    selected = req.selected_blobs if req else None
    result = rag.build_or_update_from_adls(selected)
    return result


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: Request, req: ChatRequest):
    enforce_auth(request)
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    # Conversation ID
    conv_id = req.conversation_id or str(uuid.uuid4())
    turns = conversations.get(conv_id, [])

    # Retrieve context
    hits = rag.search(msg, k=TOP_K) if rag.is_ready() else []
    hits = hits[:MAX_CONTEXT_CHUNKS]

    context_blocks = []
    for h in hits:
        context_blocks.append(
            f"[SOURCE: {h['blob_name']} | chunk {h['chunk_id']}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks).strip()

    system_instructions = SYSTEM_INSTRUCTIONS

    # Very lightweight memory (last N turns)
    # We keep it short and just append a compact transcript into the user prompt.
    turns = turns[-MAX_TURNS * 2 :]
    transcript = ""
    if turns:
        transcript_lines = []
        for t in turns:
            transcript_lines.append(f"{t['role'].upper()}: {t['content']}")
        transcript = "\n".join(transcript_lines).strip()

    user_prompt = ""
    if transcript:
        user_prompt += f"Conversation so far:\n{transcript}\n\n"

    if context:
        user_prompt += f"Sources:\n{context}\n\n"

    user_prompt += f"User question:\n{msg}"

    # Call OpenAI Responses API :contentReference[oaicite:7]{index=7}
    client = openai_client()
    resp = client.responses.create(
        model=OPENAI_MODEL,
        instructions=system_instructions,
        input=user_prompt,
    )

    answer = extract_answer_text(resp).strip()

    # Save conversation memory
    turns.append({"role": "user", "content": msg})
    turns.append({"role": "assistant", "content": answer})
    conversations[conv_id] = turns[-MAX_TURNS * 2 :]

    # Return sources without full chunk text
    sources = [
        {
            "blob_name": h["blob_name"],
            "chunk_id": h["chunk_id"],
            "score": h["score"],
            "etag": h.get("etag", ""),
            "last_modified": h.get("last_modified", ""),
        }
        for h in hits
    ]

    return ChatResponse(conversation_id=conv_id, answer=answer, sources=sources)


# Entry point for: python app.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
