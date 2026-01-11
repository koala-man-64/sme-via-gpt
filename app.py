import os
print("DEBUG: Starting app.py execution...")

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
from azure.core.exceptions import ResourceNotFoundError

from openai import OpenAI

print("DEBUG: Imports completed. Loading dotenv...")
DOTENV_LOADED = load_dotenv()

logger = logging.getLogger("rag_app")

TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}

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
AUTH_REQUIRED_ENV = os.getenv("AUTH_REQUIRED")
if AUTH_REQUIRED_ENV is None:
    AUTH_REQUIRED = bool(APP_PASSWORD)
else:
    AUTH_REQUIRED = AUTH_REQUIRED_ENV.strip().lower() in TRUE_VALUES

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

# UI template
INDEX_HTML_PATH = pathlib.Path(__file__).resolve().parent / "index.html"

# Chunking / Retrieval
CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
MAX_CONTEXT_CHUNKS = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "6"))
MODEL_LIMIT = 128000 # GPT-4o / mini limit

# Server-side conversation memory (very simple)
MAX_TURNS = int(os.getenv("CHAT_MAX_TURNS", "12"))

ALLOWED_EXTS = {".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm"}

# Model Definitions
MODELS = [
    {"id": "gpt-5.2", "name": "gpt-5.2", "desc": "Best overall reasoning/coding ($1.75/1M in)", "price_in": 1.75, "price_cache": 0.18, "price_out": 14.00},
    {"id": "gpt-5.1", "name": "gpt-5.1", "desc": "Strong reasoning, cheaper ($1.25/1M in)", "price_in": 1.25, "price_cache": 0.13, "price_out": 10.00},
    {"id": "gpt-5-mini", "name": "gpt-5-mini", "desc": "Fast + cost-efficient ($0.25/1M in)", "price_in": 0.25, "price_cache": 0.03, "price_out": 2.00},
    {"id": "gpt-5-nano", "name": "gpt-5-nano", "desc": "Cheapest GPT-5 family ($0.05/1M in)", "price_in": 0.05, "price_cache": 0.01, "price_out": 0.40},
    {"id": "gpt-4.1", "name": "gpt-4.1", "desc": "Smartest non-reasoning ($2.00/1M in)", "price_in": 2.00, "price_cache": 0.50, "price_out": 8.00},
    {"id": "gpt-4.1-mini", "name": "gpt-4.1-mini", "desc": "Balanced price/perf ($0.40/1M in)", "price_in": 0.40, "price_cache": 0.10, "price_out": 1.60},
    {"id": "gpt-4.1-nano", "name": "gpt-4.1-nano", "desc": "Very cheap text+vision ($0.10/1M in)", "price_in": 0.10, "price_cache": 0.025, "price_out": 0.40},
    {"id": "gpt-4o", "name": "gpt-4o", "desc": "Fast 'omni' general ($2.50/1M in)", "price_in": 2.50, "price_cache": 1.25, "price_out": 10.00},
    {"id": "gpt-4o-mini", "name": "gpt-4o-mini", "desc": "Cheapest mainstream ($0.15/1M in)", "price_in": 0.15, "price_cache": 0.075, "price_out": 0.60},
    {"id": "o3", "name": "o3", "desc": "Reasoning model ($2.00/1M in)", "price_in": 2.00, "price_cache": 0.50, "price_out": 8.00},
    {"id": "o4-mini", "name": "o4-mini", "desc": "Cheaper reasoning ($1.10/1M in)", "price_in": 1.10, "price_cache": 0.28, "price_out": 4.40},
    {"id": "o1", "name": "o1", "desc": "Heavy reasoning ($15.00/1M in)", "price_in": 15.00, "price_cache": 7.50, "price_out": 60.00},
]



# -----------------------------
# Conversation Memory (In-Memory)
# -----------------------------
# -----------------------------
# Conversation Memory (ADLS-backed)
# -----------------------------
class ConversationManager:
    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns

    def _get_blob_client(self, conversation_id: str):
        blob_name = f"chats/{conversation_id}.json"
        # Since azure_blob_service_client is global helper
        bsc = azure_blob_service_client() 
        return bsc.get_blob_client(container=AZURE_STORAGE_CONTAINER, blob=blob_name)

    def load_session(self, conversation_id: str) -> Tuple[List[Dict[str, str]], List[str]]:
        """Returns (history_messages, selected_blobs)"""
        if not conversation_id:
            return [], []

        # SEC-05: Path Traversal prevention
        # Validate that conversation_id is a simple UUID-like alphanumeric string
        if not re.match(r'^[a-zA-Z0-9-]+$', conversation_id):
            logger.warning(f"Invalid conversation_id attempted: {conversation_id}")
            return [], []
        
        try:
            client = self._get_blob_client(conversation_id)
            if not client.exists():
                return [], []
            
            data = json.loads(client.download_blob().readall())
            history = data.get("history", [])
            selected = data.get("selected_blobs", [])
            return history, selected
        except ResourceNotFoundError:
            return [], []
        except Exception as e:
            logger.error(f"Error loading session {conversation_id}: {e}")
            return [], []

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        hist, _ = self.load_session(conversation_id)
        return hist

    def save_turn(self, conversation_id: str, history: List[Dict[str, Any]], selected_blobs: List[str]):
        try:
            client = self._get_blob_client(conversation_id)
            payload = {
                "history": history,
                "selected_blobs": list(selected_blobs),
                "last_updated": now_iso()
            }
            client.upload_blob(json.dumps(payload, ensure_ascii=False), overwrite=True)
        except Exception as e:
            logger.error(f"Error saving session {conversation_id}: {e}")

    def add_turn(self, conversation_id: str, user_msg: str, assistant_msg: str, selected_blobs: List[str], stats: Optional[Dict[str, float]] = None):
        history, _ = self.load_session(conversation_id)
        
        history.append({"role": "user", "content": user_msg})
        
        assistant_turn = {"role": "assistant", "content": assistant_msg}
        if stats:
            assistant_turn["stats"] = stats
             
        history.append(assistant_turn)
        
        if len(history) > self.max_turns * 2:
            history = history[-(self.max_turns * 2):]
            
        self.save_turn(conversation_id, history, selected_blobs)

    def list_sessions(self) -> List[Dict[str, str]]:
        try:
            # azure_blob_service_client is global
            bsc = azure_blob_service_client()
            container = bsc.get_container_client(AZURE_STORAGE_CONTAINER)
            blobs = container.list_blobs(name_starts_with="chats/")
            
            sessions = []
            for b in blobs:
                # Name format: chats/{uuid}.json
                name = b.name
                if not name.endswith(".json"):
                    continue
                
                cid = name.replace("chats/", "").replace(".json", "")
                # We can return creation_time or last_modified as "date"
                sessions.append({
                    "id": cid,
                    "last_modified": b.last_modified.isoformat() if b.last_modified else "",
                    "name": cid # Could be enhanced to store a title
                })
            
            # Sort by last_modified desc
            sessions.sort(key=lambda x: x["last_modified"], reverse=True)
            return sessions
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []

conversation_manager = ConversationManager(MAX_TURNS)


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


def is_local(request: Request) -> bool:
    client = request.client
    if not client:
        return False
    return client.host in ("127.0.0.1", "localhost", "::1")

def enforce_auth(request: Request) -> None:
    if not AUTH_REQUIRED:
        return
    if is_local(request):
        return
    
    password_hash = request.headers.get("x-app-password", "")
    # Calculate expected hash of the stored password
    expected_hash = sha256_bytes(APP_PASSWORD.encode("utf-8")) if APP_PASSWORD else ""
    
    # Compare hashes
    if not password_hash or password_hash != expected_hash:
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


def count_messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-4o") -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        
    tokens_per_message = 3
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += 1
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


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
    exclude_interactive = (
        os.getenv("AZURE_EXCLUDE_INTERACTIVE_BROWSER_CREDENTIAL", "true").strip().lower()
        in TRUE_VALUES
    )
    cred = DefaultAzureCredential(exclude_interactive_browser_credential=exclude_interactive)
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
    # Handle ChatCompletion object
    if hasattr(response_obj, "choices") and response_obj.choices:
        return response_obj.choices[0].message.content or ""

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

    def get_full_context(self) -> str:
        if not self.is_ready():
            return ""
        # Join all text from all chunks in order
        # Assuming chunks are stored in order in meta
        # A safer way might be to group by blob and sort by chunk_id, but current simple impl might suffice
        return "\n\n".join([m.get("text", "") for m in self.meta])



# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Local RAG Chatbot (OpenAI + ADLS)")

rag = RagIndex()


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    mode: str = "rag"  # "rag", "direct", "cache"
    model: Optional[str] = None # e.g. "gpt-4o"


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: List[Dict[str, Any]]
    usage_info: Dict[str, Any]


class ModelInfo(BaseModel):
    id: str
    name: str
    desc: str
    price_in: float
    price_cache: float
    price_out: float


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    default: str


class AuthRequest(BaseModel):
    password: str


class ReindexRequest(BaseModel):
    selected_blobs: Optional[List[str]] = None


@app.on_event("startup")
def startup_event():
    # Ensure logs are visible
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    # Try load an existing index; if missing, we don't auto-build (often better to control explicitly).
    # You can hit POST /api/reindex to build.
    if AUTH_REQUIRED and not APP_PASSWORD:
        raise RuntimeError("AUTH_REQUIRED is true but APP_PASSWORD is not set.")
    reload_system_instructions()
    log_env_config()
    rag.load()


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    if not INDEX_HTML_PATH.exists():
        return HTMLResponse(
            "<!doctype html><html><body><h1>Server misconfigured</h1><p>Missing index.html</p></body></html>",
            status_code=500,
        )
    html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    
    # Determine if auth is strictly required for this user
    # If local, treated as not required
    req_auth = AUTH_REQUIRED and not is_local(request)
    
    return HTMLResponse(html.replace("__AUTH_REQUIRED__", "true" if req_auth else "false"))


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    missing: List[str] = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not AZURE_STORAGE_CONNECTION_STRING:
        if not AZURE_STORAGE_ACCOUNT:
            missing.append("AZURE_STORAGE_ACCOUNT")
        if not AZURE_STORAGE_CONTAINER:
            missing.append("AZURE_STORAGE_CONTAINER")
    if AUTH_REQUIRED and not APP_PASSWORD:
        missing.append("APP_PASSWORD")

    ready = len(missing) == 0
    status_code = 200 if ready else 503
    return JSONResponse({"ready": ready, "missing": missing}, status_code=status_code)


@app.post("/api/auth")
def auth(req: AuthRequest):
    if not AUTH_REQUIRED:
        return {"ok": True, "required": False}
    if req.password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"ok": True, "required": True}


@app.get("/api/docs")
def list_docs(request: Request):
    logger.info("API: list_docs called")
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
    logger.info("API: status called")
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


@app.get("/api/models", response_model=ModelsResponse)
def get_models(request: Request):
    logger.info("API: get_models called")
    enforce_auth(request)
    return ModelsResponse(models=MODELS, default=OPENAI_MODEL)


@app.get("/api/conversations")
async def list_conversations_endpoint(request: Request):
    """Lists available chat sessions."""
    enforce_auth(request)
    sessions = conversation_manager.list_sessions()
    return {"sessions": sessions}

@app.get("/api/history/{conversation_id}")
async def get_history_endpoint(conversation_id: str, request: Request):
    """Retrieves chat history and context for a given conversation ID."""
    enforce_auth(request)
    history, selected = conversation_manager.load_session(conversation_id)
    return {"history": history, "selected_blobs": selected}


class AuthRequest(BaseModel):
    password: str

@app.post("/api/auth")
async def auth_check(req: AuthRequest, request: Request):
    """Verify password hash. Frontend sends SHA-256 hash of user input."""
    # We expect the payload 'password' to ALREADY be a hash from the client.
    # We compare it to the hash of our server-side APP_PASSWORD.
    expected_hash = sha256_bytes(APP_PASSWORD.encode("utf-8")) if APP_PASSWORD else ""
    
    if req.password == expected_hash:
        return {"status": "ok"}
    raise HTTPException(status_code=401, detail="Incorrect password")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: Request, req: ChatRequest):
    enforce_auth(request)
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    conv_id = req.conversation_id or str(uuid.uuid4())
    mode = req.mode.lower() if req.mode else "rag"
    
    # Select Model
    selected_model = req.model or OPENAI_MODEL
    
    # 1. Retrieve Context
    context_text = ""
    hits = []
    
    if mode == "rag":
        hits = rag.search(msg, k=TOP_K) if rag.is_ready() else []
        hits = hits[:MAX_CONTEXT_CHUNKS]
        context_blocks = []
        for h in hits:
            context_blocks.append(
                f"[SOURCE: {h['blob_name']} | chunk {h['chunk_id']}]\n{h['text']}"
            )
        context_text = "\n\n---\n\n".join(context_blocks).strip()
        
    elif mode in {"direct", "cache"}:
        # Use full text of all indexed docs
        context_text = rag.get_full_context()
        # No specific hits for source highlighting, but we can list blobs?
        # For now, empty hits list as we are using "All Scoped"
        
    # 2. Build Message List
    messages = []
    
    # SYSTEM
    messages.append({"role": "system", "content": SYSTEM_INSTRUCTIONS})
    
    # CONTEXT & HISTORY
    # Cache Mode: System -> Context -> History -> User (Stable Prefix)
    # RAG/Direct Mode: System -> History -> Context -> User (Standard)
    
    history = conversation_manager.get_history(conv_id)
    
    if mode == "cache":
        # Context first (Pinned)
        if context_text:
             messages.append({
                "role": "user", 
                "content": f"Reference Context:\n\n{context_text}"
            })
             
        # Then History
        messages.extend(history)
        
    else:
        # History first
        messages.extend(history)
        
        # Then Context
        if context_text:
            messages.append({
                "role": "user", 
                "content": f"Here is the retrieved context for the next question:\n\n{context_text}"
            })

    # USER
    messages.append({"role": "user", "content": msg})
    
    
    # 3. Calculate Tokens
    # We calculate separately to give usage breakdown
    
    # Helper to count safe
    def count(msgs):
        return count_messages_tokens(msgs, model=selected_model) # Use selected model for counting

    tok_system = count([{"role": "system", "content": SYSTEM_INSTRUCTIONS}])
    tok_history = count(history)
    tok_context = 0
    if context_text:
        # Context message is user role
        # We need to reconstruct the exact message used above to be accurate
        # But for breakdown, we can just measure the content payload roughly or the specific message
        # Let's count the actual message object added
        ctx_msg = {"role": "user", "content": f"Reference Context:\n\n{context_text}"} if mode == "cache" else {"role": "user", "content": f"Here is the retrieved context for the next question:\n\n{context_text}"}
        tok_context = count([ctx_msg])
    
    tok_query = count([{"role": "user", "content": msg}])
    
    total_tokens = count(messages) # Accurate total for API

    
    oldest_turn_info = "N/A"
    if history:
        # User message is at index 0 of history list
        first_hist = history[0]
        oldest_turn_info = f"Role: {first_hist['role']}, Preview: {first_hist['content'][:30]}..."

    # 4. Call OpenAI
    client = openai_client()
    resp = client.chat.completions.create(
        model=selected_model,
        messages=messages,
    )
    
    answer = extract_answer_text(resp).strip()
    
    # Capture Cache Stats & Actual Tokens
    cached_tokens = 0
    actual_prompt_tokens = 0
    completion_tokens = 0
    
    if hasattr(resp, "usage") and resp.usage:
        actual_prompt_tokens = resp.usage.prompt_tokens
        completion_tokens = resp.usage.completion_tokens
        
        # prompt_tokens_details is available in recent models
        details = getattr(resp.usage, "prompt_tokens_details", None)
        if details:
             cached_tokens = getattr(details, "cached_tokens", 0)

    # Calculate Cached Chunks
    # Get avg chunk size
    corpus_stats = rag.stats if hasattr(rag, "stats") else {}
    avg_chunk_size = corpus_stats.get("avg_tokens_per_chunk", 0) or CHUNK_TOKENS # fallback
    
    cached_chunks_est = 0
    if avg_chunk_size > 0:
        cached_chunks_est = cached_tokens / avg_chunk_size
        
    # 5. Update History with stats
    # Pass current actual selected set (or what was used)
    # The 'rag.search' uses self.meta which is filtered by 'build_or_update_from_adls'.
    # In 'api/chat', 'rag' is global.
    # The user selection state is typically Global in this single-user app design, stored in 'state.json'.
    # So we can fetch it from there or from the 'rag' index if we store it.
    
    # Best effort: retrieve from state.json
    try:
        current_state = load_state()
        current_selection = current_state.get("selected_blobs", [])
    except:
        current_selection = []

    conversation_manager.add_turn(conv_id, msg, answer, selected_blobs=current_selection, stats={"cached_chunks": cached_chunks_est})
    
    # Calculate Aggregate Average from History
    # Iterate history, sum up 'stats.cached_chunks' for assistant messages
    hist_entries = conversation_manager.get_history(conv_id)
    total_cached = 0.0
    count_turns = 0
    for m in hist_entries:
        if m.get("role") == "assistant" and "stats" in m:
            total_cached += m["stats"].get("cached_chunks", 0.0)
            count_turns += 1
            
    avg_cached_chunks = 0.0
    if count_turns > 0:
        avg_cached_chunks = total_cached / count_turns

    # Return structure
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
    
    usage_info = {
        "context_tokens_estimated": total_tokens,
        "actual_prompt_tokens": actual_prompt_tokens,
        "completion_tokens": completion_tokens,
        "model_limit": MODEL_LIMIT,
        "breakdown": {
            "system": tok_system,
            "history": tok_history,
            "context": tok_context,
            "query": tok_query
        },
        "oldest_history_turn": oldest_turn_info,
        "history_turns_count": len(history) // 2,
        "mode": mode,
        "avg_cached_chunks": round(avg_cached_chunks, 1)
    }

    return ChatResponse(
        conversation_id=conv_id, 
        answer=answer, 
        sources=sources,
        usage_info=usage_info
    )


# Entry point for: python app.py
if __name__ == "__main__":
    print("DEBUG: Entering main block...")
    import uvicorn
    print("DEBUG: Starting uvicorn...")


    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
