"""
LangChain-powered RAG over the markdown knowledge base.

Pipeline:
  1. Chunk `data/knowledge_base.md` by paragraph (target ~1200 chars).
  2. Embed every chunk with `GoogleGenerativeAIEmbeddings` in small batches
     to respect Gemini's free-tier rate limits, retrying on 429 and honoring
     the provider's explicit `retryDelay` when given.
  3. Wrap the (text, embedding) pairs in a FAISS vector store.
  4. Persist with `FAISS.save_local` so subsequent runs skip the embed step.
     The cache is keyed on (file mtime, file size, embed model) via a
     sidecar JSON; any change invalidates it.

If `faiss-cpu` isn't available at import time we fall back transparently
to `langchain_core.vectorstores.InMemoryVectorStore` — it re-embeds every
run but keeps the bot functional.

`search(query, k=3)` returns `(chunk_text, score_0_to_1)` so the existing
confidence math in the responder (and the 0.80 escalation threshold) keep
working unchanged.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src import config


# --- pick FAISS or fall back --------------------------------------------------
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    _HAS_FAISS = True
except Exception as _faiss_exc:  # noqa: BLE001
    FAISS = None  # type: ignore[assignment]
    _HAS_FAISS = False
    print(f"[kb] faiss-cpu unavailable ({_faiss_exc}); falling back to InMemoryVectorStore.")


# --- batching + retry tunables (Gemini free tier) -----------------------------
BATCH_SIZE = 10                 # texts per embedding call
BATCH_SLEEP_SECONDS = 7.0       # pacing between batches
MAX_RETRIES = 6                 # retries on 429 per batch
RETRY_BASE_SECONDS = 8.0        # exponential backoff base
RETRY_MAX_WAIT_SECONDS = 90.0   # cap for any single sleep


_RETRY_DELAY_RE = re.compile(r"retry[_ ]?(?:in|delay)[\"']?\s*[:=]?\s*[\"']?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _is_rate_limited(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "resource_exhausted" in text or ("rate" in text and "limit" in text)


def _suggested_retry_delay(exc: Exception) -> Optional[float]:
    m = _RETRY_DELAY_RE.search(str(exc))
    if not m:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


# --- chunking -----------------------------------------------------------------

def _chunk_text(text: str, target_chars: int = 1200, overlap: int = 120) -> list[str]:
    """
    Pack blank-line paragraphs into chunks ≈ target_chars, with a tail
    overlap between adjacent chunks so cross-boundary sentences are still
    retrievable. Idempotent and pure, so we can unit-test it independently.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
        elif len(buf) + len(p) + 2 <= target_chars:
            buf = f"{buf}\n\n{p}"
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap and len(buf) > overlap else ""
            buf = (tail + "\n\n" + p).strip() if tail else p
    if buf:
        chunks.append(buf)
    return chunks


# --- cache invalidation -------------------------------------------------------

def _source_fingerprint() -> dict[str, float]:
    st = config.KB_PATH.stat()
    return {"mtime": st.st_mtime, "size": st.st_size}


def _cache_is_fresh() -> bool:
    """True if the on-disk FAISS index still matches the current KB+model."""
    if not (config.KB_INDEX_DIR.exists() and config.KB_META_PATH.exists()):
        return False
    try:
        meta = json.loads(config.KB_META_PATH.read_text(encoding="utf-8"))
        current = _source_fingerprint()
        if meta.get("mtime") != current["mtime"] or meta.get("size") != current["size"]:
            return False
        if meta.get("embed_model") != config.EMBED_MODEL:
            return False
        if meta.get("backend") != ("faiss" if _HAS_FAISS else "memory"):
            return False
        return True
    except Exception:
        return False


def _write_meta(n_chunks: int) -> None:
    meta = {
        **_source_fingerprint(),
        "embed_model": config.EMBED_MODEL,
        "backend": "faiss" if _HAS_FAISS else "memory",
        "n_chunks": n_chunks,
    }
    config.KB_META_PATH.write_text(json.dumps(meta), encoding="utf-8")


# --- batched embed with retry -------------------------------------------------

class _RateLimitedEmbeddings(Embeddings):
    """
    Wraps `GoogleGenerativeAIEmbeddings` and adds batched calls, inter-batch
    pacing, and retry-with-backoff on 429 (honouring any provider-suggested
    `retryDelay`). Inherits the LangChain `Embeddings` interface so FAISS
    and InMemoryVectorStore recognise it as a first-class embeddings object.
    """

    def __init__(self, base: GoogleGenerativeAIEmbeddings) -> None:
        self._base = base

    def _call_with_retry(self, fn, *args, label: str):
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                return fn(*args)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_rate_limited(exc) or attempt == MAX_RETRIES - 1:
                    raise
                wait = _suggested_retry_delay(exc) or (RETRY_BASE_SECONDS * (2 ** attempt))
                wait = min(wait + 1.0, RETRY_MAX_WAIT_SECONDS)
                print(f"[kb] rate limit on {label}, retrying in {wait:.1f}s ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
        raise RuntimeError(f"embedding retries exhausted: {last_exc}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            print(f"[kb] embedding chunks {start + 1}-{start + len(batch)} of {total}…")
            vectors = self._call_with_retry(self._base.embed_documents, batch, label=f"batch {start // BATCH_SIZE + 1}")
            out.extend(vectors)
            if start + BATCH_SIZE < total:
                time.sleep(BATCH_SLEEP_SECONDS)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._call_with_retry(self._base.embed_query, text, label="query embed")


# --- public store -------------------------------------------------------------

class KnowledgeBase:
    """LangChain-backed RAG store. Construct once at startup, query many times."""

    def __init__(self) -> None:
        self._store: Optional[VectorStore] = None
        self._embeddings: Optional[_RateLimitedEmbeddings] = None
        self._n_chunks: int = 0
        self._load()

    # ----- properties --------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._store is not None

    # ----- internal ----------------------------------------------------------
    def _make_embeddings(self) -> _RateLimitedEmbeddings:
        base = GoogleGenerativeAIEmbeddings(
            model=config.EMBED_MODEL,
            google_api_key=config.LLM_API_KEY,
        )
        return _RateLimitedEmbeddings(base)

    def _load(self) -> None:
        if not config.KB_PATH.exists() or config.KB_PATH.stat().st_size == 0:
            print(f"[kb] WARNING: knowledge base not found or empty at {config.KB_PATH} — RAG fallback disabled.")
            return

        text = config.KB_PATH.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        if not chunks:
            print("[kb] WARNING: knowledge base produced 0 chunks — RAG fallback disabled.")
            return

        self._embeddings = self._make_embeddings()

        # Try the persistent FAISS cache first.
        if _HAS_FAISS and _cache_is_fresh():
            try:
                store = FAISS.load_local(  # type: ignore[union-attr]
                    str(config.KB_INDEX_DIR),
                    embeddings=self._embeddings,
                    allow_dangerous_deserialization=True,  # safe: file is created by us
                )
                self._store = store
                self._n_chunks = len(chunks)
                print(f"[kb] loaded FAISS index from cache (model={config.EMBED_MODEL}, chunks={self._n_chunks})")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[kb] could not reuse FAISS cache ({exc}); rebuilding.")

        # Rebuild: embed all chunks with batched retry, then create the store.
        try:
            vectors = self._embeddings.embed_documents(chunks)
        except Exception as exc:  # noqa: BLE001
            print(f"[kb] WARNING: embedding failed ({exc}) — RAG fallback disabled.")
            return

        documents = [Document(page_content=c) for c in chunks]
        try:
            if _HAS_FAISS:
                pairs = list(zip(chunks, vectors))
                self._store = FAISS.from_embeddings(  # type: ignore[union-attr]
                    text_embeddings=pairs,
                    embedding=self._embeddings,
                )
                # persist for next run
                try:
                    config.KB_INDEX_DIR.mkdir(parents=True, exist_ok=True)
                    self._store.save_local(str(config.KB_INDEX_DIR))
                except Exception as exc:  # noqa: BLE001
                    print(f"[kb] WARNING: could not save FAISS index: {exc}")
            else:
                # InMemoryVectorStore takes pre-computed embeddings too.
                self._store = InMemoryVectorStore(self._embeddings)
                self._store.add_documents(documents)
            self._n_chunks = len(chunks)
            print(
                f"[kb] embedded {self._n_chunks} chunks "
                f"(model={config.EMBED_MODEL}, backend={'faiss' if _HAS_FAISS else 'memory'})"
            )
            try:
                _write_meta(self._n_chunks)
            except Exception as exc:  # noqa: BLE001
                print(f"[kb] WARNING: could not write meta: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[kb] WARNING: building vector store failed ({exc}) — RAG fallback disabled.")
            self._store = None

    # ----- public API --------------------------------------------------------
    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """
        Return up to `k` (chunk_text, similarity_0_to_1) pairs.

        `similarity_search_with_relevance_scores` is normalized to [0, 1] by
        LangChain (higher = better), which is exactly what the responder's
        confidence math expects.
        """
        if self._store is None or not query.strip():
            return []
        try:
            results = self._store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as exc:  # noqa: BLE001
            print(f"[kb] WARNING: similarity search failed: {exc}")
            return []
        out: list[tuple[str, float]] = []
        for doc, score in results:
            s = float(score)
            # Some backends can return negative relevance — clip to [0, 1].
            s = max(0.0, min(1.0, s))
            out.append((doc.page_content, s))
        return out


_kb_singleton: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    """Lazy singleton — first call may trigger embedding API requests."""
    global _kb_singleton
    if _kb_singleton is None:
        _kb_singleton = KnowledgeBase()
    return _kb_singleton
