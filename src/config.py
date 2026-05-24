"""
Centralized configuration. Loads .env once and exposes typed constants.

LLM access goes through LangChain's `langchain-google-genai` integration,
talking to Google Gemini's free tier. The Gemini API key is read from
`LLM_API_KEY` (with `OPENAI_API_KEY` honored as a legacy fallback for
historical .env files; only the key value is reused, not OpenAI itself).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Project root = parent of /src
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = ROOT_DIR / "data"
LOGS_DIR: Final[Path] = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load .env from project root
load_dotenv(ROOT_DIR / ".env")


# --- LLM provider config ------------------------------------------------------
# We pass api_key explicitly to LangChain classes. langchain-google-genai
# would also pick up GEMINI_API_KEY / GOOGLE_API_KEY from the environment,
# but reading it ourselves removes any ambiguity about which key is in use.
_llm_api_key_env = os.getenv("LLM_API_KEY", "").strip()
_legacy_openai_key = os.getenv("OPENAI_API_KEY", "").strip()  # legacy .env fallback
LLM_API_KEY: Final[str] = _llm_api_key_env or _legacy_openai_key

CHAT_MODEL: Final[str] = os.getenv("CHAT_MODEL", "gemini-2.5-flash-lite").strip() or "gemini-2.5-flash-lite"
EMBED_MODEL: Final[str] = os.getenv("EMBED_MODEL", "gemini-embedding-001").strip() or "gemini-embedding-001"

# --- Supabase -----------------------------------------------------------------
SUPABASE_URL: Final[str] = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY: Final[str] = os.getenv("SUPABASE_KEY", "").strip()

# --- Tunables -----------------------------------------------------------------
ESCALATION_THRESHOLD: Final[float] = float(os.getenv("ESCALATION_THRESHOLD", "0.80"))

# RAG paths — FAISS persists as a directory; the meta sidecar tracks
# (mtime, size, embed_model) so a model swap invalidates the cache.
KB_PATH: Final[Path] = DATA_DIR / "knowledge_base.md"
KB_INDEX_DIR: Final[Path] = DATA_DIR / "kb_faiss"
KB_META_PATH: Final[Path] = DATA_DIR / "kb_meta.json"

# Log paths
QUERY_LOG_PATH: Final[Path] = LOGS_DIR / "queries.jsonl"
HUMAN_QUEUE_PATH: Final[Path] = LOGS_DIR / "human_queue.jsonl"


def require_env(strict: bool = True) -> None:
    """
    Verify mandatory env vars are present. Called from entry points so missing
    config produces a clear error rather than a crash deep in the SDK.
    """
    missing = []
    if not LLM_API_KEY:
        missing.append("LLM_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")

    if missing and strict:
        print(
            "[config] Missing required environment variables: "
            + ", ".join(missing)
            + "\n[config] Copy .env.example to .env and fill in the values.",
            file=sys.stderr,
        )
        sys.exit(1)
