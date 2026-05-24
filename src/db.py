"""
Thin wrapper around the Supabase client.

Responsibilities:
  - Build the client lazily (so import-time doesn't crash if env is bad).
  - Provide three retrieval modes the orchestrator needs:
      * exact email lookup
      * fuzzy title lookup (rapidfuzz across all rows)
      * combined finder that returns 0 / 1 / many semantics
  - Always wrap network calls so transport failures surface as
    `DatabaseUnavailable` — the orchestrator uses this to switch to the
    DB-down fallback path.
  - `insert_log` is best-effort: it must never raise.
"""
from __future__ import annotations

from typing import Any, Optional

from rapidfuzz import fuzz, process
from supabase import Client, create_client

from src import config


class DatabaseUnavailable(Exception):
    """Raised when Supabase is unreachable or returns a transport-level error."""


_client: Optional[Client] = None


def _normalize_supabase_url(raw: str) -> str:
    """
    `supabase-py` expects the bare project URL (e.g.
    `https://abc.supabase.co`) and appends `/rest/v1/...` itself. If a user
    pastes the full REST URL into their .env, every request ends up at
    `/rest/v1/rest/v1/...` and Supabase rejects it with PGRST125. Strip
    any `/rest/v1` path segment and trailing slashes defensively.
    """
    url = raw.strip().rstrip("/")
    if url.lower().endswith("/rest/v1"):
        url = url[: -len("/rest/v1")]
    return url


def _get_client() -> Client:
    """Lazy singleton so importing this module never makes network calls."""
    global _client
    if _client is None:
        try:
            _client = create_client(_normalize_supabase_url(config.SUPABASE_URL), config.SUPABASE_KEY)
        except Exception as exc:  # noqa: BLE001 — surface as our custom type
            raise DatabaseUnavailable(f"Could not init Supabase client: {exc}") from exc
    return _client


def _safe_select_all() -> list[dict[str, Any]]:
    """
    Pull all authors. This DB is tiny (mock data), so a full scan is fine
    and lets us do fuzzy title matching client-side without extra infra.
    """
    try:
        resp = _get_client().table("authors").select("*").execute()
        return resp.data or []
    except DatabaseUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DatabaseUnavailable(f"Supabase select failed: {exc}") from exc


def get_author_by_email(email: str) -> Optional[dict[str, Any]]:
    """Exact (case-insensitive) email lookup. Returns one row or None."""
    if not email:
        return None
    email_norm = email.strip().lower()
    try:
        resp = (
            _get_client()
            .table("authors")
            .select("*")
            .ilike("email", email_norm)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None
    except DatabaseUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DatabaseUnavailable(f"Supabase email lookup failed: {exc}") from exc


def _fuzzy_title_matches(
    rows: list[dict[str, Any]],
    title: str,
    score_cutoff: int = 70,
) -> list[dict[str, Any]]:
    """
    Use rapidfuzz to find rows whose book_title is similar to `title`.
    Returns rows ranked by descending similarity; only those above
    `score_cutoff` (0–100 scale) are returned.
    """
    if not title or not rows:
        return []
    title_norm = title.strip().lower()
    # Build a parallel list of titles for indexed scoring
    titles = [(r.get("book_title") or "").strip().lower() for r in rows]
    scored = process.extract(
        title_norm,
        titles,
        scorer=fuzz.WRatio,
        limit=len(titles),
    )
    matches: list[dict[str, Any]] = []
    for _value, score, idx in scored:
        if score >= score_cutoff:
            row = dict(rows[idx])
            row["_fuzzy_score"] = score / 100.0  # normalize to 0–1
            matches.append(row)
    return matches


def find_authors(
    extracted_email: Optional[str] = None,
    extracted_title: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Combined finder used by the orchestrator.

    Strategy:
      1. If we have an email, do an exact lookup first — that's the strongest
         signal.
      2. Otherwise, if we have a title, fuzzy-match across all rows.
      3. Return a list so callers can branch on len(): 0 (no match),
         1 (single match), >1 (ambiguous — needs user to disambiguate).
    """
    if extracted_email:
        row = get_author_by_email(extracted_email)
        if row:
            row["_match_kind"] = "email_exact"
            return [row]

    if extracted_title:
        all_rows = _safe_select_all()
        matches = _fuzzy_title_matches(all_rows, extracted_title)
        for m in matches:
            m["_match_kind"] = "title_fuzzy"
        return matches

    return []


def insert_log(record: dict[str, Any]) -> None:
    """
    Best-effort write to `query_logs`. Never raises — the file logger is
    the source of truth and will keep the trail intact if Supabase is down.
    """
    try:
        _get_client().table("query_logs").insert(record).execute()
    except Exception:
        # Intentional swallow: file logger has us covered.
        return
