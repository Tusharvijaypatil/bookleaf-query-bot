"""
Interaction logger.

Every turn through the bot ends with a log row. File logging is the
source of truth (logs/queries.jsonl); Supabase insertion is best-effort
so that a flaky network never breaks the user-facing flow.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from src import config, db


def log_interaction(
    *,
    channel: str,
    query: str,
    intent: str,
    matched_email: Optional[str],
    confidence: float,
    response: str,
    escalated: bool,
) -> None:
    """
    Append a single interaction row to both:
      1) logs/queries.jsonl  — guaranteed local trail
      2) Supabase `query_logs` table — best-effort, never raises
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel": channel,
        "user_query": query,
        "intent": intent,
        "matched_email": matched_email,
        "confidence": round(float(confidence), 4),
        "response": response,
        "escalated": bool(escalated),
    }

    # 1) local jsonl (must never raise)
    try:
        with config.QUERY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # 2) Supabase best-effort (insert_log already swallows errors)
    db.insert_log(
        {
            "channel": record["channel"],
            "user_query": record["user_query"],
            "matched_email": record["matched_email"],
            "intent": record["intent"],
            "confidence": record["confidence"],
            "response": record["response"],
            "escalated": record["escalated"],
        }
    )
