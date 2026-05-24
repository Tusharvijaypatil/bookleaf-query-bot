"""
Human-handoff plumbing.

If the orchestrator decides confidence is too low (or hits an error
fork like "DB down"), it calls `escalate()` which:
  - returns a friendly handoff message for the user
  - appends an internal record to logs/human_queue.jsonl so an agent
    can pick it up later
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from src import config


HANDOFF_MESSAGE = (
    "I'm not fully sure on this one, so I'm connecting you with a human "
    "agent who'll follow up shortly. In the meantime, can you share your "
    "registered email and book title so we can pull up your account?"
)


def escalate(
    query: str,
    reason: str,
    confidence: float,
    channel: str = "cli",
    matched_email: Optional[str] = None,
    threshold: float = config.ESCALATION_THRESHOLD,
) -> str:
    """
    Drop a record onto the human queue and return the user-facing message.

    `reason` is short free-text used by the agent triaging the queue
    (e.g. "low_confidence", "db_unavailable", "no_match_clarify").
    `threshold` is recorded for traceability — useful when tuning it later.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel": channel,
        "query": query,
        "matched_email": matched_email,
        "confidence": round(float(confidence), 4),
        "threshold": threshold,
        "reason": reason,
    }
    try:
        with config.HUMAN_QUEUE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never raise. If disk is full / permissions are odd,
        # we still return the handoff message so the user sees something.
        pass

    return HANDOFF_MESSAGE
