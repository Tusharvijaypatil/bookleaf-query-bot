"""
The orchestrator. One function — `answer(query, channel)` — owns the
whole pipeline:

  1. Classify the query (LangChain structured output → intent / target /
     extracted email & title).
  2. If we should look in the DB, do so. Branch on 0 / 1 / many matches.
  3. If we should look in the KB (or the DB lookup was weak), run the
     LangChain FAISS retriever.
  4. Generate the final natural-language answer via an LCEL chain:
        ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser
     The prompt instructs the model to use ONLY the retrieved DB record
     and/or KB context — no inventing.
  5. Compute final_confidence and either return the answer or escalate.
  6. Log the interaction.

The function ALWAYS returns an `AnswerResult` — even on errors. Callers
never have to handle exceptions from this layer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src import classifier, config, db, escalation, logger
from src.knowledge_base import get_kb


@dataclass
class AnswerResult:
    """Everything a UI might want to display about one turn."""

    response: str
    intent: str
    target: str
    source: str                              # "db" | "kb" | "db+kb" | "none"
    matched_email: Optional[str]
    confidence: float
    escalated: bool
    db_confidence: float = 0.0
    kb_confidence: float = 0.0
    llm_confidence: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)


# --- LCEL answer chain --------------------------------------------------------
_ANSWER_SYSTEM = """You are BookLeaf's friendly author-support assistant.
Write a concise, warm reply (2-5 sentences max) to the author's question.

STRICT RULES:
- Use ONLY the facts in the "AUTHOR RECORD" and "KNOWLEDGE BASE" sections.
  Do NOT invent dates, ISBNs, royalty amounts, or policy details.
- If the provided facts don't answer the question, say so honestly and offer
  to connect them with a human agent.
- Refer to dates in plain English (e.g. "March 14, 2025").
- Never mention the words "database", "RAG", "classifier", or any internal mechanic.
- Do not start with "Dear" or any letter-style salutation; this is a chat reply.
"""

_ANSWER_USER_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "AUTHOR RECORD:\n{author_record}\n\n"
    "KNOWLEDGE BASE:\n{kb_context}\n"
)

_answer_chain = None  # type: ignore[var-annotated]


def _get_answer_chain():
    """Build the LCEL chain once: prompt | llm | parser."""
    global _answer_chain
    if _answer_chain is None:
        llm = ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL,
            api_key=config.LLM_API_KEY,
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _ANSWER_SYSTEM),
                ("human", _ANSWER_USER_TEMPLATE),
            ]
        )
        _answer_chain = prompt | llm | StrOutputParser()
    return _answer_chain


# --- formatters ---------------------------------------------------------------

def _format_author_record(row: dict[str, Any]) -> str:
    keys = [
        ("email", "Email"),
        ("book_title", "Book title"),
        ("final_submission_date", "Final submission date"),
        ("book_live_date", "Book live date"),
        ("royalty_status", "Royalty status"),
        ("isbn", "ISBN"),
        ("add_on_services", "Add-on services"),
    ]
    lines = []
    for k, label in keys:
        v = row.get(k)
        if v is None or v == "":
            v = "not set"
        lines.append(f"- {label}: {v}")
    return "\n".join(lines)


def _format_kb_chunks(chunks: list[tuple[str, float]]) -> str:
    if not chunks:
        return "(none)"
    return "\n\n".join(f"[score={s:.2f}]\n{c}" for c, s in chunks)


# --- confidence formula -------------------------------------------------------

def _compute_final_confidence(
    target: str,
    llm_confidence: float,
    db_confidence: float,
    kb_confidence: float,
) -> float:
    """
    Blend the three signals into a single 0..1 final confidence.

    Formula (chosen to be easy to reason about, not optimal):
      grounding       = max(db_confidence, kb_confidence)
      final_confidence = 0.4 * llm_confidence + 0.6 * grounding
      if both DB and KB hit:           final += 0.05  (capped at 1.0)
      if classifier said target=unknown: final = min(final, 0.4)
    """
    grounding = max(db_confidence, kb_confidence)
    base = 0.4 * llm_confidence + 0.6 * grounding
    if db_confidence > 0 and kb_confidence > 0:
        base = min(1.0, base + 0.05)
    if target == "unknown":
        base = min(base, 0.4)
    return max(0.0, min(1.0, base))


# --- DB lookup wrapper --------------------------------------------------------

@dataclass
class _DBResult:
    rows: list[dict[str, Any]]
    unavailable: bool = False


def _do_db_lookup(c: classifier.Classification) -> _DBResult:
    try:
        rows = db.find_authors(
            extracted_email=c.extracted_email,
            extracted_title=c.extracted_book_title,
        )
        return _DBResult(rows=rows, unavailable=False)
    except db.DatabaseUnavailable:
        return _DBResult(rows=[], unavailable=True)


# --- main entry point --------------------------------------------------------

def answer(query: str, channel: str = "cli") -> AnswerResult:
    """End-to-end pipeline. Always returns an AnswerResult."""
    # 1) Classify
    c = classifier.classify(query)

    db_rows: list[dict[str, Any]] = []
    db_confidence = 0.0
    db_unavailable = False

    needs_db = c.target in ("db", "both")
    needs_kb = c.target in ("knowledge_base", "both")

    # 2) DB lookup (if relevant)
    if needs_db:
        res = _do_db_lookup(c)
        db_unavailable = res.unavailable
        db_rows = res.rows

        if db_unavailable:
            msg = escalation.escalate(
                query=query,
                reason="db_unavailable",
                confidence=0.0,
                channel=channel,
                matched_email=c.extracted_email,
            )
            response = (
                "Sorry — our author records system is temporarily unavailable, "
                "so I can't pull up your account right now. " + msg
            )
            logger.log_interaction(
                channel=channel,
                query=query,
                intent=c.intent,
                matched_email=c.extracted_email,
                confidence=0.0,
                response=response,
                escalated=True,
            )
            return AnswerResult(
                response=response,
                intent=c.intent,
                target=c.target,
                source="none",
                matched_email=c.extracted_email,
                confidence=0.0,
                escalated=True,
                llm_confidence=c.llm_confidence,
                debug={"reason": "db_unavailable"},
            )

        # MULTIPLE MATCHES → ask the user to disambiguate
        if len(db_rows) > 1:
            titles = ", ".join(f'"{r.get("book_title")}"' for r in db_rows[:5])
            response = (
                f"I found {len(db_rows)} books that could match — {titles}. "
                "Which one is yours? (You can reply with the exact title or your registered email.)"
            )
            logger.log_interaction(
                channel=channel,
                query=query,
                intent=c.intent,
                matched_email=None,
                confidence=0.5,
                response=response,
                escalated=False,
            )
            return AnswerResult(
                response=response,
                intent=c.intent,
                target=c.target,
                source="db",
                matched_email=None,
                confidence=0.5,
                escalated=False,
                llm_confidence=c.llm_confidence,
                db_confidence=0.5,
                debug={"reason": "multiple_matches", "n_matches": len(db_rows)},
            )

        # NO MATCH → ask for a clarifying detail. Trigger RAG too if "both".
        if len(db_rows) == 0:
            if c.extracted_email and not needs_kb:
                response = (
                    f"I couldn't find an account for {c.extracted_email}. "
                    "Could you double-check the email you registered with, or share your book title?"
                )
                logger.log_interaction(
                    channel=channel,
                    query=query,
                    intent=c.intent,
                    matched_email=None,
                    confidence=0.4,
                    response=response,
                    escalated=False,
                )
                return AnswerResult(
                    response=response,
                    intent=c.intent,
                    target=c.target,
                    source="db",
                    matched_email=None,
                    confidence=0.4,
                    escalated=False,
                    llm_confidence=c.llm_confidence,
                    db_confidence=0.0,
                    debug={"reason": "no_db_match"},
                )
            # otherwise fall through and let KB try

        # SINGLE MATCH → score by match kind
        if len(db_rows) == 1:
            kind = db_rows[0].get("_match_kind", "")
            db_confidence = 1.0 if kind == "email_exact" else max(0.7, float(db_rows[0].get("_fuzzy_score", 0.7)))

    # 3) KB search — when target asked for it, OR when DB confidence is low
    kb_results: list[tuple[str, float]] = []
    kb_confidence = 0.0
    if needs_kb or (needs_db and db_confidence < 0.7):
        try:
            kb_results = get_kb().search(query, k=3)
            if kb_results:
                kb_confidence = max(0.0, min(1.0, kb_results[0][1]))
        except Exception as exc:  # noqa: BLE001
            print(f"[responder] KB search failed: {exc}")
            kb_results = []
            kb_confidence = 0.0

    # 4) Generate grounded answer via the LCEL chain
    author_block = "(none)"
    if db_rows and len(db_rows) == 1:
        author_block = _format_author_record(db_rows[0])
    kb_block = _format_kb_chunks(kb_results)

    try:
        generated = _get_answer_chain().invoke(
            {
                "question": query,
                "author_record": author_block,
                "kb_context": kb_block,
            }
        )
        generated = (generated or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"[responder] answer generation failed: {exc}")
        generated = ""

    if not generated:
        msg = escalation.escalate(
            query=query,
            reason="generation_failed",
            confidence=0.0,
            channel=channel,
            matched_email=(db_rows[0]["email"] if db_rows else c.extracted_email),
        )
        logger.log_interaction(
            channel=channel,
            query=query,
            intent=c.intent,
            matched_email=(db_rows[0]["email"] if db_rows else c.extracted_email),
            confidence=0.0,
            response=msg,
            escalated=True,
        )
        return AnswerResult(
            response=msg,
            intent=c.intent,
            target=c.target,
            source="none",
            matched_email=(db_rows[0]["email"] if db_rows else c.extracted_email),
            confidence=0.0,
            escalated=True,
            llm_confidence=c.llm_confidence,
            debug={"reason": "generation_failed"},
        )

    # 5) Final confidence + 6) escalate if below threshold
    final_conf = _compute_final_confidence(
        target=c.target,
        llm_confidence=c.llm_confidence,
        db_confidence=db_confidence,
        kb_confidence=kb_confidence,
    )

    escalated = final_conf < config.ESCALATION_THRESHOLD
    matched_email = db_rows[0]["email"] if db_rows else c.extracted_email
    source = (
        "db+kb" if (db_rows and kb_results)
        else "db" if db_rows
        else "kb" if kb_results
        else "none"
    )

    if escalated:
        handoff = escalation.escalate(
            query=query,
            reason=f"low_confidence ({final_conf:.2f})",
            confidence=final_conf,
            channel=channel,
            matched_email=matched_email,
        )
        response = f"{generated}\n\n{handoff}"
    else:
        response = generated

    logger.log_interaction(
        channel=channel,
        query=query,
        intent=c.intent,
        matched_email=matched_email,
        confidence=final_conf,
        response=response,
        escalated=escalated,
    )

    return AnswerResult(
        response=response,
        intent=c.intent,
        target=c.target,
        source=source,
        matched_email=matched_email,
        confidence=final_conf,
        escalated=escalated,
        db_confidence=db_confidence,
        kb_confidence=kb_confidence,
        llm_confidence=c.llm_confidence,
        debug={"classification": json.dumps(c.raw)[:300]},
    )
