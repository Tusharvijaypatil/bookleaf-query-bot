"""
LangChain-powered intent classifier.

Uses `ChatGoogleGenerativeAI.with_structured_output(ClassificationOutput)`
so the LLM returns a Pydantic-validated object — no manual JSON parsing,
no retry loop on malformed output. If the structured call fails (network
error, quota, etc.), we fall back to a safe `other/unknown` classification
so the orchestrator can still respond politely.

The downstream code consumes a `Classification` dataclass with the same
fields the old implementation produced, so this module is a drop-in
replacement.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src import config


# --- enums --------------------------------------------------------------------
VALID_INTENTS = (
    "book_status",
    "royalty",
    "dashboard_access",
    "addon_status",
    "book_sales",
    "author_copy_shipping",
    "general_info",
    "other",
)
VALID_TARGETS = ("db", "knowledge_base", "both", "unknown")


# --- structured-output schema -------------------------------------------------
class ClassificationOutput(BaseModel):
    """Pydantic schema enforced by Gemini via LangChain's `with_structured_output`."""

    intent: Literal[
        "book_status",
        "royalty",
        "dashboard_access",
        "addon_status",
        "book_sales",
        "author_copy_shipping",
        "general_info",
        "other",
    ] = Field(description="The author's intent.")
    target: Literal["db", "knowledge_base", "both", "unknown"] = Field(
        description=(
            "Where to look for the answer: 'db' for personal account questions, "
            "'knowledge_base' for generic policy questions, 'both' when both are "
            "needed, 'unknown' for vague/off-topic queries."
        )
    )
    extracted_email: Optional[str] = Field(
        default=None, description="The author's email if present in the message, else null."
    )
    extracted_book_title: Optional[str] = Field(
        default=None, description="The book title if mentioned in the message, else null."
    )
    llm_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Classification certainty between 0 and 1 — NOT certainty that the bot "
            "can answer. Vague queries should score <= 0.3."
        ),
    )
    reasoning: str = Field(description="One short sentence explaining the classification.")


# --- downstream-friendly dataclass --------------------------------------------
@dataclass
class Classification:
    """Plain dataclass the orchestrator consumes."""

    intent: str = "other"
    target: str = "unknown"
    extracted_email: Optional[str] = None
    extracted_book_title: Optional[str] = None
    llm_confidence: float = 0.0
    reasoning: str = ""
    raw: dict = field(default_factory=dict)


_SYSTEM_PROMPT = """You are the intent classifier for BookLeaf, a book-publishing house's customer-query bot.
Authors ask questions about their book's status, royalties, dashboard access, add-on services,
sales, author copies, or generic info about the publisher.

Rules:
- Personal/account questions ("is MY book live", "MY royalty", "where is MY author copy") -> target = "db".
- Generic info questions ("what's your publishing process", "how do royalties work") -> target = "knowledge_base".
- Personal + needs policy context -> target = "both".
- Vague / unclear / off-topic -> intent = "other", target = "unknown", llm_confidence <= 0.3.
- llm_confidence should reflect classification certainty, NOT certainty that the bot can answer.
"""


_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def _fallback_email_extract(text: str) -> Optional[str]:
    """Belt-and-braces email regex in case the LLM misses one in the text."""
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None


# --- lazy singleton -----------------------------------------------------------
_classifier_chain = None  # type: ignore[var-annotated]


def _get_classifier():
    """
    Build the structured-output chain once. We bind the Pydantic schema with
    `with_structured_output`; LangChain takes care of telling Gemini to emit a
    matching JSON object and parses the response into our Pydantic model.
    """
    global _classifier_chain
    if _classifier_chain is None:
        llm = ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL,
            api_key=config.LLM_API_KEY,
            temperature=0.0,
        )
        _classifier_chain = llm.with_structured_output(ClassificationOutput)
    return _classifier_chain


def _coerce(out: ClassificationOutput, raw_query: str) -> Classification:
    """Convert the Pydantic output to the orchestrator's dataclass, with light cleanup."""
    email = out.extracted_email
    if not isinstance(email, str) or "@" not in email:
        email = _fallback_email_extract(raw_query)
    if isinstance(email, str):
        email = email.strip().lower() or None

    title = out.extracted_book_title
    if isinstance(title, str):
        title = title.strip() or None

    return Classification(
        intent=out.intent,
        target=out.target,
        extracted_email=email,
        extracted_book_title=title,
        llm_confidence=float(max(0.0, min(1.0, out.llm_confidence))),
        reasoning=(out.reasoning or "")[:300],
        raw=out.model_dump(),
    )


def classify(query: str) -> Classification:
    """
    Classify a user query. Returns a safe fallback Classification if the LLM
    is unreachable or refuses to produce a valid structured output.
    """
    try:
        chain = _get_classifier()
        out = chain.invoke(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", query),
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[classifier] WARNING: LLM call failed ({exc}). Returning fallback classification.")
        return Classification(
            intent="other",
            target="unknown",
            extracted_email=_fallback_email_extract(query),
            extracted_book_title=None,
            llm_confidence=0.0,
            reasoning="classifier fallback (LLM error)",
            raw={},
        )

    if not isinstance(out, ClassificationOutput):
        # Defensive: with_structured_output should always return our type, but
        # if a provider hiccup makes it return a dict, coerce it.
        try:
            out = ClassificationOutput.model_validate(out)
        except Exception as exc:  # noqa: BLE001
            print(f"[classifier] WARNING: structured output validation failed ({exc}). Returning fallback.")
            return Classification(
                intent="other",
                target="unknown",
                extracted_email=_fallback_email_extract(query),
                extracted_book_title=None,
                llm_confidence=0.0,
                reasoning="classifier fallback (validation error)",
                raw={},
            )

    return _coerce(out, query)
