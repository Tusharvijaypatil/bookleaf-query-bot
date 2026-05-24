"""
Identity Unification — working demo.

Run with:
    python -m identity_unification.unify

It seeds a small set of raw contacts (across email / phone / dashboard
display name / instagram handle), runs the unification logic, and prints
a rich table of every candidate link with its confidence and bucket.

Approach in plain English:
  1. Normalize each contact (lowercase, strip handles/phones).
  2. For every pair (across distinct platforms):
       * if phone last-7 digits match -> deterministic short-circuit (0.95)
       * otherwise: weighted fuzzy blend over email-local, display name,
         and handle (rapidfuzz WRatio).
  3. Bucket: >=0.80 auto-link, 0.55–0.80 verify_manually, <0.55 reject.

This is intentionally lightweight — see the folder README for what I'd
upgrade with more time (embedding-based name match, deterministic keys,
graph-based clustering, human-in-the-loop review queue).
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz
from rich.console import Console
from rich.table import Table


# --- types --------------------------------------------------------------------

@dataclass
class Contact:
    platform: str                       # "email" | "whatsapp" | "dashboard" | "instagram"
    identifier: str                     # raw identifier as captured
    display_name: Optional[str] = None
    phone: Optional[str] = None         # any format; normalized internally
    # derived (filled by _normalize)
    email_local: Optional[str] = field(default=None, init=False)
    handle: Optional[str] = field(default=None, init=False)
    phone_suffix: Optional[str] = field(default=None, init=False)
    norm_name: Optional[str] = field(default=None, init=False)

    def _normalize(self) -> None:
        ident = (self.identifier or "").strip().lower()
        if self.platform == "email" and "@" in ident:
            self.email_local = re.sub(r"[^a-z0-9]", "", ident.split("@", 1)[0])
        elif self.platform == "instagram":
            self.handle = ident.lstrip("@")
        elif self.platform in ("whatsapp", "phone"):
            digits = re.sub(r"\D", "", ident)
            if digits:
                self.phone_suffix = digits[-7:]
        if self.phone and not self.phone_suffix:
            digits = re.sub(r"\D", "", self.phone)
            if digits:
                self.phone_suffix = digits[-7:]
        if self.display_name:
            self.norm_name = re.sub(r"[^a-z0-9 ]", "", self.display_name.strip().lower())


def _make_contact(**kwargs) -> Contact:
    c = Contact(**kwargs)
    c._normalize()
    return c


# --- scoring ------------------------------------------------------------------

def _name_signal(a: Contact, b: Contact) -> float:
    """Best fuzz score over any name-ish field each contact exposes."""
    pool_a = [s for s in (a.email_local, a.handle, a.norm_name) if s]
    pool_b = [s for s in (b.email_local, b.handle, b.norm_name) if s]
    if not pool_a or not pool_b:
        return 0.0
    best = 0
    for x in pool_a:
        for y in pool_b:
            best = max(best, fuzz.WRatio(x, y))
    return best / 100.0


def _email_local_signal(a: Contact, b: Contact) -> float:
    if a.email_local and b.email_local:
        return fuzz.WRatio(a.email_local, b.email_local) / 100.0
    # When one side has no email, fall back to a name-ish comparison
    return _name_signal(a, b)


def _handle_signal(a: Contact, b: Contact) -> float:
    if a.handle and b.handle:
        return fuzz.WRatio(a.handle, b.handle) / 100.0
    return _name_signal(a, b)


def _phones_match(a: Contact, b: Contact) -> bool:
    return bool(a.phone_suffix and b.phone_suffix and a.phone_suffix == b.phone_suffix)


def score_pair(a: Contact, b: Contact) -> float:
    """
    Score a pair of contacts.

    Deterministic short-circuits run first:
      * Same phone suffix → 0.95 (essentially a primary-key match;
        not a flat 1.0 so an LLM-checked name discrepancy can still
        knock it down in a future revision).

    Otherwise, weighted blend of fuzzy signals. Weights chosen so
    email-local-part (a strong on-platform name proxy) dominates,
    with smaller weight to display name and handle.
    """
    if _phones_match(a, b):
        return 0.95

    s = (
        0.45 * _email_local_signal(a, b)
        + 0.35 * _name_signal(a, b)
        + 0.20 * _handle_signal(a, b)
    )
    return max(0.0, min(1.0, s))


def bucket_for(score: float) -> str:
    if score >= 0.80:
        return "auto_link"
    if score >= 0.55:
        return "verify_manually"
    return "reject"


# --- demo data ----------------------------------------------------------------

def seed_contacts() -> list[Contact]:
    """The assignment's example author plus 2–3 decoys for a non-trivial run."""
    # BookLeaf has Sara's phone on file from registration, so her email contact
    # carries her phone number too. That lets the phone-suffix bonus link her
    # WhatsApp contact (which is just a phone) to her email/profile.
    return [
        # --- Sara Johnson's identities ---
        _make_contact(platform="email", identifier="sara.johnson@xyz.com", display_name="Sara Johnson", phone="+91 9876543210"),
        _make_contact(platform="whatsapp", identifier="+91 9876543210", display_name=None, phone="+91 9876543210"),
        _make_contact(platform="dashboard", identifier="sara_j_dash", display_name="Sara J.", phone="+91 9876543210"),
        _make_contact(platform="instagram", identifier="@sarapoetry23", display_name="Sara Writes"),
        # --- decoys ---
        _make_contact(platform="email", identifier="sam.jones@xyz.com", display_name="Sam Jones"),
        _make_contact(platform="instagram", identifier="@johnsonpoetry", display_name="Johnson Poetry"),
        _make_contact(platform="email", identifier="priya.nair@outlook.com", display_name="Priya Nair", phone="+91 9000011111"),
        _make_contact(platform="whatsapp", identifier="+91 9000011111", display_name=None, phone="+91 9000011111"),
    ]


# --- runner -------------------------------------------------------------------

def unify(contacts: list[Contact]) -> list[tuple[Contact, Contact, float, str]]:
    """Score every cross-platform pair. Same-platform pairs are skipped."""
    results: list[tuple[Contact, Contact, float, str]] = []
    for a, b in itertools.combinations(contacts, 2):
        if a.platform == b.platform:
            continue
        s = score_pair(a, b)
        results.append((a, b, s, bucket_for(s)))
    results.sort(key=lambda r: r[2], reverse=True)
    return results


def _print(results: list[tuple[Contact, Contact, float, str]]) -> None:
    console = Console()
    table = Table(title="Identity Unification — candidate links", show_lines=False)
    table.add_column("Contact A")
    table.add_column("Contact B")
    table.add_column("Score", justify="right")
    table.add_column("Bucket")
    bucket_colors = {"auto_link": "green", "verify_manually": "yellow", "reject": "dim"}
    for a, b, s, bkt in results:
        color = bucket_colors.get(bkt, "white")
        table.add_row(
            f"[{a.platform}] {a.identifier}" + (f" ({a.display_name})" if a.display_name else ""),
            f"[{b.platform}] {b.identifier}" + (f" ({b.display_name})" if b.display_name else ""),
            f"{s:.2f}",
            f"[{color}]{bkt}[/{color}]",
        )
    console.print(table)

    auto = [r for r in results if r[3] == "auto_link"]
    verify = [r for r in results if r[3] == "verify_manually"]
    console.print(
        f"\n[bold]Summary:[/bold] {len(auto)} auto-linked, "
        f"{len(verify)} need manual verification, "
        f"{len(results) - len(auto) - len(verify)} rejected."
    )


def main() -> None:
    contacts = seed_contacts()
    results = unify(contacts)
    _print(results)


if __name__ == "__main__":
    main()
