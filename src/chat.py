"""
Rich-based chat CLI — the primary user interface.

Examples:
  python -m src.chat                       # interactive chat
  python -m src.chat --debug               # show intent | source | confidence | escalated
  python -m src.chat --channel whatsapp    # simulate a different channel for logging
  python -m src.chat --demo                # scripted demo run for screenshots / video
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src import config, responder


console = Console()


# Scripted demo queries — they cover every branch:
#   * exact-email DB hit (live book, pending royalty, author copy)
#   * KB-only general info
#   * vague low-confidence query → escalation
#   * wrong email → no-match clarification
DEMO_QUERIES: list[tuple[str, str]] = [
    ("cli",      "Is my book live yet? my email is sara.johnson@xyz.com"),
    ("email",    "When will I get my royalty? amit.verma@gmail.com"),
    ("whatsapp", "Where's my author copy? priya.nair@outlook.com"),
    ("cli",      "Tell me about your publishing process and timelines"),
    ("cli",      "uhh what about the thing"),
    ("instagram","Has my book gone live? My email is unknown.person@nowhere.com"),
]


def _render_turn(result: responder.AnswerResult, debug: bool) -> None:
    """Print the bot's answer in a panel, plus an optional debug line."""
    body = Markdown(result.response)
    style = "yellow" if result.escalated else "green"
    title = "BookLeaf Bot (escalated)" if result.escalated else "BookLeaf Bot"
    console.print(Panel(body, title=title, border_style=style))
    if debug:
        dbg = Text()
        dbg.append("  intent=", style="dim")
        dbg.append(result.intent or "-", style="cyan")
        dbg.append(" | source=", style="dim")
        dbg.append(result.source or "-", style="cyan")
        dbg.append(" | conf=", style="dim")
        dbg.append(f"{result.confidence:.2f}", style="magenta")
        dbg.append(" | escalated=", style="dim")
        dbg.append("yes" if result.escalated else "no", style="red" if result.escalated else "green")
        dbg.append(f" | db={result.db_confidence:.2f} kb={result.kb_confidence:.2f} llm={result.llm_confidence:.2f}", style="dim")
        console.print(dbg)


def run_demo(debug: bool) -> None:
    console.print(Panel.fit(
        "[bold]BookLeaf Customer Query Bot — DEMO MODE[/bold]\n"
        "Running scripted queries that exercise every branch of the pipeline.",
        border_style="cyan",
    ))
    for channel, query in DEMO_QUERIES:
        console.print()
        console.print(f"[bold blue]>[/bold blue] [dim]({channel})[/dim] {query}")
        try:
            result = responder.answer(query, channel=channel)
        except Exception as exc:  # noqa: BLE001
            # The orchestrator is supposed to never raise. If it does, we want
            # the demo to keep going and tell the user something went wrong.
            console.print(Panel(f"Unexpected error: {exc}", border_style="red"))
            continue
        _render_turn(result, debug=debug)


def run_chat(channel: str, debug: bool) -> None:
    console.print(Panel.fit(
        "[bold]BookLeaf Customer Query Bot[/bold]\n"
        f"channel: [cyan]{channel}[/cyan]  •  type [cyan]exit[/cyan] to quit",
        border_style="cyan",
    ))
    while True:
        try:
            query = Prompt.ask("[bold blue]you[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye![/dim]")
            return
        if not query.strip():
            continue
        if query.strip().lower() in {"exit", "quit", ":q"}:
            console.print("[dim]bye![/dim]")
            return
        try:
            result = responder.answer(query, channel=channel)
        except Exception as exc:  # noqa: BLE001
            console.print(Panel(f"Unexpected error: {exc}", border_style="red"))
            continue
        _render_turn(result, debug=debug)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BookLeaf customer query bot — CLI.")
    p.add_argument(
        "--channel",
        choices=["cli", "email", "whatsapp", "instagram"],
        default="cli",
        help="Simulated channel for logging (default: cli)",
    )
    p.add_argument("--debug", action="store_true", help="Show intent/source/confidence after each reply.")
    p.add_argument("--demo", action="store_true", help="Run scripted sample queries end-to-end and exit.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    config.require_env(strict=True)

    if args.demo:
        run_demo(debug=args.debug or True)  # demo always shows debug for clarity
    else:
        run_chat(channel=args.channel, debug=args.debug)


if __name__ == "__main__":
    main(sys.argv[1:])
