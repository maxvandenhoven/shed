"""Play one timed Shed match and, when asked, save its replay.

Usage from the repository root::

    uv run scripts/play.py --agents random greedy --seed 42 --output results/match.json

The console shows the public actions and the final outcome. Hidden information
-- the dealt hands, every replenishment draw, the face-down identities -- stays
out of it by default: private events are reported by count. ``--omniscient``
opts into the operator view instead, printing those identities and the whole
final position, which is how you read back what an agent was actually holding
when it decided. It changes only what is printed: the agents in this very match
were given filtered observations regardless. The saved replay is a complete
trusted artifact either way, which is why it is written only where the caller
asks for it.

``--quiet`` and ``--omniscient`` are orthogonal. ``--quiet`` drops the action
log; ``--omniscient`` unredacts it and adds the final position. Together they
print the summary and the final position and nothing else.

This file is an argument parser and nothing else. Lineups, narration, and
summaries live in :mod:`shed.cli`, and the match itself is run by
:class:`~shed.match.MatchRunner`. Starting it is behind a main guard because
every decision starts a worker process, and a spawned or forkserver worker
re-imports this module.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from shed.agents import AGENT_KINDS
from shed.cli import (
    ConsoleStyle,
    agent_kinds_help,
    build_lineup,
    console_style,
    describe_position,
    match_summary,
    narrate,
    positive_count,
    positive_seconds,
    restore_default_sigpipe,
)
from shed.engine import PlayerId
from shed.match import MatchConfig, MatchResult, MatchRunner, MatchStatus
from shed.replay import detect_source_revision, match_deck, write_match

FAILED_STATUSES = (MatchStatus.AGENT_FAILED, MatchStatus.ENGINE_FAILED)
"""Statuses reported with a nonzero exit; a truncated match is not a failure."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command's argument parser.

    Returns:
        The parser, with every option the play command accepts.
    """
    parser = argparse.ArgumentParser(
        prog="play.py",
        description="Play one timed Shed match between built-in agents.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        choices=AGENT_KINDS,
        metavar="KIND",
        help=f"lineup in seat order, one kind per seat ({agent_kinds_help()})",
    )
    parser.add_argument("--seed", type=int, default=0, help="deck seed for the deal")
    parser.add_argument(
        "--seconds-per-turn",
        type=positive_seconds,
        default=2.0,
        help="acceptance budget per decision, in seconds",
    )
    parser.add_argument("--dealer", type=int, default=0, help="dealing seat")
    parser.add_argument(
        "--max-play-decisions",
        type=positive_count,
        default=MatchConfig().max_play_decisions,
        help="bound on play decisions before the match is truncated",
    )
    parser.add_argument("--agent-seed", type=int, default=1, help="seed of the agent-seed stream")
    parser.add_argument(
        "--fallback-seed", type=int, default=0, help="seed of the fallback-move stream"
    )
    parser.add_argument(
        "--strict-failures",
        action="store_true",
        help="abort the match on any agent failure instead of playing on",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="write the replay to this JSON file"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print the summary without the action log"
    )
    parser.add_argument(
        "--omniscient",
        action="store_true",
        help="show hidden identities in the action log and print the final position",
    )
    parser.add_argument(
        "--show-suit",
        action="store_true",
        help="spell cards with their suit (Jc) instead of the rank alone (J)",
    )
    return parser


def _report(result: MatchResult, style: ConsoleStyle, *, quiet: bool) -> None:
    """Print one match to the console.

    Args:
        result: The match to report.
        style: How much to show, and how to spell a card.
        quiet: Whether to leave out the action log. The position block and the
            summary are printed either way.
    """
    if not quiet:
        events = list(result.initial_events)
        for decision in result.decisions:
            events.extend(decision.events)
        print("\n".join(narrate(events, style)))
        print()
    if style.omniscient:
        deck = match_deck(result.metadata)
        print("\n".join(describe_position(result.final_position, deck, style)))
        print()
    print("\n".join(match_summary(result)))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the play command.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code: ``0`` for a match that finished or was
        truncated, ``1`` for an aborted match or an unwritable output file, and
        ``2`` for arguments that describe no runnable match.
    """
    args = build_parser().parse_args(argv)
    try:
        lineup = build_lineup(args.agents)
        if args.dealer not in range(len(lineup)):
            raise ValueError(f"dealer {args.dealer} is not a seat in a {len(lineup)}-player game")
        config = MatchConfig(
            seconds_per_turn=args.seconds_per_turn,
            max_play_decisions=args.max_play_decisions,
            fallback_seed=args.fallback_seed,
            agent_seed=args.agent_seed,
            strict_failures=args.strict_failures,
        )
        runner = MatchRunner(lineup, config)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    result = runner.run(deal_seed=args.seed, dealer=PlayerId(args.dealer))

    # The artifact is written before anything is printed, so a console that goes
    # away -- a pipe into `head`, a closed terminal -- cannot cost the replay.
    written = None
    if args.output is not None:
        try:
            written = write_match(result, args.output, source_revision=detect_source_revision())
        except OSError as error:
            print(f"error: cannot write {args.output}: {error}", file=sys.stderr)
            return 1

    style = console_style(omniscient=args.omniscient, show_suits=args.show_suit)
    _report(result, style, quiet=args.quiet)
    if written is not None:
        print(f"replay: {written}")

    return 1 if result.status in FAILED_STATUSES else 0


if __name__ == "__main__":
    restore_default_sigpipe()
    raise SystemExit(main())
