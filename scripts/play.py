"""Play one timed Shed match and, when asked, save its replay.

Usage from the repository root::

    uv run scripts/play.py --agents random greedy --seed 42 --output results/match.json

The console shows the public actions and the final outcome. Hidden information
-- the dealt hands, every replenishment draw, the face-down identities -- stays
out of it: private events are reported by count. The saved replay is the
opposite, a complete trusted artifact, which is why it is written only where the
caller asks for it.

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
    agent_kinds_help,
    build_lineup,
    match_summary,
    narrate,
    positive_count,
    positive_seconds,
)
from shed.engine import PlayerId
from shed.match import MatchConfig, MatchRunner, MatchStatus
from shed.replay import detect_source_revision, write_match

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
        "--quiet", action="store_true", help="print the summary without the public actions"
    )
    return parser


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

    if not args.quiet:
        events = list(result.initial_events)
        for decision in result.decisions:
            events.extend(decision.events)
        print("\n".join(narrate(events)))
        print()
    print("\n".join(match_summary(result)))

    if args.output is not None:
        try:
            written = write_match(result, args.output, source_revision=detect_source_revision())
        except OSError as error:
            print(f"error: cannot write {args.output}: {error}", file=sys.stderr)
            return 1
        print(f"replay: {written}")

    return 1 if result.status in FAILED_STATUSES else 0


if __name__ == "__main__":
    raise SystemExit(main())
