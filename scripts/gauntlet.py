"""Run a sequential Shed gauntlet and report how the agents compared.

Usage from the repository root::

    uv run scripts/gauntlet.py --agents random greedy --deals 100 --seed 42 \
        --seconds-per-turn 2 --output results/gauntlet.json

The lineup is explicit and holds two to five agents. Each deal in the bank is
played once per cyclic seat rotation -- ``--deals 100`` with two agents is 200
matches -- so every participant plays every seat on every deal. That controls
for seat advantage; it does not enumerate every seating permutation for three or
more agents, because the participants keep their cyclic order relative to each
other.

Matches run one at a time. The comparison is about wall-clock decisions, so two
matches thinking at once would measure the machine's load rather than the
strategies.

The console gets a compact table; ``--output`` gets the same numbers as JSON,
with each match's full replay embedded, so a finished match from a gauntlet file
can be verified exactly like one from ``play.py``. ``--no-replays`` drops those
embedded documents for a smaller file that can no longer be replayed.

Progress goes to standard error while the run is in flight, so a long gauntlet
says where it is without polluting the table on standard output. ``--quiet``
silences it.

This file is an argument parser and nothing else: the schedule, the seeds, the
accounting, and the report document live in :mod:`shed.gauntlet`, and the
rendering in :mod:`shed.cli`. Starting it is behind a main guard because every
decision starts a worker process, and a spawned or forkserver worker re-imports
this module.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from shed.agents import AGENT_KINDS, AgentSpec
from shed.cli import (
    agent_kinds_help,
    build_participants,
    gauntlet_progress,
    gauntlet_summary,
    positive_count,
    positive_seconds,
    restore_default_sigpipe,
)
from shed.engine import PlayerId
from shed.gauntlet import (
    GauntletConfig,
    GauntletRun,
    MatchRecord,
    build_schedule,
    run_gauntlet,
    write_gauntlet,
)
from shed.match import MatchConfig, MatchStatus
from shed.replay import detect_source_revision

FAILED_STATUSES = (MatchStatus.AGENT_FAILED, MatchStatus.ENGINE_FAILED)
"""Statuses reported with a nonzero exit; a truncated match is not a failure."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command's argument parser.

    Returns:
        The parser, with every option the gauntlet command accepts.
    """
    parser = argparse.ArgumentParser(
        prog="gauntlet.py",
        description="Compare built-in Shed agents over shared deals and rotated seats.",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        choices=AGENT_KINDS,
        metavar="KIND",
        help=f"lineup of 2-5 agents; repeated kinds get distinct labels ({agent_kinds_help()})",
    )
    parser.add_argument(
        "--deals",
        type=positive_count,
        default=10,
        help="deals in the bank; each is played once per seat rotation",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="root seed every stream is derived from"
    )
    parser.add_argument(
        "--seconds-per-turn",
        type=positive_seconds,
        default=2.0,
        help="acceptance budget per decision, in seconds",
    )
    parser.add_argument("--dealer", type=int, default=0, help="dealing seat, held fixed")
    parser.add_argument(
        "--max-play-decisions",
        type=positive_count,
        default=MatchConfig().max_play_decisions,
        help="bound on play decisions before a match is truncated",
    )
    parser.add_argument(
        "--strict-failures",
        action="store_true",
        help="abort a match on any agent failure instead of playing on",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="write the report to this JSON file"
    )
    parser.add_argument(
        "--no-replays",
        action="store_true",
        help="leave the embedded replays out of the report file, making it smaller",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="do not report progress while the run is in flight"
    )
    return parser


def _plan(args: argparse.Namespace) -> tuple[tuple[AgentSpec, ...], GauntletConfig]:
    """Turn parsed arguments into a lineup and a run configuration.

    Every decoded value is validated here, outside the engine and outside the
    gauntlet: the table size, the dealing seat, and -- inside the configuration
    itself -- the deal count and the timing settings.

    Args:
        args: The parsed arguments.

    Returns:
        The participants in lineup order, and the run configuration.

    Raises:
        ValueError: If the arguments describe no runnable gauntlet.
    """
    participants = build_participants(args.agents)
    if args.dealer not in range(len(participants)):
        raise ValueError(f"dealer {args.dealer} is not a seat in a {len(participants)}-player game")
    config = GauntletConfig(
        deals=args.deals,
        seed=args.seed,
        dealer=PlayerId(args.dealer),
        seconds_per_turn=args.seconds_per_turn,
        max_play_decisions=args.max_play_decisions,
        strict_failures=args.strict_failures,
    )
    return participants, config


def _play(
    participants: Sequence[AgentSpec],
    config: GauntletConfig,
    *,
    quiet: bool,
) -> GauntletRun:
    """Play the whole schedule, optionally narrating progress.

    Args:
        participants: The lineup, in participant order.
        config: The run's configuration.
        quiet: Whether to suppress the per-match progress lines.

    Returns:
        The completed run.
    """
    total = len(build_schedule(participants, config))

    def report(record: MatchRecord) -> None:
        """Print one completed match to standard error.

        Args:
            record: The match that just finished.
        """
        print(gauntlet_progress(record, total, participants), file=sys.stderr)

    return run_gauntlet(participants, config, on_match=None if quiet else report)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gauntlet command.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code: ``0`` when every match finished or was truncated,
        ``1`` when a match aborted or the report cannot be written, and ``2``
        for arguments that describe no runnable gauntlet.
    """
    args = build_parser().parse_args(argv)
    try:
        participants, config = _plan(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    run = _play(participants, config, quiet=args.quiet)

    # The artifact is written before anything is printed, so a console that goes
    # away -- a pipe into `head`, a closed terminal -- cannot cost a long run.
    written = None
    if args.output is not None:
        try:
            written = write_gauntlet(
                run,
                args.output,
                source_revision=detect_source_revision(),
                replays=not args.no_replays,
            )
        except OSError as error:
            print(f"error: cannot write {args.output}: {error}", file=sys.stderr)
            return 1

    print("\n".join(gauntlet_summary(run.report)))
    if written is not None:
        print(f"report: {written}")

    return 1 if any(record.result.status in FAILED_STATUSES for record in run.records) else 0


if __name__ == "__main__":
    restore_default_sigpipe()
    raise SystemExit(main())
