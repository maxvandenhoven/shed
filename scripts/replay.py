"""Summarize or verify a saved Shed replay.

Usage from the repository root::

    uv run scripts/replay.py results/match.json --verify

Reading a replay is the untrusted direction: the file is decoded and validated
by :mod:`shed.replay`, which rejects an unsupported schema or rules profile, a
malformed shape, or a field of the wrong primitive type. ``--verify`` then deals
the recorded deck and applies the recorded decisions through the engine,
comparing events, outcome, and final position. No agent is built and no worker
is started, so verification is deterministic and does not depend on the timing
the match was played under.

The console summary stays public: it reports private events by count, even
though the file it reads holds their identities.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from shed.cli import narrate, replay_summary
from shed.replay import Replay, ReplayFormatError, read_replay, verify_replay


def build_parser() -> argparse.ArgumentParser:
    """Build the command's argument parser.

    Returns:
        The parser, with every option the replay command accepts.
    """
    parser = argparse.ArgumentParser(
        prog="replay.py",
        description="Summarize a saved Shed replay, or replay it to verify it.",
    )
    parser.add_argument("path", type=Path, help="replay JSON file to read")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="replay the recorded decisions and compare them with the recording",
    )
    parser.add_argument(
        "--events", action="store_true", help="also print the recorded public actions"
    )
    return parser


def _print_events(replay: Replay) -> None:
    """Print the recorded match as public commentary.

    Args:
        replay: The decoded replay. Its private events are printed by count
            only, so the identities the file holds are not dumped.
    """
    events = list(replay.initial_events)
    for decision in replay.decisions:
        events.extend(decision.events)
    print("\n".join(narrate(events)))
    print()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the replay command.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code: ``0`` when the file was read and any requested
        verification passed, ``1`` when verification failed, and ``2`` when the
        file cannot be read or is not a replay this release supports.
    """
    args = build_parser().parse_args(argv)
    try:
        replay = read_replay(args.path)
    except ReplayFormatError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: cannot read {args.path}: {error}", file=sys.stderr)
        return 2

    if args.events:
        _print_events(replay)
    print("\n".join(replay_summary(replay)))

    if not args.verify:
        return 0

    check = verify_replay(replay)
    if not check.ok:
        print(f"verification failed after {check.applied} decisions:", file=sys.stderr)
        for problem in check.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"verified: {check.applied} recorded decisions reproduce this replay exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
