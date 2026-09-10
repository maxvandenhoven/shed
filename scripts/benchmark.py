"""Measure what the Shed engine costs, on representative positions.

Usage from the repository root::

    uv run scripts/benchmark.py --iterations 10000

The engine is measured alone. No worker process is started, no agent is built,
and no acceptance deadline is waited on, because :mod:`shed.benchmark` does not
import :mod:`shed.match` at all -- so a number printed here cannot include
process startup or a timed wait. Four costs are reported separately: legal-move
generation, an apply/undo pair, building one observation, and complete random
playouts. The observation rows include legal-move generation for the acting
seat, since ``observe`` fills ``PlayerView.legal_moves`` on every call; the
legality rows above them are what that part costs on its own.

Fixtures are discovered from seeded games, so the same ``--seed`` measures the
same positions, and their sizes are printed before any duration. Nothing here is
a performance gate: the tests assert the report's shape and never its timings.

This file is an argument parser and nothing else. The fixtures and measurements
live in :mod:`shed.benchmark`, and the console layout in :mod:`shed.cli`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from shed.benchmark import (
    DEFAULT_ITERATIONS,
    DEFAULT_PLAYERS,
    DEFAULT_PLAYOUTS,
    DEFAULT_REPEATS,
    DEFAULT_SEED,
    DEFAULT_TRANSITIONS,
    BenchmarkConfig,
    run_benchmark,
    write_benchmark,
)
from shed.cli import benchmark_summary, positive_count, restore_default_sigpipe


def build_parser() -> argparse.ArgumentParser:
    """Build the command's argument parser.

    Returns:
        The parser, with every option the benchmark command accepts.
    """
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Measure the Shed engine on representative positions.",
    )
    parser.add_argument(
        "--iterations",
        type=positive_count,
        default=DEFAULT_ITERATIONS,
        help="calls per repeat for legal moves and observations",
    )
    parser.add_argument(
        "--transitions",
        type=positive_count,
        default=DEFAULT_TRANSITIONS,
        help="apply/undo pairs per repeat; each pair copies the whole position twice",
    )
    parser.add_argument(
        "--playouts",
        type=positive_count,
        default=DEFAULT_PLAYOUTS,
        help="complete engine-only random games per repeat",
    )
    parser.add_argument(
        "--repeats",
        type=positive_count,
        default=DEFAULT_REPEATS,
        help="how many times each measurement runs; the fastest repeat is reported",
    )
    parser.add_argument(
        "--players",
        type=positive_count,
        default=DEFAULT_PLAYERS,
        help="seats in the benchmarked games",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="seed for fixture discovery and for the playout deals",
    )
    parser.add_argument("--output", type=Path, help="also write the report as JSON to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark command.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code: ``0`` once the report is printed and any
        requested file written, ``2`` for arguments that describe no runnable
        benchmark, and ``1`` when the report could not be written.
    """
    args = build_parser().parse_args(argv)
    try:
        config = BenchmarkConfig(
            seed=args.seed,
            players=args.players,
            iterations=args.iterations,
            transitions=args.transitions,
            playouts=args.playouts,
            repeats=args.repeats,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    report = run_benchmark(config)
    print("\n".join(benchmark_summary(report)))

    if args.output is None:
        return 0
    try:
        written = write_benchmark(report, args.output)
    except OSError as error:
        print(f"error: cannot write {args.output}: {error}", file=sys.stderr)
        return 1
    print(f"\nwrote {written}")
    return 0


if __name__ == "__main__":
    restore_default_sigpipe()
    raise SystemExit(main())
