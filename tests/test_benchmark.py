"""The engine benchmark: its fixtures, its measurements, and what it may not import.

Nothing here asserts a duration. A test that says "an apply/undo pair takes less
than a millisecond" fails on a loaded machine and tells nobody anything about
the engine, so the timings are checked for shape -- positive, one row per
fixture, counted in the units the report claims -- and never for speed.

Two properties do matter and are asserted directly. The benchmark must not reach
the match runner, because a number that included worker startup would not be an
engine measurement; that is checked structurally, by reading the module's
imports. And measuring must leave its fixtures exactly as it found them, or the
second repeat would be timing a different position from the first.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from shed.benchmark import (
    MAX_PLAYOUT_DECISIONS,
    BenchmarkConfig,
    BenchmarkReport,
    Fixture,
    benchmark_document,
    build_fixtures,
    describe_platform,
    measure_legality,
    measure_observations,
    measure_playouts,
    measure_transitions,
    run_benchmark,
    write_benchmark,
)
from shed.cli import benchmark_summary
from shed.engine import Phase, PickUp, Play, Rank, Reveal, Zone
from shed.match import MatchConfig
from tests.conftest import collect_cards

TINY = BenchmarkConfig(iterations=3, transitions=2, playouts=1, repeats=2)
"""A configuration that measures every row without spending real time on any of them."""

FIXTURE_NAMES = ("setup", "opening", "grown-hand", "pickup", "burn", "face-up", "face-down")
"""Every shape a report covers, in the order it captures them."""

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""The directory the documented command is run from."""

BENCHMARK_SOURCE = REPOSITORY_ROOT / "src" / "shed" / "benchmark.py"
"""The module whose imports the engine-only claim depends on."""


@pytest.fixture(scope="module")
def fixtures() -> tuple[Fixture, ...]:
    """Discover the benchmark fixtures once for the whole module.

    Returns:
        The captured positions from the default seed.
    """
    return build_fixtures()


@pytest.fixture(scope="module")
def report() -> BenchmarkReport:
    """Run the smallest complete benchmark once for the whole module.

    Returns:
        The completed report, measured with counts small enough that the suite
        stays fast; nothing in these tests reads a duration.
    """
    return run_benchmark(TINY)


def imported_modules(source: Path) -> set[str]:
    """Collect every module a source file imports at any level.

    Reading the imports rather than the loaded interpreter keeps the check
    honest: a module pulled in by something else in the test session would not
    show up as this file's dependency.

    Args:
        source: The file to read.

    Returns:
        Every dotted module name that appears in an ``import`` statement,
        including ones inside function bodies.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


class TestEngineOnly:
    """The structural half of the "engine-only" claim."""

    @pytest.mark.parametrize(
        "forbidden",
        ["shed.match", "shed.agents", "shed.gauntlet", "shed.replay", "multiprocessing"],
        ids=["match", "agents", "gauntlet", "replay", "multiprocessing"],
    )
    def test_the_benchmark_cannot_reach_the_runner(self, forbidden: str) -> None:
        """No worker, agent, or codec is reachable, so no timing can include one."""
        assert forbidden not in imported_modules(BENCHMARK_SOURCE)

    def test_the_benchmark_imports_the_engine(self) -> None:
        """The check above would also pass on a module that measured nothing."""
        assert "shed.engine" in imported_modules(BENCHMARK_SOURCE)


class TestFixtures:
    """Discovering representative positions."""

    def test_every_declared_shape_is_captured(self, fixtures: tuple[Fixture, ...]) -> None:
        """A short report would be a silently narrower benchmark."""
        assert tuple(fixture.name for fixture in fixtures) == FIXTURE_NAMES

    def test_discovery_is_deterministic(self, fixtures: tuple[Fixture, ...]) -> None:
        """The same seed measures the same positions, so two runs compare."""
        again = build_fixtures()
        assert [fixture.sizes for fixture in again] == [fixture.sizes for fixture in fixtures]
        assert [fixture.move for fixture in again] == [fixture.move for fixture in fixtures]

    def test_each_fixture_is_a_live_decision_for_its_viewer(
        self, fixtures: tuple[Fixture, ...]
    ) -> None:
        """The viewer is the actor, and the recorded move is one they may make."""
        for fixture in fixtures:
            assert fixture.state.current_player == fixture.viewer
            assert fixture.move in fixture.state.get_legal_moves()

    def test_fixtures_share_no_mutable_state(self, fixtures: tuple[Fixture, ...]) -> None:
        """Each position is a private deep copy of the game it came from."""
        piles = [id(fixture.state.draw_pile) for fixture in fixtures]
        assert len(set(piles)) == len(piles)

    def test_every_fixture_conserves_the_deck(self, fixtures: tuple[Fixture, ...]) -> None:
        """A captured position is a real position, not a hand-cut one."""
        for fixture in fixtures:
            assert len(collect_cards(fixture.state)) == 54

    def test_the_shapes_are_what_their_names_claim(self, fixtures: tuple[Fixture, ...]) -> None:
        """Each fixture measures the situation its label promises."""
        by_name = {fixture.name: fixture for fixture in fixtures}

        assert by_name["setup"].state.phase is Phase.SETUP
        assert by_name["setup"].legal_move_count == 20
        assert by_name["opening"].state.current_ply == 0
        assert by_name["grown-hand"].sizes["hand"] >= 8
        assert by_name["pickup"].move == PickUp()
        assert by_name["pickup"].legal_move_count == 1

        burn = by_name["burn"]
        assert isinstance(burn.move, Play)
        assert burn.move.rank is Rank.TEN or burn.move.count == 4
        assert burn.state.discard_pile

        face_up = by_name["face-up"]
        assert isinstance(face_up.move, Play)
        assert face_up.move.source is Zone.FACE_UP
        assert isinstance(by_name["face-down"].move, Reveal)

    def test_sizes_describe_the_position(self, fixtures: tuple[Fixture, ...]) -> None:
        """The printed sizes are read off the fixture, not assumed."""
        for fixture in fixtures:
            state = fixture.state
            actor = state.players[fixture.viewer]
            assert fixture.sizes == {
                "hand": len(actor.hand),
                "face_up": len(actor.face_up),
                "face_down": len(actor.face_down),
                "draw": len(state.draw_pile),
                "discard": len(state.discard_pile),
                "legal_moves": len(state.get_legal_moves()),
                "history": len(fixture.history),
            }

    def test_history_is_filtered_for_its_viewer(self, fixtures: tuple[Fixture, ...]) -> None:
        """A benchmark fixture carries the history the runner would pass, not more."""
        for fixture in fixtures:
            observed = fixture.state.observe(fixture.viewer, history=fixture.history)
            assert observed.history == fixture.history
            assert observed.viewer == fixture.viewer


class TestMeasurements:
    """What each measurement produces, and what it must leave behind."""

    def test_legality_reports_one_row_per_fixture(self, fixtures: tuple[Fixture, ...]) -> None:
        """Every fixture is measured, and no row claims more work than it did."""
        timings = measure_legality(fixtures, TINY)
        assert [timing.subject for timing in timings] == list(FIXTURE_NAMES)
        for timing in timings:
            assert timing.measurement == "legality"
            assert timing.operations == TINY.iterations
            assert timing.repeats == TINY.repeats
            assert timing.best_seconds > 0
            assert timing.per_second > 0

    def test_observations_are_measured_per_fixture(self, fixtures: tuple[Fixture, ...]) -> None:
        """Observation rows carry the legal-move count they include."""
        timings = measure_observations(fixtures, TINY)
        counts = [fixture.legal_move_count for fixture in fixtures]
        for timing, count in zip(timings, counts, strict=True):
            assert timing.measurement == "observation"
            assert f"{count} legal moves" in timing.detail

    def test_a_transition_pair_leaves_the_fixture_untouched(
        self, fixtures: tuple[Fixture, ...]
    ) -> None:
        """Apply and undo must cancel, or later repeats measure another position."""

        def snapshot() -> list[object]:
            """Capture what every fixture looks like right now.

            Returns:
                One entry per fixture: its ply, its legal moves, and its cards.
            """
            return [
                (
                    fixture.state.current_ply,
                    fixture.state.get_legal_moves(),
                    collect_cards(fixture.state),
                )
                for fixture in fixtures
            ]

        before = snapshot()
        measure_transitions(fixtures, TINY)
        assert snapshot() == before

    def test_a_transition_row_counts_pairs(self, fixtures: tuple[Fixture, ...]) -> None:
        """The unit is the pair, because that is what a search spends."""
        for timing in measure_transitions(fixtures, TINY):
            assert timing.unit == "pair"
            assert timing.operations == TINY.transitions

    def test_playouts_report_the_same_run_two_ways(self) -> None:
        """Games and decisions are two views of one timed run, not two runs."""
        (games, decisions), resolved, truncated = measure_playouts(TINY)
        assert games.seconds == decisions.seconds
        assert games.operations == TINY.playouts
        assert decisions.operations == resolved > 0
        assert truncated in range(TINY.playouts + 1)

    def test_playouts_are_reproducible(self) -> None:
        """The same seed plays the same games, so two runs are comparable."""
        _, first, _ = measure_playouts(TINY)
        _, second, _ = measure_playouts(TINY)
        assert first == second

    def test_a_playout_seed_change_changes_the_games(self) -> None:
        """The seed really drives the deals, rather than being recorded and ignored."""
        _, decisions, _ = measure_playouts(TINY)
        _, other, _ = measure_playouts(replace(TINY, seed=99))
        assert decisions != other


class TestConfiguration:
    """Rejecting a run that would measure nothing."""

    @pytest.mark.parametrize(
        "field", ["iterations", "transitions", "playouts", "repeats"], ids=lambda name: name
    )
    def test_a_count_must_be_positive(self, field: str) -> None:
        """A zero-iteration measurement would divide by zero, not run fast."""
        with pytest.raises(ValueError, match=f"{field} must be positive"):
            BenchmarkConfig(**{field: 0})

    @pytest.mark.parametrize("players", [1, 6], ids=["too-few", "too-many"])
    def test_the_table_size_is_the_profile_s(self, players: int) -> None:
        """The benchmark plays real games, so it is bound by the same range."""
        with pytest.raises(ValueError, match="supports 2-5 players"):
            BenchmarkConfig(players=players)


class TestReport:
    """The completed report, its document, and its console rendering."""

    def test_a_report_covers_every_measurement(self, report: BenchmarkReport) -> None:
        """All four measurements are present, each with rows of its own."""
        measurements = {timing.measurement for timing in report.timings}
        assert measurements == {"legality", "transition", "observation", "playout"}

    def test_the_document_is_strict_json(self, report: BenchmarkReport) -> None:
        """Nothing measured may be a NaN, which no strict reader accepts."""
        text = json.dumps(benchmark_document(report), allow_nan=False)
        assert json.loads(text)["rules"] == "shed-v1"

    def test_the_document_records_provenance_and_sizes(self, report: BenchmarkReport) -> None:
        """A timing without its machine and its fixture sizes cannot be read."""
        document = benchmark_document(report)
        platform_info = document["platform"]
        assert isinstance(platform_info, dict)
        assert platform_info["python_version"] == describe_platform().python_version
        fixtures = document["fixtures"]
        assert isinstance(fixtures, list)
        assert [entry["name"] for entry in fixtures] == list(FIXTURE_NAMES)
        assert all(entry["sizes"]["legal_moves"] > 0 for entry in fixtures)
        assert document["config"] == {
            "seed": TINY.seed,
            "players": TINY.players,
            "iterations": TINY.iterations,
            "transitions": TINY.transitions,
            "playouts": TINY.playouts,
            "repeats": TINY.repeats,
        }

    def test_writing_creates_missing_directories(
        self, report: BenchmarkReport, tmp_path: Path
    ) -> None:
        """``results/benchmark.json`` works in a fresh checkout."""
        path = write_benchmark(report, tmp_path / "results" / "benchmark.json")
        assert json.loads(path.read_text(encoding="utf-8"))["timings"]

    def test_the_summary_reports_fixtures_and_caveats(self, report: BenchmarkReport) -> None:
        """The console says what was measured, on what, and what it does not mean."""
        text = "\n".join(benchmark_summary(report))
        for name in FIXTURE_NAMES:
            assert name in text
        assert "shed-v1 engine benchmark" in text
        assert "includes the acting seat's legal moves" in text
        assert f"{MAX_PLAYOUT_DECISIONS:,}" in text
        assert "Nothing here is a" in text


class TestPlayoutsUseTheEngineOnly:
    """The behavioural half of the engine-only claim."""

    def test_a_playout_ends_in_a_finished_or_truncated_game(self) -> None:
        """Random play resolves through the engine alone, with a truncation limit."""
        _, decisions, truncated = measure_playouts(BenchmarkConfig(playouts=2, repeats=1))
        assert decisions > 0
        assert truncated in (0, 1, 2)

    def test_the_truncation_limit_matches_the_runner_default(self) -> None:
        """A playout truncates where a match would, so the two counts compare.

        The benchmark cannot import the runner, so the two limits are separate
        constants; this is what keeps them the same number.
        """
        assert MAX_PLAYOUT_DECISIONS == MatchConfig().max_play_decisions


def run_command(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the benchmark command from the repository root.

    Args:
        *arguments: Command-line arguments to pass it.

    Returns:
        The finished process, with its output captured as text.
    """
    # A fixed argument vector, no shell, and the project's own script.
    return subprocess.run(
        [sys.executable, "scripts/benchmark.py", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


SHORT_RUN = ("--iterations", "50", "--transitions", "5", "--playouts", "1", "--repeats", "1")
"""Enough of every measurement to check the command, and no more."""


class TestCommand:
    """``scripts/benchmark.py``, run as documented."""

    def test_the_documented_command_reports_every_measurement(self, tmp_path: Path) -> None:
        """The command prints the four tables and writes the same run as JSON."""
        output = tmp_path / "results" / "benchmark.json"
        finished = run_command(*SHORT_RUN, "--output", str(output))

        assert finished.returncode == 0, finished.stderr
        assert "shed-v1 engine benchmark" in finished.stdout
        assert "legal moves (get_legal_moves; no view is built):" in finished.stdout
        assert "transitions (apply_move then undo_move, timed as one pair):" in finished.stdout
        assert "observations (observe; includes the acting seat's legal moves):" in finished.stdout
        assert "engine-only playouts" in finished.stdout
        assert str(output) in finished.stdout

        document = json.loads(output.read_text(encoding="utf-8"))
        assert document["config"]["iterations"] == 50
        assert {timing["measurement"] for timing in document["timings"]} == {
            "legality",
            "transition",
            "observation",
            "playout",
        }

    def test_a_report_needs_no_output_file(self) -> None:
        """The console report is the command's product; the file is optional."""
        finished = run_command(*SHORT_RUN)

        assert finished.returncode == 0, finished.stderr
        assert "wrote" not in finished.stdout

    def test_an_unsupported_table_size_exits_two(self) -> None:
        """A benchmark of six players describes no runnable game."""
        finished = run_command(*SHORT_RUN, "--players", "6")

        assert finished.returncode == 2
        assert "supports 2-5 players" in finished.stderr

    def test_a_zero_count_is_refused_by_the_parser(self) -> None:
        """``--repeats 0`` is rejected as an argument, before anything is measured."""
        finished = run_command(*SHORT_RUN, "--repeats", "0")

        assert finished.returncode == 2
        assert "positive integer" in finished.stderr
