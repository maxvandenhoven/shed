"""Engine-only benchmark: representative positions and what the engine costs on them.

This module measures the engine and nothing else. It imports :mod:`shed.engine`
and the standard library, and deliberately not :mod:`shed.match`, so no worker
process is ever started and no timed wait can leak into a number reported here.
Startup, transport, agent construction, and the acceptance deadline are the
runner's costs and are measured nowhere in this file.

Four things are measured separately, because they are charged separately during
a search and mixing them would hide which one dominates:

* **Legality** -- :meth:`~shed.engine.GameState.get_legal_moves` alone. No
  observation is built, so this is the cost of grouping the active zone and
  comparing ranks, and nothing else.
* **Transitions** -- one :meth:`~shed.engine.GameState.apply_move` followed by
  the matching :meth:`~shed.engine.GameState.undo_move`, timed as a pair
  because that is how a search spends them. Both ends copy the whole position:
  ``apply_move`` snapshots before mutating and ``undo_move`` copies the snapshot
  back, which is the deliberate first-release choice the specification asks to
  measure before anything cheaper is attempted.
* **Observations** -- :meth:`~shed.engine.GameState.observe`. This is *not*
  legality-free: ``observe`` generates the acting seat's legal moves to fill
  ``PlayerView.legal_moves``, and does so whoever is viewing, so an observation
  costs a legal-move generation plus the snapshot around it. The two rows are
  reported side by side precisely so that inclusion is visible rather than
  implied. History is passed in already filtered and stored by reference, so its
  length does not drive this measurement; filtering it belongs to the runner.
* **Playout throughput** -- complete random games driven straight through
  ``get_legal_moves`` and ``apply_move``, with no view, no agent, and no clock.
  The deal is inside the timed region because it is engine work too.

Fixtures are discovered rather than hand-written: seeded random games are played
and the first position matching each wanted shape is copied out. That keeps them
representative of positions the engine actually reaches -- a hand grown by a
pickup, a forced pickup, a burn, the face-up collection, a blind reveal -- rather
than of positions chosen to look fast, and it keeps them reproducible, since the
same seed finds the same positions.

Timings are wall-clock :func:`time.perf_counter` totals over a loop of a fixed
size, repeated a few times. The headline figure is the *fastest* repeat, which
is the one least contaminated by scheduling noise; the slowest is reported
beside it so a reader can see how noisy the machine was. Nothing here is a
performance gate: no test asserts a duration, and no number in this module is a
comparison against another implementation.
"""

from __future__ import annotations

import json
import platform
import random
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import shed
from shed.engine import (
    DEFAULT_DEALER,
    DEFAULT_RULES,
    GameState,
    Move,
    ObservedEvent,
    Phase,
    PickUp,
    Play,
    PlayerId,
    Rank,
    StateInvariantError,
    Zone,
    filter_events_for,
)

__all__ = [
    "DEFAULT_ITERATIONS",
    "DEFAULT_PLAYERS",
    "DEFAULT_PLAYOUTS",
    "DEFAULT_REPEATS",
    "DEFAULT_SEED",
    "DEFAULT_TRANSITIONS",
    "MAX_PLAYOUT_DECISIONS",
    "BenchmarkConfig",
    "BenchmarkReport",
    "Fixture",
    "PlatformInfo",
    "Timing",
    "benchmark_document",
    "build_fixtures",
    "describe_platform",
    "measure_legality",
    "measure_observations",
    "measure_playouts",
    "measure_transitions",
    "run_benchmark",
    "write_benchmark",
]

DEFAULT_SEED = 0
"""Seed for fixture discovery and for the playout deals."""

DEFAULT_PLAYERS = 3
"""Seats in the benchmarked games; the middle of the profile's 2-5 range."""

DEFAULT_ITERATIONS = 2_000
"""Calls per repeat for the per-call measurements: legality and observation."""

DEFAULT_TRANSITIONS = 100
"""Apply/undo pairs per repeat; far fewer, because each pair copies the position twice."""

DEFAULT_PLAYOUTS = 10
"""Complete random games per repeat."""

DEFAULT_REPEATS = 3
"""How many times each measurement is repeated; the fastest repeat is the headline."""

MAX_PLAYOUT_DECISIONS = 10_000
"""Truncation limit for one playout, matching the match runner's default action limit."""

_FIXTURE_DEALS = 64
"""How many seeded games fixture discovery may search before giving up."""


@dataclass(frozen=True, slots=True)
class PlatformInfo:
    """What machine and interpreter produced a set of measurements.

    Timings are meaningless without it: the same engine on another interpreter
    or another machine is another number.

    Attributes:
        python_version: Interpreter version, such as ``3.12.7``.
        implementation: Interpreter implementation, such as ``CPython``.
        system: Operating system name and release.
        machine: Processor architecture reported by the platform module.
        package_version: Version of the installed ``shed`` distribution.
    """

    python_version: str
    implementation: str
    system: str
    machine: str
    package_version: str


def describe_platform() -> PlatformInfo:
    """Describe the interpreter and machine this process is running on.

    Returns:
        The provenance recorded alongside every measurement in a report.
    """
    return PlatformInfo(
        python_version=platform.python_version(),
        implementation=platform.python_implementation(),
        system=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        package_version=shed.__version__,
    )


@dataclass(frozen=True, slots=True)
class Fixture:
    """One captured position, the move measured on it, and how big it is.

    The state is a private deep copy taken out of a seeded game, so measuring on
    it cannot disturb the game it came from, and two fixtures never share a
    mutable object.

    Attributes:
        name: Short stable label, used in tables and in the JSON report.
        description: What shape of position this is and why it is interesting.
        state: The captured position, at a valid decision boundary.
        viewer: The seat that must decide here, and the seat observed for the
            observation measurement.
        move: The move the transition measurement applies and undoes. It is the
            actor's first legal move unless the fixture exists to measure
            something specific, such as a burn.
        history: That viewer's already-filtered history at capture time, passed
            to ``observe`` exactly as the runner passes it.
    """

    name: str
    description: str
    state: GameState
    viewer: PlayerId
    move: Move
    history: tuple[ObservedEvent, ...]

    @property
    def legal_move_count(self) -> int:
        """How many moves the actor may choose between in this position."""
        return len(self.state.get_legal_moves())

    @property
    def sizes(self) -> dict[str, int]:
        """Report the fixture's size, so a timing can be read against it.

        Returns:
            The counts that drive the measured work: the actor's three zones,
            the two piles, how many moves were generated, and how many events
            the viewer's history already holds.
        """
        actor = self.state.players[self.viewer]
        return {
            "hand": len(actor.hand),
            "face_up": len(actor.face_up),
            "face_down": len(actor.face_down),
            "draw": len(self.state.draw_pile),
            "discard": len(self.state.discard_pile),
            "legal_moves": self.legal_move_count,
            "history": len(self.history),
        }


@dataclass(frozen=True, slots=True)
class Timing:
    """One measured operation: how much of it ran, and how long that took.

    Attributes:
        measurement: Which of the four measurements this row belongs to.
        subject: What was measured on -- a fixture name, or the playout
            configuration.
        unit: The singular name of one operation, such as ``call`` or ``pair``.
        operations: Operations performed in each repeat.
        seconds: Wall-clock total of each repeat, in the order they ran.
        detail: A short note about what one operation covers.
    """

    measurement: str
    subject: str
    unit: str
    operations: int
    seconds: tuple[float, ...]
    detail: str

    @property
    def repeats(self) -> int:
        """How many times the loop was repeated."""
        return len(self.seconds)

    @property
    def best_seconds(self) -> float:
        """Total wall-clock time of the fastest repeat, in seconds."""
        return min(self.seconds)

    @property
    def worst_seconds(self) -> float:
        """Total wall-clock time of the slowest repeat, in seconds."""
        return max(self.seconds)

    @property
    def per_operation(self) -> float:
        """Seconds for one operation in the fastest repeat.

        Returns:
            The fastest repeat's total divided by its operation count. Loop
            overhead is included and is not subtracted out; it is a fraction of
            a microsecond per iteration, which matters for the cheapest
            measurement here and for none of the others.
        """
        return self.best_seconds / self.operations

    @property
    def per_second(self) -> float:
        """Operations per second at the fastest repeat's rate."""
        elapsed = self.best_seconds
        return self.operations / elapsed if elapsed > 0 else float("inf")

    @property
    def spread(self) -> float:
        """Ratio of the slowest repeat to the fastest, as a noise indicator.

        Returns:
            ``1.0`` for repeats that took exactly the same time, and more as the
            machine's other work interferes. A large spread means the headline
            figure should be read as an order of magnitude.
        """
        best = self.best_seconds
        return self.worst_seconds / best if best > 0 else 1.0


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """What one benchmark run measures, and how much of it.

    Attributes:
        seed: Seed for fixture discovery and for the playout deals. The same
            seed measures the same positions and replays the same games.
        players: Seats in the benchmarked games, within the profile's range.
        iterations: Calls per repeat for legality and observation.
        transitions: Apply/undo pairs per repeat.
        playouts: Complete random games per repeat.
        repeats: How many times each measurement runs.
    """

    seed: int = DEFAULT_SEED
    players: int = DEFAULT_PLAYERS
    iterations: int = DEFAULT_ITERATIONS
    transitions: int = DEFAULT_TRANSITIONS
    playouts: int = DEFAULT_PLAYOUTS
    repeats: int = DEFAULT_REPEATS

    def __post_init__(self) -> None:
        """Reject a configuration that would measure nothing.

        Raises:
            ValueError: If the player count is outside the profile's range or
                any of the counts is not positive. A zero-iteration measurement
                would divide by zero rather than report a fast engine.
        """
        rules = DEFAULT_RULES
        if not rules.min_players <= self.players <= rules.max_players:
            raise ValueError(
                f"{rules.id} supports {rules.min_players}-{rules.max_players} players, "
                f"got {self.players}"
            )
        for name in ("iterations", "transitions", "playouts", "repeats"):
            value: int = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Everything one benchmark run produced.

    Attributes:
        config: What was asked for.
        platform: Where it ran.
        rules_profile: Identifier of the rules profile that was measured.
        fixtures: The captured positions, in discovery order.
        timings: Every measured row, grouped measurement by measurement.
        playout_decisions: Decisions resolved in one repeat of the playouts.
        truncated_playouts: Playouts that hit :data:`MAX_PLAYOUT_DECISIONS`
            instead of ending, counted per repeat. A random game is not
            guaranteed to terminate, and a truncated one still contributes its
            decisions to the throughput figure.
    """

    config: BenchmarkConfig
    platform: PlatformInfo
    rules_profile: str
    fixtures: tuple[Fixture, ...]
    timings: tuple[Timing, ...]
    playout_decisions: int
    truncated_playouts: int


def _burning_play(state: GameState) -> Move | None:
    """Find a legal move that would burn the pile, if the actor has one.

    Args:
        state: The position to inspect.

    Returns:
        A ten, or a batch of four of one rank -- the two ways ``shed-v1`` burns
        in a single action -- or ``None`` when neither is available. The pile
        must be worth burning: burning an empty pile is legal but is not the
        transition this fixture exists to measure.
    """
    if not state.discard_pile:
        return None
    for move in state.get_legal_moves():
        if isinstance(move, Play) and (move.rank is Rank.TEN or move.count == 4):
            return move
    return None


def _first_move(state: GameState) -> Move:
    """Return the actor's first legal move.

    Args:
        state: The position to inspect.

    Returns:
        The first move in the engine's deterministic order, which is what the
        transition measurement applies unless a fixture wants a specific one.
    """
    return state.get_legal_moves()[0]


def _setup_move(state: GameState) -> Move | None:
    """Select the arrangement fixture's move, if this is a setup decision.

    Args:
        state: The position to inspect.

    Returns:
        The first of the 20 arrangements, or ``None`` outside SETUP.
    """
    return _first_move(state) if state.phase is Phase.SETUP else None


def _opening_move(state: GameState) -> Move | None:
    """Select the opening-play fixture's move, if this is the first PLAY decision.

    Args:
        state: The position to inspect.

    Returns:
        The opener's first legal move on an empty pile with a full deck, or
        ``None`` anywhere else.
    """
    if state.phase is not Phase.PLAY or state.current_ply != 0:
        return None
    return _first_move(state)


def _grown_hand_move(state: GameState) -> Move | None:
    """Select the large-hand fixture's move, if the actor's hand has grown.

    Args:
        state: The position to inspect.

    Returns:
        The first legal move for an actor holding at least eight cards -- a hand
        a pickup grew well past the refill target, which is where rank grouping
        has the most to do -- or ``None`` otherwise.
    """
    if state.phase is not Phase.PLAY or state.current_player is None:
        return None
    if len(state.players[state.current_player].hand) < 8:
        return None
    return _first_move(state)


def _pickup_move(state: GameState) -> Move | None:
    """Select the forced-pickup fixture's move, if pickup is all there is.

    Args:
        state: The position to inspect.

    Returns:
        The forced :class:`~shed.engine.PickUp`, or ``None`` when the actor has
        a playable batch.
    """
    if state.phase is not Phase.PLAY:
        return None
    return PickUp() if state.get_legal_moves() == (PickUp(),) else None


def _zone_move(state: GameState, zone: Zone) -> Move | None:
    """Select the first legal move when the actor is playing from one zone.

    Args:
        state: The position to inspect.
        zone: The active zone the fixture is looking for.

    Returns:
        The actor's first legal move when their active zone is ``zone``, or
        ``None`` otherwise.
    """
    if state.phase is not Phase.PLAY or state.current_player is None:
        return None
    active = state.players[state.current_player].active_zone(len(state.draw_pile))
    return _first_move(state) if active is zone else None


_WANTED: tuple[tuple[str, str, Callable[[GameState], Move | None]], ...] = (
    ("setup", "arrangement: 6 cards, 20 choices", _setup_move),
    ("opening", "first play of the game: full deck, empty pile", _opening_move),
    ("grown-hand", "hand grown past the refill target by a pickup", _grown_hand_move),
    ("pickup", "blocked: pickup is the only legal move", _pickup_move),
    ("burn", "a ten or a batch of four clears a live pile", _burning_play),
    (
        "face-up",
        "hand and deck empty: playing off the table",
        lambda s: _zone_move(s, Zone.FACE_UP),
    ),
    (
        "face-down",
        "only face-down slots left: a blind reveal",
        lambda s: _zone_move(s, Zone.FACE_DOWN),
    ),
)
"""The positions a report covers, in the order they are captured and printed."""


def build_fixtures(
    *,
    seed: int = DEFAULT_SEED,
    players: int = DEFAULT_PLAYERS,
) -> tuple[Fixture, ...]:
    """Capture one representative position of each measured shape.

    Seeded random games are played through the engine and the first position
    matching each wanted shape is deep-copied out, together with the history the
    runner would have handed that seat. Discovery is deterministic: the same
    seed and player count capture the same positions, which is what makes two
    runs of the benchmark comparable.

    Args:
        seed: Seed for the deals and for the moves that walk through them.
        players: Seats in the games searched.

    Returns:
        One fixture per shape, in the order they are declared.

    Raises:
        RuntimeError: If a shape was not reached within the search budget, which
            means the fixture set needs a different seed rather than a silently
            shorter report.
    """
    seeds = random.Random(seed)
    found: dict[str, Fixture] = {}

    for _ in range(_FIXTURE_DEALS):
        deal_seed = seeds.randrange(2**32)
        rng = random.Random(seeds.randrange(2**32))
        state = GameState.create(players, seed=deal_seed, dealer=DEFAULT_DEALER)
        events: list[ObservedEvent] = list(state.initial_events())

        decisions = 0
        while not state.is_finished and decisions < MAX_PLAYOUT_DECISIONS:
            actor = state.current_player
            if actor is None:
                raise StateInvariantError(f"No actor is scheduled in phase {state.phase.value}")
            for name, description, select in _WANTED:
                if name in found:
                    continue
                move = select(state)
                if move is None:
                    continue
                found[name] = Fixture(
                    name=name,
                    description=description,
                    state=deepcopy(state),
                    viewer=actor,
                    move=move,
                    history=filter_events_for(tuple(events), actor),
                )
            if len(found) == len(_WANTED):
                return tuple(found[name] for name, _, _ in _WANTED)
            events.extend(state.apply_move(rng.choice(state.get_legal_moves())).events)
            decisions += 1

    missing = ", ".join(name for name, _, _ in _WANTED if name not in found)
    raise RuntimeError(
        f"Fixture discovery found no {missing} position in {_FIXTURE_DEALS} "
        f"{players}-player games from seed {seed}"
    )


def _repeat(operation: Callable[[], object], *, iterations: int, repeats: int) -> tuple[float, ...]:
    """Time a loop of one operation, several times over.

    Args:
        operation: The callable to time. It must leave the position it works on
            unchanged, so every iteration and every repeat measures the same
            work.
        iterations: Calls per repeat.
        repeats: How many repeats to run.

    Returns:
        The wall-clock total of each repeat, in the order they ran. The caller
        reads the fastest as the headline and the slowest as the noise floor.
    """
    totals: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            operation()
        totals.append(time.perf_counter() - start)
    return tuple(totals)


def measure_legality(fixtures: Sequence[Fixture], config: BenchmarkConfig) -> tuple[Timing, ...]:
    """Measure legal-move generation on its own.

    No observation is constructed here: this is ``get_legal_moves`` and nothing
    around it, which is what makes the observation rows readable as "this plus a
    snapshot".

    Args:
        fixtures: The positions to measure.
        config: Iteration and repeat counts.

    Returns:
        One timing per fixture, in fixture order.
    """
    return tuple(
        Timing(
            measurement="legality",
            subject=fixture.name,
            unit="call",
            operations=config.iterations,
            seconds=_repeat(
                fixture.state.get_legal_moves,
                iterations=config.iterations,
                repeats=config.repeats,
            ),
            detail=f"{fixture.legal_move_count} moves generated",
        )
        for fixture in fixtures
    )


def measure_transitions(fixtures: Sequence[Fixture], config: BenchmarkConfig) -> tuple[Timing, ...]:
    """Measure one apply and its undo, timed as a pair.

    A pair is the unit a search spends, and both halves copy the whole position:
    ``apply_move`` snapshots before it mutates, and ``undo_move`` copies that
    snapshot back so the record stays reusable. Timing them together measures
    the round trip a caller actually pays for, and leaves the fixture exactly as
    it was for the next iteration.

    Args:
        fixtures: The positions to measure.
        config: Transition and repeat counts.

    Returns:
        One timing per fixture, in fixture order.
    """
    timings: list[Timing] = []
    for fixture in fixtures:
        state = fixture.state
        move = fixture.move

        def pair(state: GameState = state, move: Move = move) -> None:
            """Apply the fixture's move and immediately undo it.

            Args:
                state: The fixture's position, bound at definition time so the
                    loop does not look it up through a closure cell.
                move: The fixture's move, bound the same way.
            """
            state.undo_move(state.apply_move(move))

        timings.append(
            Timing(
                measurement="transition",
                subject=fixture.name,
                unit="pair",
                operations=config.transitions,
                seconds=_repeat(pair, iterations=config.transitions, repeats=config.repeats),
                detail="apply_move + undo_move",
            )
        )
    return tuple(timings)


def measure_observations(
    fixtures: Sequence[Fixture], config: BenchmarkConfig
) -> tuple[Timing, ...]:
    """Measure building one player view.

    The acting seat is the viewer, and the history is the filtered tuple the
    runner would pass. Observation *includes* legal-move generation: ``observe``
    fills ``PlayerView.legal_moves`` for the acting seat on every call, so each
    row here contains the whole of the matching legality row plus the snapshot
    of the public position around it. History is stored by reference and is not
    copied, so its length does not appear in this cost.

    Args:
        fixtures: The positions to measure.
        config: Iteration and repeat counts.

    Returns:
        One timing per fixture, in fixture order.
    """
    timings: list[Timing] = []
    for fixture in fixtures:
        state = fixture.state
        viewer = fixture.viewer
        history = fixture.history

        def observe(
            state: GameState = state,
            viewer: PlayerId = viewer,
            history: tuple[ObservedEvent, ...] = history,
        ) -> object:
            """Build the fixture's observation.

            Args:
                state: The fixture's position, bound at definition time.
                viewer: The acting seat, bound the same way.
                history: That seat's filtered history, bound the same way.

            Returns:
                The view, discarded by the timing loop.
            """
            return state.observe(viewer, history=history)

        timings.append(
            Timing(
                measurement="observation",
                subject=fixture.name,
                unit="call",
                operations=config.iterations,
                seconds=_repeat(observe, iterations=config.iterations, repeats=config.repeats),
                detail=f"includes {fixture.legal_move_count} legal moves",
            )
        )
    return tuple(timings)


def _playout(state: GameState, rng: random.Random) -> int:
    """Play one game to its end with uniformly random legal moves.

    Nothing but the engine is involved: no view is built, no agent is
    constructed, and no clock is consulted.

    Args:
        state: A fresh position to play out; it is mutated to completion.
        rng: The generator choosing among legal moves.

    Returns:
        How many decisions were resolved, whether the game ended or hit
        :data:`MAX_PLAYOUT_DECISIONS` first.
    """
    decisions = 0
    while not state.is_finished and decisions < MAX_PLAYOUT_DECISIONS:
        state.apply_move(rng.choice(state.get_legal_moves()))
        decisions += 1
    return decisions


def measure_playouts(config: BenchmarkConfig) -> tuple[tuple[Timing, ...], int, int]:
    """Measure engine-only random playout throughput.

    Every repeat plays the same games -- the deal seeds and move seeds are drawn
    once, before the first repeat -- so the repeats are comparable and their
    spread is machine noise rather than a different amount of work. The deal is
    inside the timed region: shuffling and dealing is engine work, and a search
    that starts games pays for it.

    Args:
        config: Playout and repeat counts, the player count, and the seed.

    Returns:
        A triple of the two timings -- the same run reported per game and per
        decision -- the decisions one repeat resolved, and how many of the
        playouts were truncated at :data:`MAX_PLAYOUT_DECISIONS` rather than
        ending.
    """
    seeds = random.Random(config.seed)
    games = [(seeds.randrange(2**32), seeds.randrange(2**32)) for _ in range(config.playouts)]
    decisions = 0
    truncated = 0

    def run() -> None:
        """Play every game of one repeat, counting decisions on the way."""
        nonlocal decisions, truncated
        decisions = 0
        truncated = 0
        for deal_seed, move_seed in games:
            state = GameState.create(config.players, seed=deal_seed, dealer=DEFAULT_DEALER)
            resolved = _playout(state, random.Random(move_seed))
            decisions += resolved
            if not state.is_finished:
                truncated += 1

    seconds = _repeat(run, iterations=1, repeats=config.repeats)
    detail = f"{config.players} players, {decisions} decisions, {truncated} truncated"
    return (
        (
            Timing(
                measurement="playout",
                subject="random game",
                unit="game",
                operations=config.playouts,
                seconds=seconds,
                detail=detail,
            ),
            Timing(
                measurement="playout",
                subject="random decision",
                unit="decision",
                operations=decisions,
                seconds=seconds,
                detail="deal, legality, and apply for every decision",
            ),
        ),
        decisions,
        truncated,
    )


def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    """Discover the fixtures and run every measurement on them.

    Args:
        config: What to measure and how much of it.

    Returns:
        The complete report, ready to print or to write as JSON.

    Raises:
        RuntimeError: If fixture discovery could not reach one of the shapes.
    """
    fixtures = build_fixtures(seed=config.seed, players=config.players)
    playouts, decisions, truncated = measure_playouts(config)
    return BenchmarkReport(
        config=config,
        platform=describe_platform(),
        rules_profile=DEFAULT_RULES.id,
        fixtures=fixtures,
        timings=(
            *measure_legality(fixtures, config),
            *measure_transitions(fixtures, config),
            *measure_observations(fixtures, config),
            *playouts,
        ),
        playout_decisions=decisions,
        truncated_playouts=truncated,
    )


def _encode_timing(timing: Timing) -> dict[str, object]:
    """Encode one measured row.

    Args:
        timing: The row to encode.

    Returns:
        The raw repeat totals together with the figures derived from them, so a
        reader neither has to recompute them nor has to trust that they were
        computed the same way twice.
    """
    return {
        "measurement": timing.measurement,
        "subject": timing.subject,
        "unit": timing.unit,
        "operations": timing.operations,
        "repeats": timing.repeats,
        "seconds": list(timing.seconds),
        "best_seconds": timing.best_seconds,
        "worst_seconds": timing.worst_seconds,
        "seconds_per_operation": timing.per_operation,
        "operations_per_second": timing.per_second,
        "detail": timing.detail,
    }


def benchmark_document(report: BenchmarkReport) -> dict[str, object]:
    """Build the JSON-ready document for one benchmark run.

    Args:
        report: The completed run.

    Returns:
        Provenance, the configuration, the fixture sizes, and every measured
        row. No timing is a threshold and nothing here is compared against
        another engine; the document records what this machine did.
    """
    config = report.config
    return {
        "rules": report.rules_profile,
        "platform": {
            "python_version": report.platform.python_version,
            "implementation": report.platform.implementation,
            "system": report.platform.system,
            "machine": report.platform.machine,
            "package_version": report.platform.package_version,
        },
        "config": {
            "seed": config.seed,
            "players": config.players,
            "iterations": config.iterations,
            "transitions": config.transitions,
            "playouts": config.playouts,
            "repeats": config.repeats,
        },
        "fixtures": [
            {
                "name": fixture.name,
                "description": fixture.description,
                "phase": fixture.state.phase.value,
                "viewer": int(fixture.viewer),
                "sizes": fixture.sizes,
            }
            for fixture in report.fixtures
        ],
        "playouts": {
            "decisions": report.playout_decisions,
            "truncated": report.truncated_playouts,
            "max_decisions": MAX_PLAYOUT_DECISIONS,
        },
        "timings": [_encode_timing(timing) for timing in report.timings],
    }


def write_benchmark(report: BenchmarkReport, path: Path) -> Path:
    """Write a benchmark report to a UTF-8 JSON file.

    Missing parent directories are created, so a caller can name
    ``results/benchmark.json`` in a fresh checkout.

    Args:
        report: The completed run.
        path: Destination file, overwritten if it exists.

    Returns:
        The path written, for the caller to report.

    Raises:
        ValueError: If a measurement is not finite, which JSON cannot represent.
    """
    document = benchmark_document(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(f"{text}\n", encoding="utf-8")
    return path
