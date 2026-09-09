"""Worker lifecycle tests that actually spawn processes.

These are separated from ``tests.test_match`` because they cost real time: every
decision starts a fresh interpreter, so budgets are set comfortably above process
startup and assertions are about selected moves and cleanup rather than
millisecond-perfect timing.

The scripted agents come from ``tests.test_match``; :func:`spawn_scripted_worker`
is the child entry point that builds one and runs the production worker body, so
these tests exercise the real pipe context, the real message protocol, the real
selection policy, and the real cleanup path.
"""

import multiprocessing
import os
import random
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection

import pytest

from shed.agents import AgentSpec
from shed.engine import (
    CardsPlayed,
    GameEnded,
    GameStarted,
    GameState,
    HandDealt,
    Phase,
    PlayerId,
    PlayerView,
)
from shed.match import (
    CloseReason,
    MatchConfig,
    MatchResult,
    MatchRunner,
    MatchStatus,
    Selection,
    _shutdown_worker,
    run_agent_worker,
    select_candidate,
)
from tests.test_match import SCRIPTED_AGENTS

GENEROUS_BUDGET = 5.0
"""Budget comfortably above process startup, for workers that must finish."""

SHORT_BUDGET = 1.0
"""Budget for workers that are meant to be stopped at the deadline."""

CLEANUP_SLACK = 4.0
"""Extra seconds a deadline decision may spend on spawn, receipt, and reaping."""

IMPOSSIBLE_BUDGET = 0.001
"""Budget no spawned worker can meet, so every decision falls back."""


def spawn_scripted_worker(sender: Connection, view: PlayerView, deadline: float, kind: str) -> None:
    """Child entry point: build one scripted agent and run the worker body.

    Kept at module scope with serializable arguments so a spawned child can
    import this module and rebuild the call.

    Args:
        sender: The worker's end of the decision's pipe.
        view: The observation to decide on.
        deadline: Monotonic instant the budget ends.
        kind: Which scripted behaviour to build.
    """
    run_agent_worker(sender, lambda: SCRIPTED_AGENTS[kind](seed=1), view, deadline=deadline)


@dataclass(frozen=True, slots=True)
class SpawnedDecision:
    """What one really-spawned decision produced.

    Attributes:
        selection: What the policy selected.
        elapsed: Wall seconds from starting the worker to finishing cleanup.
        pid: The worker's process ID, kept so a test can prove it was reaped.
    """

    selection: Selection
    elapsed: float
    pid: int


def run_spawned_decision(
    kind: str,
    view: PlayerView,
    *,
    budget: float,
    fallback_seed: int = 0,
) -> SpawnedDecision:
    """Spawn one scripted worker and run the real selection policy against it.

    This mirrors :meth:`shed.match.MatchRunner.choose_move` -- fresh pipe, fresh
    process, deadline taken before starting it, cleanup in ``finally`` -- but
    lets the test dictate the agent's behaviour, which a specification cannot.

    Args:
        kind: Which scripted behaviour the worker runs.
        view: The observation, whose ``legal_moves`` is the acceptance tuple.
        budget: Acceptance budget in seconds.
        fallback_seed: Seed of this decision's fallback stream.

    Returns:
        The selection and the timing around it.
    """
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    started = time.perf_counter()
    deadline = time.monotonic() + budget
    process = context.Process(
        target=spawn_scripted_worker,
        args=(sender, view, deadline, kind),
        daemon=True,
    )
    process.start()
    sender.close()
    pid = process.pid
    assert pid is not None, "A started process has a PID"
    try:
        selection = select_candidate(
            view.legal_moves,
            receiver,
            deadline=deadline,
            fallback_rng=random.Random(fallback_seed),
        )
    finally:
        _shutdown_worker(process, receiver, sender)
    return SpawnedDecision(selection=selection, elapsed=time.perf_counter() - started, pid=pid)


def assert_reaped(pid: int) -> None:
    """Assert no worker survived the decision.

    Args:
        pid: The worker's process ID.

    Raises:
        AssertionError: If a child is still tracked, or the process still
            exists. A recycled PID could in principle mask the second check;
            within one short test run that is vanishingly unlikely.
    """
    assert not multiprocessing.active_children(), "A worker outlived its decision"
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def setup_view_for(seed: int) -> PlayerView:
    """Build the opening SETUP observation of a two-player deal.

    Args:
        seed: Deck seed for the deal.

    Returns:
        The first arranger's view, which offers 20 distinct arrangements.
    """
    state = GameState.create(2, seed=seed)
    actor = state.current_player
    assert actor is not None
    return state.observe(actor, history=state.initial_events())


@pytest.fixture
def view() -> PlayerView:
    """Provide the observation the scripted spawn tests decide on."""
    return setup_view_for(0)


def lineup(kind: str, count: int = 2) -> dict[PlayerId, AgentSpec]:
    """Build a lineup of one built-in kind.

    Args:
        kind: Agent kind for every seat.
        count: Number of seats.

    Returns:
        One specification per seat, labelled by seat.

    """
    return {PlayerId(seat): AgentSpec(kind=kind, name=f"{kind}-{seat}") for seat in range(count)}


class TestSpawnedWorkers:
    """One real process per decision, for each way an agent can behave."""

    def test_a_final_submission_closes_the_decision_early(self, view: PlayerView) -> None:
        """The ordinary case: the worker finalizes long before the deadline."""
        decision = run_spawned_decision("final", view, budget=GENEROUS_BUDGET)
        assert decision.selection.move == view.legal_moves[0]
        assert decision.selection.reason is CloseReason.FINAL
        assert decision.selection.accepted == 1
        assert not decision.selection.used_fallback
        assert decision.elapsed < GENEROUS_BUDGET
        assert_reaped(decision.pid)

    def test_returning_without_submitting_uses_the_fallback(self, view: PlayerView) -> None:
        """A worker that returns closes the turn; the fallback supplies the move."""
        decision = run_spawned_decision("silent", view, budget=GENEROUS_BUDGET)
        assert decision.selection.reason is CloseReason.RETURNED
        assert decision.selection.used_fallback
        assert decision.selection.move in view.legal_moves
        assert decision.elapsed < GENEROUS_BUDGET
        assert_reaped(decision.pid)

    def test_repeated_submissions_leave_the_latest_standing(self, view: PlayerView) -> None:
        """Every candidate is accepted, and the last one is the decision."""
        decision = run_spawned_decision("chatty", view, budget=GENEROUS_BUDGET)
        assert decision.selection.accepted == len(view.legal_moves)
        assert decision.selection.move == view.legal_moves[-1]
        assert decision.selection.reason is CloseReason.RETURNED
        assert_reaped(decision.pid)

    def test_an_exception_after_a_submission_keeps_that_candidate(self, view: PlayerView) -> None:
        """A crash is recorded, and the candidate it already sent is still used."""
        decision = run_spawned_decision("raising", view, budget=GENEROUS_BUDGET)
        assert decision.selection.move == view.legal_moves[1]
        assert decision.selection.reason is CloseReason.FAILED
        assert decision.selection.failure is not None
        assert decision.selection.failure.exception == "RuntimeError"
        assert not decision.selection.used_fallback
        assert_reaped(decision.pid)

    def test_a_construction_failure_reaches_the_parent(self, view: PlayerView) -> None:
        """An agent that cannot be built fails inside its own worker."""
        decision = run_spawned_decision("broken", view, budget=GENEROUS_BUDGET)
        assert decision.selection.reason is CloseReason.FAILED
        assert decision.selection.failure is not None
        assert decision.selection.failure.exception == "ValueError"
        assert decision.selection.used_fallback
        assert_reaped(decision.pid)

    def test_infinite_computation_is_stopped_at_the_deadline(self, view: PlayerView) -> None:
        """Polling cannot stop a spinning worker, so the parent kills it."""
        decision = run_spawned_decision("infinite", view, budget=SHORT_BUDGET)
        assert decision.selection.reason is CloseReason.DEADLINE
        assert decision.selection.used_fallback
        assert decision.selection.move in view.legal_moves
        assert SHORT_BUDGET <= decision.elapsed < SHORT_BUDGET + CLEANUP_SLACK
        assert_reaped(decision.pid)

    def test_a_candidate_at_the_deadline_is_a_decision_not_a_failure(
        self, view: PlayerView
    ) -> None:
        """A worker that submits and then spins still made an ordinary decision."""
        decision = run_spawned_decision("linger", view, budget=SHORT_BUDGET)
        assert decision.selection.move == view.legal_moves[1]
        assert decision.selection.reason is CloseReason.DEADLINE
        assert not decision.selection.used_fallback
        assert decision.selection.failure is None
        assert_reaped(decision.pid)

    def test_finalizing_then_computing_forever_still_closes_early(self, view: PlayerView) -> None:
        """A final candidate ends the turn; the agent cannot talk its way out of it."""
        decision = run_spawned_decision("final-then-infinite", view, budget=GENEROUS_BUDGET)
        assert decision.selection.move == view.legal_moves[2]
        assert decision.selection.reason is CloseReason.FINAL
        assert decision.selection.accepted == 1
        assert decision.elapsed < GENEROUS_BUDGET
        assert_reaped(decision.pid)

    def test_a_flooded_pipe_cannot_affect_a_later_decision(self, view: PlayerView) -> None:
        """Each decision gets a fresh pipe, so unread candidates die with it."""
        flooded = run_spawned_decision("flood", view, budget=SHORT_BUDGET)
        assert flooded.selection.accepted > 1
        assert_reaped(flooded.pid)

        later = setup_view_for(1)
        assert not set(later.legal_moves) & set(view.legal_moves), "The deals must differ"
        decision = run_spawned_decision("final", later, budget=GENEROUS_BUDGET)
        assert decision.selection.move == later.legal_moves[0]
        assert decision.selection.accepted == 1
        assert_reaped(decision.pid)


class TestRunnerDecisions:
    """The runner's own decision path, with the shipped baselines in workers."""

    def test_the_runner_selects_through_a_real_worker(self, view: PlayerView) -> None:
        """A baseline finalizes immediately, and the record describes that."""
        runner = MatchRunner(lineup("greedy"), MatchConfig(seconds_per_turn=GENEROUS_BUDGET))
        record = runner.choose_move(view)
        assert record.move in view.legal_moves
        assert record.reason is CloseReason.FINAL
        assert not record.agent_failed
        assert record.decision_id == 0
        assert record.player == view.viewer
        assert record.phase is Phase.SETUP
        assert record.budget_seconds == GENEROUS_BUDGET
        assert 0.0 < record.selection_seconds < GENEROUS_BUDGET
        assert record.cleanup_seconds >= 0.0
        assert not multiprocessing.active_children()

    def test_consecutive_decisions_get_fresh_agent_seeds(self, view: PlayerView) -> None:
        """Per-decision seeds are drawn from a stream, never reset to one value."""
        runner = MatchRunner(lineup("random"), MatchConfig(seconds_per_turn=GENEROUS_BUDGET))
        first = runner.choose_move(view)
        second = runner.choose_move(view)
        assert first.agent_seed != second.agent_seed
        assert (first.decision_id, second.decision_id) == (0, 1)
        assert not multiprocessing.active_children()


def assert_coherent(result: MatchResult, *, player_count: int) -> None:
    """Assert the bookkeeping of a match result agrees with itself.

    Args:
        result: The result to check.
        player_count: Seats the match was played with.

    Raises:
        AssertionError: If the records, events, or counters disagree.
    """
    assert [turn.decision_id for turn in result.turns] == list(range(len(result.turns)))
    assert len({turn.agent_seed for turn in result.turns}) == len(result.turns)
    assert [turn.phase for turn in result.turns[:player_count]] == [Phase.SETUP] * player_count
    assert all(turn.phase is Phase.PLAY for turn in result.turns[player_count:])
    assert result.play_decisions == sum(
        decision.turn.phase is Phase.PLAY for decision in result.decisions
    )
    applied = [decision.turn for decision in result.decisions]
    assert applied == [turn for turn in result.turns if turn in applied]
    assert isinstance(result.initial_events[0], GameStarted)
    assert sum(isinstance(event, HandDealt) for event in result.initial_events) == player_count
    assert result.metadata.player_count == player_count
    assert len(result.metadata.agents) == player_count


class TestTimedMatches:
    """Complete matches, played one spawned worker at a time."""

    def test_a_complete_timed_match_produces_coherent_records(self) -> None:
        """A greedy two-player deal finishes with a winner and clean diagnostics."""
        runner = MatchRunner(lineup("greedy"), MatchConfig(seconds_per_turn=GENEROUS_BUDGET))
        result = runner.run(deal_seed=0)

        assert result.status is MatchStatus.FINISHED
        assert result.outcome is not None
        assert result.failure is None
        assert len(result.decisions) == len(result.turns)
        assert result.unapplied_turns == ()
        assert result.play_decisions == len(result.turns) - 2
        assert not any(turn.agent_failed for turn in result.turns)
        assert_coherent(result, player_count=2)

        played = [
            event
            for decision in result.decisions
            for event in decision.events
            if isinstance(event, CardsPlayed)
        ]
        assert played, "A finished match must have played cards"
        assert isinstance(result.decisions[-1].events[-1], GameEnded)
        assert result.decisions[-1].events[-1].outcome == result.outcome
        assert not multiprocessing.active_children()

    def test_a_random_lineup_plays_legally_through_real_workers(self) -> None:
        """The other baseline also decides inside workers, on every phase it meets.

        The limit keeps the test short; what matters is that every decision was
        the worker's own final candidate and that the engine accepted all of
        them, which it only does for moves it generated itself.
        """
        config = MatchConfig(seconds_per_turn=GENEROUS_BUDGET, max_play_decisions=20)
        result = MatchRunner(lineup("random"), config).run(deal_seed=7)

        assert result.status is MatchStatus.TRUNCATED
        assert result.play_decisions == 20
        assert all(turn.reason is CloseReason.FINAL for turn in result.turns)
        assert not any(turn.agent_failed for turn in result.turns)
        setup, play = result.decisions[:2], result.decisions[2:]
        # A stored arrangement stays private until every seat has chosen, so the
        # first submission emits nothing; every PLAY decision resolves into events.
        assert [decision.events for decision in setup][0] == ()
        assert all(decision.events for decision in play)
        assert_coherent(result, player_count=2)
        assert not multiprocessing.active_children()

    def test_the_action_limit_truncates_without_inventing_an_outcome(self) -> None:
        """Truncation is a runner status, never a rules-level result."""
        config = MatchConfig(seconds_per_turn=GENEROUS_BUDGET, max_play_decisions=2)
        first = MatchRunner(lineup("greedy"), config).run(deal_seed=0)

        assert first.status is MatchStatus.TRUNCATED
        assert first.outcome is None
        assert first.play_decisions == 2
        assert first.failure is not None
        assert len(first.turns) == 4  # Two arrangements, then the two play decisions.
        assert len(first.decisions) == 4
        assert_coherent(first, player_count=2)

        second = MatchRunner(lineup("greedy"), config).run(deal_seed=0)
        assert [turn.agent_seed for turn in second.turns] == [
            turn.agent_seed for turn in first.turns
        ]
        assert [turn.move for turn in second.turns] == [turn.move for turn in first.turns]

    def test_seed_streams_restart_at_the_start_of_every_match(self) -> None:
        """A reused runner replays the same seed stream for the next match."""
        config = MatchConfig(seconds_per_turn=GENEROUS_BUDGET, max_play_decisions=1)
        runner = MatchRunner(lineup("greedy"), config)
        first = runner.run(deal_seed=3)
        second = runner.run(deal_seed=3)
        assert [turn.agent_seed for turn in first.turns] == [
            turn.agent_seed for turn in second.turns
        ]
        assert first.turns[0].decision_id == second.turns[0].decision_id == 0


class TestFailurePolicies:
    """A budget no worker can meet, in both failure modes."""

    def test_fallback_mode_keeps_playing_through_missed_deadlines(self) -> None:
        """Every decision falls back, and the match still progresses legally."""
        config = MatchConfig(seconds_per_turn=IMPOSSIBLE_BUDGET, max_play_decisions=2)
        result = MatchRunner(lineup("random"), config).run(deal_seed=0)

        assert result.status is MatchStatus.TRUNCATED
        assert all(turn.used_fallback for turn in result.turns)
        assert all(turn.reason is CloseReason.DEADLINE for turn in result.turns)
        assert all(turn.accepted == 0 for turn in result.turns)
        assert len(result.decisions) == len(result.turns)
        assert not multiprocessing.active_children()

    def test_strict_mode_aborts_before_applying_the_selected_move(self) -> None:
        """The first failing decision ends the match and is marked unapplied."""
        config = MatchConfig(seconds_per_turn=IMPOSSIBLE_BUDGET, strict_failures=True)
        result = MatchRunner(lineup("random"), config).run(deal_seed=0)

        assert result.status is MatchStatus.AGENT_FAILED
        assert result.outcome is None
        assert len(result.turns) == 1
        assert result.decisions == ()
        assert result.unapplied_turns == result.turns
        assert result.play_decisions == 0
        assert result.failure is not None
        assert "no candidate was accepted" in result.failure
        assert not multiprocessing.active_children()
