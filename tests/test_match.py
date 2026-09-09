"""Selection policy, turn context, worker protocol, and runner bookkeeping.

Everything here runs in one process. The selection policy is driven by a clock a
test advances by hand and a scripted message source, so every boundary in the
design's selection table -- late messages, buffered messages, illegal finals,
completion, failure, fallback -- is exercised without spawning anything. The
worker protocol is driven the same way, over a real in-process pipe.

The scripted agents defined here are also the agents the real spawn tests in
``tests.test_match_processes`` run, so one set of behaviours covers both the
in-process protocol and the actual process lifecycle.
"""

import math
import multiprocessing
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection

import pytest

from shed.agents import Agent, AgentSpec, TurnContext
from shed.engine import (
    DEFAULT_RULES,
    GameState,
    Move,
    Phase,
    PickUp,
    Play,
    PlayerId,
    PlayerView,
    Rank,
    RulesConfig,
    StateInvariantError,
    Zone,
)
from shed.match import (
    AppliedDecision,
    CloseReason,
    MatchConfig,
    MatchMetadata,
    MatchResult,
    MatchRunner,
    MatchStatus,
    PipeTurnContext,
    Selection,
    Submission,
    TurnRecord,
    WorkerFailed,
    WorkerFinished,
    run_agent_worker,
    select_candidate,
)

LEGAL: tuple[Move, ...] = (
    Play(Zone.HAND, Rank.FIVE, 1),
    Play(Zone.HAND, Rank.FIVE, 2),
    PickUp(),
)
"""A stand-in acceptance tuple; the policy only ever tests membership in it."""

ILLEGAL: Move = Play(Zone.HAND, Rank.ACE, 1)
"""A well-formed move that is not in :data:`LEGAL`."""

BUDGET = 1.0
"""Fake-clock budget; the deadline is at this instant on every fake clock."""


def malformed_submission(*, move: object = LEGAL[0], final: object = False) -> Submission:
    """Build a submission whose fields defy their own annotations.

    Unpickling bypasses ``__init__``, so a worker really can put such a message
    on the pipe; the parent must reject it rather than trust the annotations.

    Args:
        move: Payload to put in the move field.
        final: Payload to put in the final field.

    Returns:
        A submission that only looks like one.
    """
    message = Submission.__new__(Submission)
    object.__setattr__(message, "move", move)
    object.__setattr__(message, "final", final)
    return message


class FakeClock:
    """A monotonic clock a test moves by hand.

    Attributes:
        now: The current instant, in seconds.
    """

    def __init__(self, now: float = 0.0) -> None:
        """Start the clock.

        Args:
            now: Initial instant.
        """
        self.now = now

    def __call__(self) -> float:
        """Return the current instant, matching the clock callable the policy takes."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward.

        Args:
            seconds: How far to advance.
        """
        self.now += seconds


@dataclass(frozen=True, slots=True)
class Delivery:
    """One scripted message and how long the parent waits for it.

    Attributes:
        message: What ``recv()`` returns, or an exception it raises instead.
        after: Seconds of waiting before the message becomes readable.
    """

    message: object
    after: float = 0.0


class FakeSource:
    """A scripted message source that drives the fake clock as it delivers.

    Waiting is what moves time here: :meth:`poll` advances the clock by the next
    delivery's delay, or by the whole timeout when nothing more arrives. That
    makes "the message came too late" and "the message was never received"
    expressible without real time.

    Attributes:
        received: Messages actually handed to the policy, in order.
    """

    def __init__(self, clock: FakeClock, deliveries: Sequence[Delivery]) -> None:
        """Script a source against one clock.

        Args:
            clock: The clock deliveries advance.
            deliveries: Messages in arrival order.
        """
        self._clock = clock
        self._pending = list(deliveries)
        self.received: list[object] = []

    def poll(self, timeout: float) -> bool:
        """Wait for the next scripted delivery.

        Args:
            timeout: Seconds the policy is willing to wait.

        Returns:
            Whether a delivery became readable within the timeout.
        """
        if not self._pending:
            self._clock.advance(timeout)
            return False
        wait = self._pending[0].after
        if wait > timeout:
            self._clock.advance(timeout)
            return False
        self._clock.advance(wait)
        return True

    def recv(self) -> object:
        """Deliver the next scripted message.

        Returns:
            The message.

        Raises:
            BaseException: If the script says this delivery is a transport
                failure, such as the EOF a dead worker produces.
        """
        delivery = self._pending.pop(0)
        if isinstance(delivery.message, BaseException):
            raise delivery.message
        self.received.append(delivery.message)
        return delivery.message

    @property
    def unread(self) -> tuple[object, ...]:
        """Return the scripted messages the policy never received."""
        return tuple(delivery.message for delivery in self._pending)


def run_policy(
    deliveries: Sequence[Delivery],
    *,
    legal_moves: tuple[Move, ...] = LEGAL,
    fallback_seed: int = 0,
) -> tuple[Selection, FakeSource]:
    """Run the selection policy over a scripted script on a fake clock.

    Args:
        deliveries: The scripted messages.
        legal_moves: Acceptance tuple frozen for the decision.
        fallback_seed: Seed of the fallback stream.

    Returns:
        The selection and the source, so a test can also assert what was never
        received.
    """
    clock = FakeClock()
    source = FakeSource(clock, deliveries)
    selection = select_candidate(
        legal_moves,
        source,
        deadline=BUDGET,
        fallback_rng=random.Random(fallback_seed),
        clock=clock,
    )
    return selection, source


def fallback_for(seed: int = 0, legal_moves: tuple[Move, ...] = LEGAL) -> Move:
    """Return the move the seeded fallback stream picks first.

    Args:
        seed: Fallback seed.
        legal_moves: The acceptance tuple.

    Returns:
        The first choice of a fresh fallback stream, which is what a decision
        with no accepted candidate must select.
    """
    return random.Random(seed).choice(legal_moves)


class TestSelectionBoundaries:
    """The selection table: what closes a decision, and with which move."""

    def test_latest_legal_candidate_wins(self) -> None:
        """Every candidate replaces the previous one until the worker returns."""
        selection, _ = run_policy(
            [
                Delivery(Submission(move=LEGAL[0]), after=0.1),
                Delivery(Submission(move=LEGAL[2]), after=0.1),
                Delivery(Submission(move=LEGAL[1]), after=0.1),
                Delivery(WorkerFinished(), after=0.1),
            ]
        )
        assert selection.move == LEGAL[1]
        assert selection.reason is CloseReason.RETURNED
        assert (selection.accepted, selection.rejected) == (3, 0)
        assert not selection.used_fallback

    def test_accepted_candidate_is_the_engines_own_instance(self) -> None:
        """An equal submission is canonicalized to the move the engine generated."""
        selection, _ = run_policy(
            [Delivery(Submission(move=Play(Zone.HAND, Rank.FIVE, 1), final=True))]
        )
        assert selection.move is LEGAL[0]

    def test_illegal_candidate_preserves_the_previous_one(self) -> None:
        """A rejection is counted and leaves the standing candidate alone."""
        selection, _ = run_policy(
            [
                Delivery(Submission(move=LEGAL[1]), after=0.1),
                Delivery(Submission(move=ILLEGAL), after=0.1),
                Delivery(WorkerFinished(), after=0.1),
            ]
        )
        assert selection.move == LEGAL[1]
        assert (selection.accepted, selection.rejected) == (1, 1)
        assert not selection.used_fallback

    def test_legal_final_closes_immediately(self) -> None:
        """A final candidate ends the decision; later messages are never read."""
        later = Submission(move=LEGAL[2])
        selection, source = run_policy(
            [
                Delivery(Submission(move=LEGAL[0], final=True), after=0.1),
                Delivery(later, after=0.1),
                Delivery(WorkerFinished(), after=0.1),
            ]
        )
        assert selection.move == LEGAL[0]
        assert selection.reason is CloseReason.FINAL
        assert selection.accepted == 1
        assert source.unread == (later, WorkerFinished())

    def test_illegal_final_keeps_the_earlier_candidate(self) -> None:
        """A final flag cannot make an illegal move legal."""
        selection, _ = run_policy(
            [
                Delivery(Submission(move=LEGAL[2]), after=0.1),
                Delivery(Submission(move=ILLEGAL, final=True), after=0.1),
                Delivery(WorkerFinished(), after=0.1),
            ]
        )
        assert selection.move == LEGAL[2]
        assert selection.reason is CloseReason.RETURNED
        assert (selection.accepted, selection.rejected) == (1, 1)

    def test_illegal_final_without_a_candidate_falls_back(self) -> None:
        """With nothing accepted, the decision uses the seeded fallback."""
        selection, _ = run_policy(
            [
                Delivery(Submission(move=ILLEGAL, final=True), after=0.1),
                Delivery(WorkerFinished(), after=0.1),
            ]
        )
        assert selection.move == fallback_for()
        assert selection.used_fallback
        assert (selection.accepted, selection.rejected) == (0, 1)
        assert selection.reason is CloseReason.RETURNED

    def test_deadline_with_a_candidate_is_an_ordinary_decision(self) -> None:
        """A silent worker at the deadline still leaves its candidate standing."""
        selection, _ = run_policy([Delivery(Submission(move=LEGAL[1]), after=0.2)])
        assert selection.move == LEGAL[1]
        assert selection.reason is CloseReason.DEADLINE
        assert not selection.used_fallback
        assert selection.failure is None

    def test_deadline_without_a_candidate_uses_the_seeded_fallback(self) -> None:
        """No submission at all is the only case the fallback stream is used for."""
        selection, _ = run_policy([])
        assert selection.move == fallback_for()
        assert selection.reason is CloseReason.DEADLINE
        assert selection.used_fallback
        assert selection.accepted == 0

    def test_fallback_is_reproducible_and_only_advances_when_used(self) -> None:
        """The fallback stream is seeded, and an accepted candidate never touches it."""
        first, _ = run_policy([], fallback_seed=7)
        second, _ = run_policy([], fallback_seed=7)
        assert first.move == second.move == fallback_for(7)

        stream = random.Random(7)
        select_candidate(
            LEGAL,
            FakeSource(FakeClock(), [Delivery(Submission(move=LEGAL[0], final=True))]),
            deadline=BUDGET,
            fallback_rng=stream,
            clock=FakeClock(),
        )
        assert stream.choice(LEGAL) == fallback_for(7)

    def test_arrival_exactly_at_the_deadline_is_late(self) -> None:
        """Equality with the deadline is late: the message is read but ignored."""
        message = Submission(move=LEGAL[0], final=True)
        selection, source = run_policy([Delivery(message, after=BUDGET)])
        assert source.received == [message]
        assert selection.used_fallback
        assert selection.reason is CloseReason.DEADLINE

    def test_completion_exactly_at_the_deadline_closes_as_deadline(self) -> None:
        """A completion message is subject to the same receipt check."""
        selection, _ = run_policy(
            [
                Delivery(Submission(move=LEGAL[1]), after=0.5),
                Delivery(WorkerFinished(), after=0.5),
            ]
        )
        assert selection.move == LEGAL[1]
        assert selection.reason is CloseReason.DEADLINE

    def test_buffered_messages_past_expiry_are_never_drained(self) -> None:
        """The policy stops at expiry instead of hunting for a newer candidate."""
        buffered = Submission(move=LEGAL[2], final=True)
        selection, source = run_policy(
            [
                Delivery(Submission(move=LEGAL[0]), after=0.5),
                Delivery(buffered, after=0.6),
            ]
        )
        assert selection.move == LEGAL[0]
        assert selection.reason is CloseReason.DEADLINE
        assert source.unread == (buffered,)

    def test_worker_failure_keeps_the_latest_candidate_and_records_the_crash(self) -> None:
        """A crash after a good submission is still that submission's decision."""
        failure = WorkerFailed(exception="RuntimeError", message="boom")
        selection, _ = run_policy(
            [
                Delivery(Submission(move=LEGAL[1]), after=0.1),
                Delivery(failure, after=0.1),
            ]
        )
        assert selection.move == LEGAL[1]
        assert selection.reason is CloseReason.FAILED
        assert selection.failure == failure
        assert not selection.used_fallback

    def test_worker_failure_without_a_candidate_falls_back(self) -> None:
        """A crash before any submission needs the fallback, and is still a crash."""
        selection, _ = run_policy(
            [Delivery(WorkerFailed(exception="ValueError", message="no"), after=0.1)]
        )
        assert selection.move == fallback_for()
        assert selection.used_fallback
        assert selection.reason is CloseReason.FAILED

    def test_a_dead_pipe_is_recorded_as_a_failure(self) -> None:
        """A worker that vanishes without reporting is a failure, not a return."""
        selection, _ = run_policy([Delivery(EOFError("pipe closed"), after=0.1)])
        assert selection.reason is CloseReason.FAILED
        assert selection.failure is not None
        assert selection.failure.exception == "EOFError"
        assert selection.used_fallback

    @pytest.mark.parametrize(
        "message",
        [
            "not a message",
            42,
            None,
            malformed_submission(move="not a move"),
            malformed_submission(final="yes"),
            Submission(move=ILLEGAL),
        ],
        ids=["text", "number", "none", "not-a-move", "bad-final-flag", "illegal-move"],
    )
    def test_malformed_messages_are_rejected_and_the_decision_continues(
        self, message: object
    ) -> None:
        """Anything that is not a valid, legal submission is counted and skipped."""
        selection, _ = run_policy(
            [
                Delivery(message, after=0.1),
                Delivery(Submission(move=LEGAL[2], final=True), after=0.1),
            ]
        )
        assert selection.move == LEGAL[2]
        assert (selection.accepted, selection.rejected) == (1, 1)

    def test_a_decision_needs_at_least_one_legal_move(self) -> None:
        """An empty acceptance tuple means the caller asked for a nonexistent decision."""
        with pytest.raises(ValueError, match="at least one legal move"):
            run_policy([], legal_moves=())


class ScriptedAgent(Agent):
    """Base class for the agents that script one worker's behaviour.

    Subclasses are deliberately at module scope in an importable test module, so
    a spawned worker can rebuild them.
    """

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Run the scripted behaviour.

        Args:
            view: The observation; ``legal_moves`` supplies the candidates.
            turn: The submission channel.
        """
        raise NotImplementedError


class FinalAgent(ScriptedAgent):
    """Submits the first legal move as final, like the shipped baselines do."""

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Submit one final candidate.

        Args:
            view: The observation.
            turn: The submission channel.
        """
        turn.submit(view.legal_moves[0], final=True)


class SilentAgent(ScriptedAgent):
    """Returns without submitting anything, forcing a fallback."""

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Return immediately.

        Args:
            view: The observation, ignored.
            turn: The submission channel, unused.
        """


class ChattyAgent(ScriptedAgent):
    """Submits every legal move in order without ever finalizing, then returns."""

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Submit each legal move in turn.

        Args:
            view: The observation supplying the candidates.
            turn: The submission channel.
        """
        for move in view.legal_moves:
            turn.submit(move)


class RaisingAgent(ScriptedAgent):
    """Submits a candidate and then raises, the ordinary agent-failure case."""

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Submit once, then fail.

        Args:
            view: The observation.
            turn: The submission channel.

        Raises:
            RuntimeError: Always, after the submission.
        """
        turn.submit(view.legal_moves[1])
        raise RuntimeError("scripted failure")


class BrokenAgent(ScriptedAgent):
    """Fails during construction, before it can be asked to think."""

    def __init__(self, *, seed: int) -> None:
        """Refuse to be built.

        Args:
            seed: Ignored.

        Raises:
            ValueError: Always.
        """
        raise ValueError("scripted construction failure")

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Never reached.

        Args:
            view: The observation.
            turn: The submission channel.
        """


class LingerAgent(ScriptedAgent):
    """Submits a legal candidate, then computes until the parent stops it."""

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Submit one candidate, then spin forever.

        Args:
            view: The observation.
            turn: The submission channel.
        """
        turn.submit(view.legal_moves[1])
        while True:
            pass


class InfiniteAgent(ScriptedAgent):
    """Computes forever without ever submitting."""

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Spin forever.

        Args:
            view: The observation, ignored.
            turn: The submission channel, unused.
        """
        while True:
            pass


class FinalThenInfiniteAgent(ScriptedAgent):
    """Finalizes and then keeps computing, which must not change the decision."""

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Submit a final candidate, then spin forever.

        Args:
            view: The observation.
            turn: The submission channel.
        """
        turn.submit(view.legal_moves[2], final=True)
        while True:
            turn.submit(view.legal_moves[0])


class FloodAgent(ScriptedAgent):
    """Submits the same legal candidate forever, never finalizing.

    It exists to leave a decision's pipe full of unread messages when the parent
    stops it, so a test can show that those messages cannot reach a later turn.
    """

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Submit the first legal move over and over.

        Args:
            view: The observation.
            turn: The submission channel.
        """
        while True:
            turn.submit(view.legal_moves[0])


SCRIPTED_AGENTS: dict[str, type[ScriptedAgent]] = {
    "final": FinalAgent,
    "silent": SilentAgent,
    "chatty": ChattyAgent,
    "raising": RaisingAgent,
    "broken": BrokenAgent,
    "linger": LingerAgent,
    "infinite": InfiniteAgent,
    "final-then-infinite": FinalThenInfiniteAgent,
    "flood": FloodAgent,
}
"""Scripted behaviours by name, so a spawned worker can be told which to build."""


@pytest.fixture
def setup_view() -> PlayerView:
    """Provide the opening SETUP observation of a two-player deal.

    Returns:
        The first arranger's view, whose 20 arrangements give scripted agents
        several distinct legal candidates to choose between.
    """
    state = GameState.create(2, seed=0)
    actor = state.current_player
    assert actor is not None
    return state.observe(actor, history=state.initial_events())


def drain(receiver: Connection) -> list[object]:
    """Read every message already waiting on a pipe, stopping at end of file.

    A closed writer makes the endpoint readable, so the end of the stream has to
    be caught rather than polled for.

    Args:
        receiver: The parent's endpoint, whose writer has been closed.

    Returns:
        The buffered messages in order.
    """
    messages: list[object] = []
    while receiver.poll(0.0):
        try:
            messages.append(receiver.recv())
        except EOFError:
            break
    return messages


class TestPipeTurnContext:
    """The concrete agent-facing turn, over a real in-process pipe."""

    def test_submissions_travel_and_final_closes_the_channel(self) -> None:
        """After a final submission the context stops sending, silently."""
        receiver, sender = multiprocessing.Pipe(duplex=False)
        turn = PipeTurnContext(sender, deadline=time.monotonic() + 30.0)
        turn.submit(LEGAL[0])
        turn.submit(LEGAL[1], final=True)
        turn.submit(LEGAL[2])
        sender.close()

        received = drain(receiver)
        receiver.close()
        assert received == [
            Submission(move=LEGAL[0], final=False),
            Submission(move=LEGAL[1], final=True),
        ]

    def test_remaining_seconds_clamps_to_zero_and_expiry_stops_sending(self) -> None:
        """An expired context reports no time and sends nothing at all."""
        receiver, sender = multiprocessing.Pipe(duplex=False)
        turn = PipeTurnContext(sender, deadline=time.monotonic() - 1.0)
        assert turn.remaining_seconds() == 0.0
        turn.submit(LEGAL[0], final=True)
        sender.close()
        assert drain(receiver) == []
        receiver.close()

    def test_a_broken_pipe_closes_the_turn_instead_of_raising(self) -> None:
        """The parent closing its endpoint is the end of the turn, not an error."""
        receiver, sender = multiprocessing.Pipe(duplex=False)
        receiver.close()
        turn = PipeTurnContext(sender, deadline=time.monotonic() + 30.0)
        turn.submit(LEGAL[0])
        turn.submit(LEGAL[1])
        sender.close()


def run_worker_body(kind: str, view: PlayerView, *, budget: float = 30.0) -> list[object]:
    """Run the worker body in this process and return everything it sent.

    No process is spawned: the point is the message protocol, not the lifecycle.

    Args:
        kind: Which scripted agent to build.
        view: The observation to hand the agent.
        budget: Seconds until the turn's deadline.

    Returns:
        The messages the worker sent, in order.
    """
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        run_agent_worker(
            sender,
            lambda: SCRIPTED_AGENTS[kind](seed=1),
            view,
            deadline=time.monotonic() + budget,
        )
        return drain(receiver)
    finally:
        receiver.close()


class TestWorkerProtocol:
    """What the worker body reports for each way a decision can end."""

    def test_a_finalizing_agent_sends_its_candidate_then_completion(
        self, setup_view: PlayerView
    ) -> None:
        """Completion is always announced, including after a final submission."""
        messages = run_worker_body("final", setup_view)
        assert messages == [
            Submission(move=setup_view.legal_moves[0], final=True),
            WorkerFinished(),
        ]

    def test_a_silent_agent_only_announces_completion(self, setup_view: PlayerView) -> None:
        """Returning without submitting is a completion, not a failure."""
        assert run_worker_body("silent", setup_view) == [WorkerFinished()]

    def test_repeated_submissions_are_all_sent_in_order(self, setup_view: PlayerView) -> None:
        """There is no application-level cap on candidates."""
        messages = run_worker_body("chatty", setup_view)
        assert messages[-1] == WorkerFinished()
        assert messages[:-1] == [
            Submission(move=move, final=False) for move in setup_view.legal_moves
        ]

    def test_an_agent_exception_is_reported_as_data(self, setup_view: PlayerView) -> None:
        """The worker boundary turns an agent exception into a small record."""
        messages = run_worker_body("raising", setup_view)
        assert messages == [
            Submission(move=setup_view.legal_moves[1], final=False),
            WorkerFailed(exception="RuntimeError", message="scripted failure"),
        ]

    def test_a_construction_failure_is_reported_like_any_agent_failure(
        self, setup_view: PlayerView
    ) -> None:
        """Building the agent inside the worker keeps its failures inside too."""
        assert run_worker_body("broken", setup_view) == [
            WorkerFailed(exception="ValueError", message="scripted construction failure")
        ]

    def test_an_expired_turn_sends_no_candidates(self, setup_view: PlayerView) -> None:
        """A locally expired context drops submissions; completion still travels."""
        assert run_worker_body("final", setup_view, budget=-1.0) == [WorkerFinished()]


class TestMatchConfig:
    """Validation of the timing and failure policy."""

    @pytest.mark.parametrize(
        "budget",
        [0.0, -1.0, math.nan, math.inf],
        ids=["zero", "negative", "nan", "infinity"],
    )
    def test_the_budget_must_be_positive_and_finite(self, budget: float) -> None:
        """NaN and infinity are rejected alongside non-positive budgets."""
        with pytest.raises(ValueError, match="positive and finite"):
            MatchConfig(seconds_per_turn=budget)

    def test_the_action_limit_must_be_positive(self) -> None:
        """A match must be allowed at least one play decision."""
        with pytest.raises(ValueError, match="max_play_decisions"):
            MatchConfig(max_play_decisions=0)

    def test_the_defaults_are_the_documented_ones(self) -> None:
        """The default budget and limit match the design's defaults."""
        config = MatchConfig()
        assert config.seconds_per_turn == 2.0
        assert config.max_play_decisions == 10_000
        assert not config.strict_failures


class TestRunnerConstruction:
    """Lineup validation, before any process is started."""

    def test_seats_must_be_zero_through_player_count(self) -> None:
        """Results index participants by seat, so the seats must be dense."""
        agents = {
            PlayerId(0): AgentSpec(kind="random", name="a"),
            PlayerId(2): AgentSpec(kind="random", name="b"),
        }
        with pytest.raises(ValueError, match="Agent seats must be exactly"):
            MatchRunner(agents, MatchConfig())

    def test_the_table_size_must_suit_the_profile(self) -> None:
        """A one-player lineup is refused by the profile's range."""
        with pytest.raises(ValueError, match="supports 2-5 players"):
            MatchRunner({PlayerId(0): AgentSpec(kind="random", name="a")}, MatchConfig())

    def test_only_the_fixed_profile_is_accepted(self) -> None:
        """A modified profile cannot be smuggled in through the runner."""
        agents = {
            PlayerId(0): AgentSpec(kind="random", name="a"),
            PlayerId(1): AgentSpec(kind="greedy", name="b"),
        }
        with pytest.raises(ValueError, match="is fixed"):
            MatchRunner(agents, MatchConfig(), rules=RulesConfig(refill_target=4))

    def test_a_view_without_legal_moves_is_refused(self) -> None:
        """Only the current actor's observation can be decided on."""
        agents = {
            PlayerId(0): AgentSpec(kind="random", name="a"),
            PlayerId(1): AgentSpec(kind="greedy", name="b"),
        }
        runner = MatchRunner(agents, MatchConfig())
        state = GameState.create(2, seed=0)
        actor = state.current_player
        assert actor is not None
        idle = state.observe(PlayerId(1 - actor))
        assert not idle.legal_moves
        with pytest.raises(StateInvariantError, match="no legal moves"):
            runner.choose_move(idle)


def turn_record(**changes: object) -> TurnRecord:
    """Build a turn record for bookkeeping tests.

    Args:
        **changes: Fields to override on an otherwise clean record.

    Returns:
        A record describing a clean decision, with the overrides applied.
    """
    clean = TurnRecord(
        decision_id=0,
        player=PlayerId(0),
        phase=Phase.PLAY,
        move=LEGAL[0],
        reason=CloseReason.FINAL,
        used_fallback=False,
        accepted=1,
        rejected=0,
        failure=None,
        budget_seconds=2.0,
        selection_seconds=0.2,
        cleanup_seconds=0.01,
        agent_seed=1234,
    )
    return replace(clean, **changes)  # type: ignore[arg-type]


class TestRecords:
    """What the records say about failures and about the applied stream."""

    @pytest.mark.parametrize(
        ("changes", "failed"),
        [
            ({}, False),
            ({"reason": CloseReason.DEADLINE}, False),
            ({"used_fallback": True}, True),
            ({"rejected": 1}, True),
            ({"failure": WorkerFailed(exception="RuntimeError", message="x")}, True),
        ],
        ids=["clean-final", "deadline-with-candidate", "fallback", "rejection", "crash"],
    )
    def test_strict_mode_counts_only_real_agent_failures(
        self, changes: dict[str, object], failed: bool
    ) -> None:
        """A deadline reached with a legal candidate is an ordinary decision."""
        assert turn_record(**changes).agent_failed is failed

    def test_unapplied_selections_stay_out_of_the_applied_stream(self) -> None:
        """A move selected but never applied is reported, not replayed."""
        applied = turn_record(decision_id=0)
        aborted = turn_record(decision_id=1, used_fallback=True)
        result = MatchResult(
            status=MatchStatus.AGENT_FAILED,
            outcome=None,
            turns=(applied, aborted),
            initial_events=(),
            decisions=(AppliedDecision(turn=applied, events=()),),
            play_decisions=1,
            failure="scripted",
            metadata=MatchMetadata(
                rules=DEFAULT_RULES,
                player_count=2,
                dealer=PlayerId(0),
                deal_seed=0,
                agents=(AgentSpec(kind="random", name="a"), AgentSpec(kind="greedy", name="b")),
                config=MatchConfig(),
            ),
        )
        assert result.unapplied_turns == (aborted,)


class ScriptedRunner(MatchRunner):
    """A runner whose decisions are dictated instead of spawned.

    Overriding the decision keeps these tests free of processes: the point is
    what :meth:`shed.match.MatchRunner.run` does *around* a decision.

    Attributes:
        _move: The move every decision selects.
    """

    def __init__(self, move: Move, config: MatchConfig) -> None:
        """Prepare a two-player runner that always selects one move.

        Args:
            move: The move to hand back for every decision.
            config: Timing, limits, and failure policy.
        """
        super().__init__(
            {
                PlayerId(0): AgentSpec(kind="random", name="a"),
                PlayerId(1): AgentSpec(kind="greedy", name="b"),
            },
            config,
        )
        self._move = move

    def choose_move(self, view: PlayerView) -> TurnRecord:
        """Return the scripted decision without starting a worker.

        Args:
            view: The actor's observation.

        Returns:
            A clean record naming the scripted move.
        """
        return turn_record(player=view.viewer, phase=view.phase, move=self._move)


class TestRunnerFailureHandling:
    """What a match does when the engine, rather than an agent, refuses."""

    def test_an_illegal_selection_ends_the_match_as_an_engine_failure(self) -> None:
        """The runner never turns a rejected move into a different one."""
        result = ScriptedRunner(PickUp(), MatchConfig()).run(deal_seed=0)

        assert result.status is MatchStatus.ENGINE_FAILED
        assert result.outcome is None
        assert result.decisions == ()
        assert len(result.turns) == 1
        assert result.unapplied_turns == result.turns
        assert result.failure is not None
        assert result.failure.startswith("IllegalMoveError")

    def test_a_deal_that_cannot_be_created_is_reported_to_the_caller(self) -> None:
        """Nothing has been played yet, so there is no match result to return."""
        runner = ScriptedRunner(PickUp(), MatchConfig())
        with pytest.raises(ValueError, match="Dealer 5"):
            runner.run(deal_seed=0, dealer=PlayerId(5))
