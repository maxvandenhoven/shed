"""Timed decisions: the process worker, the selection policy, and the runner.

This module owns everything the engine deliberately does not: clocks, worker
processes, pipe framing, message validation, and the records a match leaves
behind. The engine stays pure -- it never learns that a decision was timed --
and agents stay unaware of processes, because the only thing they touch here is
:class:`PipeTurnContext`, which implements the two-method turn protocol.

The layers are separable on purpose:

* :func:`select_candidate` is the selection policy. It takes a clock and a
  message source, so the whole contract in section 10.1 of the design can be
  tested without a process, a pipe, or a real deadline.
* :func:`run_agent_worker` is the worker body: build the agent, let it think
  against a pipe-backed turn, then report completion or failure.
* :class:`MatchRunner` owns the lifecycle -- one freshly spawned process with
  one fresh pipe and one fresh agent seed per decision -- and the match loop.

The trust model is the design's: local, cooperative Python agents sending small,
well-formed messages. Validation here rejects malformed or illegal submissions
and the parent independently enforces the deadline, but multiprocessing
deserialization is not a security boundary and this is not a sandbox for hostile
code. The budget is an *acceptance* deadline, not a promise that
:meth:`MatchRunner.choose_move` returns at that instant: spawning, scheduling,
message receipt, and reaping all add latency after it.

Workers are spawned rather than forked so a child starts from a fresh
interpreter and never inherits a copy of the parent's memory, which is where the
authoritative :class:`~shed.engine.GameState` lives. A worker receives only its
specification, a fresh seed, the observation, and its deadline; deck seeds,
fallback seeds, authoritative state, and results never cross the boundary.
"""

from __future__ import annotations

import multiprocessing
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Protocol

from shed.agents import Agent, AgentSpec, build_agent
from shed.engine import (
    DEFAULT_DEALER,
    DEFAULT_RULES,
    Arrange,
    GameState,
    IllegalMoveError,
    Move,
    ObservedEvent,
    Outcome,
    Phase,
    PickUp,
    Play,
    PlayerId,
    PlayerView,
    Reveal,
    RulesConfig,
    StateInvariantError,
    filter_events_for,
)

__all__ = [
    "AppliedDecision",
    "CloseReason",
    "MatchConfig",
    "MatchMetadata",
    "MatchResult",
    "MatchRunner",
    "MatchStatus",
    "MessageSource",
    "PipeTurnContext",
    "Selection",
    "Submission",
    "TurnRecord",
    "WorkerFailed",
    "WorkerFinished",
    "WorkerMessage",
    "run_agent_worker",
    "select_candidate",
]

CLEANUP_GRACE_SECONDS = 0.1
"""Seconds a terminated worker is given to exit before it is killed."""

SEED_SPACE = 2**32
"""Range per-decision agent seeds are drawn from."""

MAX_FAILURE_MESSAGE = 500
"""Characters kept from an agent exception message; tracebacks are never sent."""

MOVE_TYPES: tuple[type, ...] = (Arrange, Play, Reveal, PickUp)
"""Concrete move classes, for the shape check at the transport boundary.

``Move`` is a type alias rather than a class, so it cannot be handed to
``isinstance``; this tuple is the runtime spelling of the same union.
"""


@dataclass(frozen=True, slots=True)
class Submission:
    """One candidate move a worker offers the runner.

    Attributes:
        move: The proposed move. It is validated by the parent, which stays the
            legality authority, so an illegal proposal is simply rejected.
        final: Whether this candidate closes the turn.
    """

    move: Move
    final: bool = False


@dataclass(frozen=True, slots=True)
class WorkerFinished:
    """Sent once ``think()`` returns normally; the turn closes on receipt."""


@dataclass(frozen=True, slots=True)
class WorkerFailed:
    """Sent when the agent raised, and recorded as the decision's failure.

    Only the exception's type name and a truncated message travel: a worker
    never ships a traceback or arbitrary objects across the pipe.

    Attributes:
        exception: Name of the exception class the agent raised.
        message: The exception's message, truncated to
            :data:`MAX_FAILURE_MESSAGE` characters.
    """

    exception: str
    message: str


type WorkerMessage = Submission | WorkerFinished | WorkerFailed


class MessageSource(Protocol):
    """The half of a receiving endpoint the selection policy actually uses.

    A :class:`multiprocessing.connection.Connection` satisfies it, and so does a
    scripted in-memory fake, which is what lets the policy be tested without a
    process.
    """

    def poll(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for a message to become readable.

        Args:
            timeout: Seconds to wait.

        Returns:
            Whether a message can be received without blocking.
        """

    def recv(self) -> object:
        """Receive one complete message.

        Returns:
            The deserialized object, which is untrusted and must be validated.
        """


def _send(sender: Connection, message: WorkerMessage) -> bool:
    """Send one message, treating a dead pipe as a closed turn.

    The parent closes its endpoint as soon as a decision is over, so a worker
    that is still talking will see the pipe break. That is the normal end of a
    turn, not an agent error, and it must not crash worker cleanup.

    Args:
        sender: The worker's sending endpoint.
        message: The message to send.

    Returns:
        Whether the message was handed to the pipe.
    """
    try:
        sender.send(message)
    except (BrokenPipeError, EOFError, OSError, ValueError):
        return False
    return True


class PipeTurnContext:
    """The concrete turn an agent holds inside a worker process.

    It mirrors the agent-facing protocol exactly -- submit a candidate, ask how
    much time is left -- and holds no legal moves of its own: those stay on the
    observation the worker was given, so the boundary carries one copy of them.

    The local expiry and closed checks are a cooperative optimisation, not the
    enforcement mechanism: the parent independently stops the worker.

    Attributes:
        _sender: The worker's sending endpoint.
        _deadline: Monotonic instant this decision's budget ends.
        _closed: Whether this context has stopped sending, because a final
            candidate was submitted or the pipe broke.
    """

    def __init__(self, sender: Connection, *, deadline: float) -> None:
        """Open a turn on one sending endpoint.

        Args:
            sender: The worker's end of the decision's pipe.
            deadline: Monotonic instant the budget ends, measured by the parent
                immediately before the process was started.
        """
        self._sender = sender
        self._deadline = deadline
        self._closed = False

    def remaining_seconds(self) -> float:
        """Return the time remaining in this decision's budget.

        Returns:
            Seconds until the monotonic deadline, clamped to zero. This value
            is a cooperative hint; the match runner enforces the deadline.
        """
        return max(0.0, self._deadline - time.monotonic())

    def submit(self, move: Move, *, final: bool = False) -> None:
        """Offer one candidate to the runner, unless the turn is already over.

        Nothing is acknowledged and nothing is raised: a submission that arrives
        late, is illegal, or cannot be sent at all simply leaves the previous
        candidate standing.

        Args:
            move: The candidate move.
            final: Whether this candidate closes the turn. After a final
                submission this context sends nothing further, so an agent that
                keeps computing cannot change its own decision.
        """
        if self._closed or self.remaining_seconds() <= 0.0:
            return
        sent = _send(self._sender, Submission(move=move, final=final))
        if final or not sent:
            self._closed = True


def run_agent_worker(
    sender: Connection,
    build: Callable[[], Agent],
    view: PlayerView,
    *,
    deadline: float,
) -> None:
    """Build one agent, let it think, and report how the decision ended.

    This is the worker body, shared by the spawned entry point and by tests that
    script an agent's behaviour. The agent is constructed here rather than
    passed in, so no live object and no generator state ever crosses a process
    boundary.

    The broad exception handler is the worker boundary the design asks for: an
    agent raising is ordinary strategy failure, reported to the parent as data
    rather than propagated as a crash. ``BaseException`` is deliberately not
    caught -- an interrupted or terminated worker is the parent's business, and
    the parent sees the pipe close.

    Args:
        sender: The worker's end of the decision's pipe. It is closed before
            this function returns, whatever happened.
        build: Zero-argument factory for the agent, called inside the worker so
            construction failures are reported like any other agent failure.
        view: The observation to decide on; its ``legal_moves`` are the agent's
            options.
        deadline: Monotonic instant the budget ends.
    """
    try:
        try:
            build().think(view, PipeTurnContext(sender, deadline=deadline))
        except Exception as error:
            _send(
                sender,
                WorkerFailed(
                    exception=type(error).__name__,
                    message=str(error)[:MAX_FAILURE_MESSAGE],
                ),
            )
        else:
            _send(sender, WorkerFinished())
    finally:
        sender.close()


def _worker_main(
    sender: Connection,
    spec: AgentSpec,
    view: PlayerView,
    seed: int,
    deadline: float,
) -> None:
    """Spawned entry point for one decision.

    Kept at module scope with serializable arguments, because a spawned child
    imports this module and rebuilds its arguments by unpickling them.

    Args:
        sender: The worker's end of the decision's pipe.
        spec: Which agent to build.
        seed: Fresh seed for this decision's agent; never a deck or fallback
            seed.
        view: The observation to decide on.
        deadline: Monotonic instant the budget ends.
    """
    run_agent_worker(sender, partial(build_agent, spec, seed=seed), view, deadline=deadline)


class CloseReason(Enum):
    """Why a decision stopped accepting messages.

    Attributes:
        FINAL: A legal final candidate was accepted.
        RETURNED: The worker reported that ``think()`` returned.
        FAILED: The worker reported an agent exception, or died without
            reporting anything.
        DEADLINE: The budget ran out first.
    """

    FINAL = "final"
    RETURNED = "returned"
    FAILED = "failed"
    DEADLINE = "deadline"


@dataclass(frozen=True, slots=True)
class Selection:
    """What the selection policy decided for one turn.

    Attributes:
        move: The move to apply: the latest accepted candidate, or the seeded
            fallback when nothing was accepted.
        reason: Why the decision closed.
        used_fallback: Whether ``move`` came from the fallback stream rather
            than from the agent.
        accepted: How many candidates were accepted, including the final one.
        rejected: How many messages were rejected as malformed or illegal.
        failure: The worker's failure record, if it reported or suffered one. A
            failure is recorded even when an earlier candidate is still used.
    """

    move: Move
    reason: CloseReason
    used_fallback: bool
    accepted: int
    rejected: int
    failure: WorkerFailed | None


def _canonical_candidate(message: Submission, legal_moves: tuple[Move, ...]) -> Move | None:
    """Validate one submission and return the legal move it names.

    This is the transport boundary, so the checks are about the message rather
    than the game: a well-formed final flag, a recognised move shape, and
    membership in the tuple frozen when the decision opened. Legality is decided
    against that tuple alone -- nothing is recomputed, and the engine revalidates
    independently when the move is applied.

    Args:
        message: A received submission, whose fields are untrusted.
        legal_moves: The moves frozen for this decision.

    Returns:
        The engine's own instance of the submitted move, or ``None`` if the
        submission is rejected.
    """
    if not isinstance(message.final, bool) or not isinstance(message.move, MOVE_TYPES):
        return None
    if message.move not in legal_moves:
        return None
    return legal_moves[legal_moves.index(message.move)]


def select_candidate(
    legal_moves: tuple[Move, ...],
    messages: MessageSource,
    *,
    deadline: float,
    fallback_rng: random.Random,
    clock: Callable[[], float] = time.monotonic,
) -> Selection:
    """Run the selection policy for one decision.

    The policy keeps the latest legal candidate and closes on the first of: a
    legal final candidate, the worker reporting completion or failure, or the
    deadline. A message is eligible only if a *complete* message was received
    strictly before the deadline, measured on this clock; equality with the
    deadline is late, and buffered messages are never drained past expiry to
    find a newer one. Rejected messages leave the previous candidate standing.

    No game state is touched: the caller froze ``legal_moves`` before the worker
    started, and exactly one move comes back out.

    Args:
        legal_moves: The non-empty tuple frozen for this decision.
        messages: Where complete messages arrive from.
        deadline: Monotonic instant the budget ends, on ``clock``'s scale.
        fallback_rng: Dedicated generator for the seeded legal fallback. It is
            advanced only when a fallback is actually needed.
        clock: Monotonic clock, injectable so the policy can be tested without
            real time.

    Returns:
        The selected move and the diagnostics for this decision.

    Raises:
        ValueError: If no legal moves were supplied, which means the caller
            asked for a decision that does not exist.
    """
    if not legal_moves:
        raise ValueError("A decision needs at least one legal move")

    latest: Move | None = None
    accepted = 0
    rejected = 0
    failure: WorkerFailed | None = None
    reason = CloseReason.DEADLINE

    while True:
        remaining = deadline - clock()
        if remaining <= 0.0 or not messages.poll(remaining):
            break
        try:
            message = messages.recv()
        except (EOFError, OSError) as error:
            # The worker vanished without reporting: a crash, not a decision.
            failure = WorkerFailed(exception=type(error).__name__, message=str(error))
            reason = CloseReason.FAILED
            break
        if clock() >= deadline:
            break  # Received too late: the message is ignored entirely.

        match message:
            case WorkerFinished():
                reason = CloseReason.RETURNED
                break
            case WorkerFailed():
                failure = message
                reason = CloseReason.FAILED
                break
            case Submission():
                candidate = _canonical_candidate(message, legal_moves)
                if candidate is None:
                    rejected += 1
                    continue
                latest = candidate
                accepted += 1
                if message.final:
                    reason = CloseReason.FINAL
                    break
            case _:
                rejected += 1

    return Selection(
        move=latest if latest is not None else fallback_rng.choice(legal_moves),
        reason=reason,
        used_fallback=latest is None,
        accepted=accepted,
        rejected=rejected,
        failure=failure,
    )


def _shutdown_worker(
    process: SpawnProcess,
    *endpoints: Connection,
    grace: float = CLEANUP_GRACE_SECONDS,
) -> None:
    """Stop a worker and release every resource the decision held.

    Polling inside a worker cannot interrupt an infinite loop, so the parent
    stops it here: terminate, wait briefly, then kill. Process resources are
    released only once the process has actually stopped, and the pipe is
    discarded rather than reused, so no message can survive into a later turn.

    Args:
        process: The decision's worker. An unstarted process is only closed.
        *endpoints: Pipe endpoints to close; closing twice is harmless.
        grace: Seconds to wait after termination before killing the worker.
    """
    try:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(grace)
            if process.is_alive():
                process.kill()
            process.join()
    finally:
        for endpoint in endpoints:
            endpoint.close()
        process.close()


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """Everything one decision produced besides the move itself.

    Attributes:
        decision_id: Sequential index of this decision within the match,
            counting setup decisions and extra turns after a burn.
        player: Who decided.
        phase: Phase the decision was made in.
        move: The selected move.
        reason: Why the decision closed.
        used_fallback: Whether the move came from the fallback stream because
            no candidate was accepted.
        accepted: Accepted candidates, including the final one.
        rejected: Messages rejected as malformed or illegal.
        failure: The worker's failure record, if there was one.
        budget_seconds: The acceptance budget this decision was given.
        selection_seconds: Wall time from starting the worker to closing the
            decision.
        cleanup_seconds: Wall time spent stopping the worker and closing its
            pipe afterwards.
        agent_seed: Seed the agent was built with for this decision.
    """

    decision_id: int
    player: PlayerId
    phase: Phase
    move: Move
    reason: CloseReason
    used_fallback: bool
    accepted: int
    rejected: int
    failure: WorkerFailed | None
    budget_seconds: float
    selection_seconds: float
    cleanup_seconds: float
    agent_seed: int

    @property
    def agent_failed(self) -> bool:
        """Whether strict mode treats this decision as an agent failure.

        A deadline reached with a legal candidate is an ordinary decision in
        either mode; a rejected submission, a worker failure, or the absence of
        any accepted candidate is not.
        """
        return self.used_fallback or self.rejected > 0 or self.failure is not None


class MatchStatus(Enum):
    """How a match ended, independently of the engine's phase.

    Attributes:
        FINISHED: The rules ended the game; the result carries the outcome.
        TRUNCATED: The action limit stopped a potentially cyclic game. There is
            no winner and none is invented.
        AGENT_FAILED: Strict mode aborted the match on an agent failure.
        ENGINE_FAILED: The engine or the runner's infrastructure failed.
    """

    FINISHED = "finished"
    TRUNCATED = "truncated"
    AGENT_FAILED = "agent_failed"
    ENGINE_FAILED = "engine_failed"


@dataclass(frozen=True, slots=True)
class AppliedDecision:
    """One decision that was actually applied, with what it resolved into.

    Attributes:
        turn: The selection record for the decision.
        events: The full events the transition emitted, identities intact.
            These are trusted post-match data: filter them before they reach an
            agent.
    """

    turn: TurnRecord
    events: tuple[ObservedEvent, ...]


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Timing, limits, and failure policy for one match.

    Attributes:
        seconds_per_turn: Acceptance budget per decision. Process startup,
            argument transport, and agent construction all count against it.
        max_play_decisions: Bound on resolved PLAY decisions. Reaching it
            truncates the match; it is never a rules-level draw.
        fallback_seed: Seed of the fallback stream, used only when a decision
            accepts no candidate.
        agent_seed: Seed of the stream that draws one fresh agent seed per
            decision. It is independent of the deck seed, so no agent can infer
            the deal from its own seed.
        strict_failures: Whether an agent failure aborts the match instead of
            continuing from the latest candidate or a fallback.
    """

    seconds_per_turn: float = 2.0
    max_play_decisions: int = 10_000
    fallback_seed: int = 0
    agent_seed: int = 1
    strict_failures: bool = False

    def __post_init__(self) -> None:
        """Check the budget and the action limit are usable.

        Raises:
            ValueError: If the budget is not strictly positive and finite --
                NaN and infinity are rejected here -- or the action limit is not
                positive.
        """
        budget = self.seconds_per_turn
        if not budget > 0.0 or budget == float("inf"):
            raise ValueError(f"seconds_per_turn must be positive and finite, got {budget!r}")
        if self.max_play_decisions < 1:
            raise ValueError(f"max_play_decisions must be positive, got {self.max_play_decisions}")


@dataclass(frozen=True, slots=True)
class MatchMetadata:
    """What a match was set up with, for results and for the replay writer.

    The deck seed is trusted metadata and never reaches an agent. The shuffled
    deck order itself is not stored: it is reconstructed deterministically from
    this seed with :func:`shed.engine.shuffled_deck`, the same helper the runner
    dealt with. The replay writer adds the schema version, the package and
    Python versions, and the deck order it records alongside the seed.

    Attributes:
        rules: The profile the match ran under.
        player_count: Number of seats.
        dealer: The dealing seat.
        deal_seed: Deck seed the deal came from.
        agents: Participants in seat order; seat ``i`` played ``agents[i]``.
        config: Timing, limits, and failure policy.
    """

    rules: RulesConfig
    player_count: int
    dealer: PlayerId
    deal_seed: int
    agents: tuple[AgentSpec, ...]
    config: MatchConfig


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Everything one match produced.

    Attributes:
        status: How the match ended; separate from the engine's phase, so a
            truncated or aborted match never carries a fabricated outcome.
        outcome: The winner, present only when the rules finished the game.
        turns: Every selection in order, applied or not.
        initial_events: The full events describing the deal.
        decisions: The applied-decision stream, in order. A move selected but
            not applied -- strict mode aborts before applying -- is in
            :attr:`turns` and deliberately absent here.
        play_decisions: Applied PLAY decisions; setup does not count.
        failure: Human-readable detail for a non-finished status.
        metadata: How the match was set up.
    """

    status: MatchStatus
    outcome: Outcome | None
    turns: tuple[TurnRecord, ...]
    initial_events: tuple[ObservedEvent, ...]
    decisions: tuple[AppliedDecision, ...]
    play_decisions: int
    failure: str | None
    metadata: MatchMetadata

    @property
    def unapplied_turns(self) -> tuple[TurnRecord, ...]:
        """Return the selections that were never applied to the game.

        Returns:
            The records missing from :attr:`decisions`, in order. There is at
            most one: a strict-mode abort stops the match immediately.
        """
        applied = {decision.turn.decision_id for decision in self.decisions}
        return tuple(turn for turn in self.turns if turn.decision_id not in applied)


class MatchRunner:
    """Runs timed matches: one worker, one pipe, and one agent seed per decision.

    The runner is the only place where timing, processes, and the game meet. It
    freezes the actor's legal moves once, spawns a worker to think about them,
    selects exactly one move, and applies it through the engine, which
    revalidates independently. Filtered history is kept here, outside the state,
    and each seat sees only its own.

    Attributes:
        _agents: Participant per seat.
        _config: Timing, limits, and failure policy.
        _rules: The profile matches run under.
        _context: The spawn context every worker is started from.
        _decision_id: Sequential decision counter for records.
        _agent_seeds: Stream of per-decision agent seeds.
        _fallback: Stream the seeded legal fallback is drawn from.
    """

    def __init__(
        self,
        agents: dict[PlayerId, AgentSpec],
        config: MatchConfig,
        *,
        rules: RulesConfig = DEFAULT_RULES,
    ) -> None:
        """Prepare a runner for a fixed lineup.

        Args:
            agents: Participant per seat. Seats must be exactly
                ``0..len(agents)-1``, which is what lets results index
                participants by seat.
            config: Timing, limits, and failure policy.
            rules: Rules profile; only the fixed ``shed-v1`` profile exists.

        Raises:
            ValueError: If the profile is unsupported, the table size is outside
                the profile's range, or the seats are not ``0..len(agents)-1``.
        """
        rules.validate()
        count = len(agents)
        if not rules.min_players <= count <= rules.max_players:
            raise ValueError(
                f"{rules.id} supports {rules.min_players}-{rules.max_players} players, got {count}"
            )
        if sorted(agents) != list(range(count)):
            raise ValueError(f"Agent seats must be exactly 0..{count - 1}, got {sorted(agents)}")

        self._agents = dict(agents)
        self._config = config
        self._rules = rules
        self._context: SpawnContext = multiprocessing.get_context("spawn")
        self._decision_id = 0
        self._agent_seeds = random.Random(config.agent_seed)
        self._fallback = random.Random(config.fallback_seed)

    def choose_move(self, view: PlayerView) -> TurnRecord:
        """Run one timed decision and return what it selected.

        The observation's ``legal_moves`` tuple is frozen as the acceptance set
        for this decision: candidates are accepted by membership in it, and it
        is never recomputed. The worker is spawned with only the specification,
        a fresh seed, this observation, and its deadline. The budget starts
        immediately before the process starts, so startup counts against it, and
        the worker is stopped and reaped on every path out of this method.

        Args:
            view: The actor's observation, carrying its legal moves.

        Returns:
            The record for this decision, including the move to apply.

        Raises:
            StateInvariantError: If the observation offers no legal moves, which
                means it does not belong to the current actor.
            OSError: If the worker cannot be started at all. That is
                infrastructure failing, not an agent misbehaving.
        """
        legal_moves = view.legal_moves
        if not legal_moves:
            raise StateInvariantError(
                f"Player {view.viewer} was asked to decide with no legal moves"
            )
        decision_id = self._decision_id
        self._decision_id += 1
        seed = self._agent_seeds.randrange(SEED_SPACE)
        budget = self._config.seconds_per_turn

        receiver, sender = self._context.Pipe(duplex=False)
        selection_started = time.perf_counter()
        deadline = time.monotonic() + budget
        # Daemonic: the worker cannot outlive this process and cannot start
        # children of its own, which terminating a single worker would not reap.
        process = self._context.Process(
            target=_worker_main,
            args=(sender, self._agents[view.viewer], view, seed, deadline),
            daemon=True,
        )
        try:
            process.start()
            sender.close()  # The parent's copy, so the worker's EOF is visible.
            selection = select_candidate(
                legal_moves,
                receiver,
                deadline=deadline,
                fallback_rng=self._fallback,
            )
        finally:
            cleanup_started = time.perf_counter()
            _shutdown_worker(process, receiver, sender)
            cleanup_seconds = time.perf_counter() - cleanup_started

        return TurnRecord(
            decision_id=decision_id,
            player=view.viewer,
            phase=view.phase,
            move=selection.move,
            reason=selection.reason,
            used_fallback=selection.used_fallback,
            accepted=selection.accepted,
            rejected=selection.rejected,
            failure=selection.failure,
            budget_seconds=budget,
            selection_seconds=cleanup_started - selection_started,
            cleanup_seconds=cleanup_seconds,
            agent_seed=seed,
        )

    def run(self, *, deal_seed: int, dealer: PlayerId = DEFAULT_DEALER) -> MatchResult:
        """Play one complete timed match.

        The deal is created here from ``deal_seed``, which never leaves this
        process. Filtered histories start from the deal's events and grow by one
        filtered batch per applied decision, so each seat remembers exactly what
        it was allowed to see.

        Both seed streams restart at the start of every match, so a match is
        reproducible from this runner's configuration -- though the *timed*
        decisions are not, because scheduling decides how much an agent finishes.

        Args:
            deal_seed: Deck seed; trusted metadata, never shown to an agent.
            dealer: Dealing seat.

        Returns:
            The match result, whatever ended it.

        Raises:
            ValueError: If the deal itself cannot be created, for instance from
                an out-of-range dealer. Nothing has been played at that point,
                so it is reported to the caller rather than as a match result.
        """
        self._decision_id = 0
        self._agent_seeds = random.Random(self._config.agent_seed)
        self._fallback = random.Random(self._config.fallback_seed)

        player_count = len(self._agents)
        state = GameState.create(player_count, seed=deal_seed, dealer=dealer, rules=self._rules)
        metadata = MatchMetadata(
            rules=self._rules,
            player_count=player_count,
            dealer=dealer,
            deal_seed=deal_seed,
            agents=tuple(self._agents[PlayerId(seat)] for seat in range(player_count)),
            config=self._config,
        )
        initial_events = state.initial_events()
        histories = {
            seat: list(filter_events_for(initial_events, seat)) for seat in state.seat_order
        }

        turns: list[TurnRecord] = []
        decisions: list[AppliedDecision] = []
        play_decisions = 0
        status = MatchStatus.FINISHED
        failure: str | None = None

        try:
            while not state.is_finished:
                if play_decisions >= self._config.max_play_decisions:
                    status = MatchStatus.TRUNCATED
                    failure = f"Stopped after {play_decisions} play decisions"
                    break
                actor = state.current_player
                if actor is None:
                    raise StateInvariantError("A live match always has an actor")

                view = state.observe(actor, history=tuple(histories[actor]))
                record = self.choose_move(view)
                turns.append(record)

                if self._config.strict_failures and record.agent_failed:
                    status = MatchStatus.AGENT_FAILED
                    failure = self._failure_detail(record)
                    break

                was_play = state.phase is Phase.PLAY
                transition = state.apply_move(record.move)
                if was_play:
                    play_decisions += 1
                decisions.append(AppliedDecision(turn=record, events=transition.events))
                for seat in state.seat_order:
                    histories[seat].extend(filter_events_for(transition.events, seat))
        except (IllegalMoveError, StateInvariantError, OSError) as error:
            status = MatchStatus.ENGINE_FAILED
            failure = f"{type(error).__name__}: {error}"

        return MatchResult(
            status=status,
            outcome=state.outcome if status is MatchStatus.FINISHED else None,
            turns=tuple(turns),
            initial_events=initial_events,
            decisions=tuple(decisions),
            play_decisions=play_decisions,
            failure=failure,
            metadata=metadata,
        )

    @staticmethod
    def _failure_detail(record: TurnRecord) -> str:
        """Describe why strict mode refused a decision.

        Args:
            record: The offending decision's record.

        Returns:
            A one-line description naming the decision and what went wrong.
        """
        reasons: list[str] = []
        if record.failure is not None:
            reasons.append(
                f"worker failed with {record.failure.exception}: {record.failure.message}"
            )
        if record.rejected:
            reasons.append(f"{record.rejected} rejected submission(s)")
        if record.used_fallback:
            reasons.append("no candidate was accepted")
        return f"Decision {record.decision_id} by player {record.player}: " + "; ".join(reasons)
