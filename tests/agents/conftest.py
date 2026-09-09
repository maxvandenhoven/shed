"""Fakes and helpers shared by the agent tests.

Two things live here. :class:`FakeTurn` is an in-memory ``TurnContext`` that
captures submissions and lets a test dictate the clock, so baselines can be
driven without a process, a pipe, or a real deadline.
:func:`play_baseline_match` is the tests' synchronous stand-in for the match
runner: it drives complete games straight through the engine, building a fresh
agent per decision and keeping each seat's filtered history, with no timing or
production runner code involved.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest

from shed.agents import AgentSpec, build_agent
from shed.engine import (
    GameState,
    Move,
    ObservedEvent,
    Outcome,
    Phase,
    PlayerId,
    PlayerView,
    filter_events_for,
)

DEFAULT_ACTION_LIMIT = 5_000
"""Test-only bound on decisions in one match; a truncation, never a draw."""

SEED_SPACE = 2**32
"""Range the per-decision agent seeds are drawn from."""


@dataclass(frozen=True, slots=True)
class Submission:
    """One candidate an agent offered during a decision.

    Attributes:
        move: The submitted move.
        final: Whether the submission closed the turn.
    """

    move: Move
    final: bool


class FakeTurn:
    """An in-memory ``TurnContext`` that records submissions and fakes the clock.

    The fake is stricter than the real pipe-backed context on purpose: the real
    one silently ignores anything sent after a final submission, while this one
    fails the test, so a baseline that keeps talking after finalizing is caught.

    Attributes:
        submissions: Everything the agent offered, in order.
        remaining: Seconds the agent is told it has left. A test may change it
            between calls to drive an agent that watches the clock.
        closed: Whether a final submission has closed the turn.
    """

    def __init__(self, remaining: float = 1.0) -> None:
        """Open a turn with a fixed amount of fake time.

        Args:
            remaining: Seconds :meth:`remaining_seconds` reports until a test
                changes it.
        """
        self.submissions: list[Submission] = []
        self.remaining = remaining
        self.closed = False

    def remaining_seconds(self) -> float:
        """Return the fake time left in this decision's budget.

        Returns:
            :attr:`remaining`, clamped to zero like the real context.
        """
        return max(0.0, self.remaining)

    def submit(self, move: Move, *, final: bool = False) -> None:
        """Record one candidate.

        Args:
            move: The candidate move.
            final: Whether this candidate closes the turn.

        Raises:
            AssertionError: If the agent submits after finalizing.
        """
        assert not self.closed, f"{move!r} was submitted after the turn was finalized"
        self.submissions.append(Submission(move=move, final=final))
        self.closed = final

    @property
    def selected(self) -> Move:
        """Return the move a runner would select from this turn.

        Returns:
            The latest submitted move.

        Raises:
            AssertionError: If the agent submitted nothing, or never finalized.
        """
        assert self.submissions, "The agent submitted nothing"
        assert self.closed, "The agent never submitted a final move"
        return self.submissions[-1].move


@dataclass(frozen=True, slots=True)
class DecisionLog:
    """What one decision of a synchronous match looked like.

    Attributes:
        decision_id: Position of this decision in the match, counting setup.
        player: Who decided.
        phase: Phase the decision was made in.
        seed: Seed the agent was constructed with.
        view: The observation the agent was given.
        submissions: Everything the agent offered.
        move: The move that was applied.
    """

    decision_id: int
    player: PlayerId
    phase: Phase
    seed: int
    view: PlayerView
    submissions: tuple[Submission, ...]
    move: Move


@dataclass(slots=True)
class MatchLog:
    """What one synchronous match produced.

    Attributes:
        state: The final state, mutated in place by the match.
        decisions: Every decision in order.
        histories: Each seat's filtered history at the end of the match.
        truncated: Whether the action bound stopped the match before it
            finished. A truncated match has no winner.
    """

    state: GameState
    decisions: list[DecisionLog] = field(default_factory=list)
    histories: dict[PlayerId, list[ObservedEvent]] = field(default_factory=dict)
    truncated: bool = False

    @property
    def outcome(self) -> Outcome | None:
        """Return the match result, or ``None`` if it truncated or is unfinished."""
        return self.state.outcome


def play_baseline_match(
    specs: Sequence[AgentSpec],
    *,
    deal_seed: int,
    agent_seed: int = 0,
    action_limit: int = DEFAULT_ACTION_LIMIT,
    inspect: Callable[[GameState, PlayerView], None] | None = None,
) -> MatchLog:
    """Play one complete match synchronously through the engine.

    The loop is the whole first-release contract minus timing: create the state,
    filter the opening events per seat, observe the actor with that seat's
    history, build a fresh agent from its specification and a fresh seed, let it
    think against a :class:`FakeTurn`, apply what it finalized, then append the
    filtered transition events to every seat's history.

    Per-decision seeds come from a dedicated generator seeded with
    ``agent_seed``, independent of the deck seed, so no agent can learn the deal
    from its own seed and no two decisions share a generator state.

    Args:
        specs: Participants by seat; seat ``i`` plays ``specs[i]``.
        deal_seed: Deck seed. It is trusted metadata: it reaches
            :meth:`GameState.create` and nothing else.
        agent_seed: Seed of the per-decision agent-seed stream.
        action_limit: Test-only bound on decisions; reaching it truncates.
        inspect: Optional hook called with the live state and the actor's view
            before each move is applied, for assertions about what an agent can
            see at that moment.

    Returns:
        The match log, including every decision and the final state.

    Raises:
        AssertionError: If a decision's finalized move is not one the view
            offered, which the engine would reject anyway.
    """
    state = GameState.create(len(specs), seed=deal_seed)
    seeds = random.Random(agent_seed)
    log = MatchLog(
        state=state,
        histories={
            seat: list(filter_events_for(state.initial_events(), seat)) for seat in state.seat_order
        },
    )

    while not state.is_finished:
        if len(log.decisions) >= action_limit:
            log.truncated = True
            break
        actor = state.current_player
        assert actor is not None, "A live match always has an actor"

        view = state.observe(actor, history=tuple(log.histories[actor]))
        if inspect is not None:
            inspect(state, view)

        seed = seeds.randrange(SEED_SPACE)
        agent = build_agent(specs[actor], seed=seed)
        turn = FakeTurn()
        agent.think(view, turn)
        move = turn.selected
        assert move in view.legal_moves, f"{move!r} was not offered to player {actor}"

        log.decisions.append(
            DecisionLog(
                decision_id=len(log.decisions),
                player=actor,
                phase=state.phase,
                seed=seed,
                view=view,
                submissions=tuple(turn.submissions),
                move=move,
            )
        )
        transition = state.apply_move(move)
        for seat in state.seat_order:
            log.histories[seat].extend(filter_events_for(transition.events, seat))

    return log


@pytest.fixture
def turn() -> FakeTurn:
    """Provide a fresh in-memory turn context for one decision."""
    return FakeTurn()
