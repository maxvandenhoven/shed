"""Tests for the common agent interface, the specification, and the factory.

These cover the shape every strategy shares -- how a turn is finalized, what an
agent may see, and how one is constructed -- rather than either baseline's
choices, which have their own modules.
"""

import inspect
from dataclasses import fields, replace

import pytest

from shed.agents import (
    AGENT_KINDS,
    Agent,
    AgentSpec,
    GreedyAgent,
    RandomAgent,
    TurnContext,
    build_agent,
    legal_choices,
)
from shed.engine import GameState, Play, PlayerId, PlayerView, Rank, SlotId, Zone
from tests.agents.conftest import FakeTurn
from tests.conftest import DeckPicker, build_play_state

SECOND_SEAT: PlayerId = PlayerId(1)
"""The opponent in the crafted two-player positions here."""


def _actor_view(state: GameState) -> PlayerView:
    """Observe the state as its current actor.

    Args:
        state: A live state.

    Returns:
        The actor's view, carrying their legal moves.
    """
    actor = state.current_player
    assert actor is not None
    return state.observe(actor)


def _every_agent(seed: int = 7) -> tuple[Agent, ...]:
    """Build one of every built-in agent.

    Args:
        seed: Seed handed to each agent.

    Returns:
        One freshly constructed agent per kind, in :data:`AGENT_KINDS` order.
    """
    return tuple(build_agent(AgentSpec(kind=kind, name=kind), seed=seed) for kind in AGENT_KINDS)


def test_spec_is_frozen_and_hashable() -> None:
    """Specifications are values: comparable, hashable, and immutable."""
    spec = AgentSpec(kind="random", name="random-1")

    assert spec == AgentSpec(kind="random", name="random-1")
    assert len({spec, AgentSpec(kind="random", name="random-1")}) == 1
    with pytest.raises(AttributeError):
        spec.name = "other"  # ty: ignore[invalid-assignment]


def test_spec_carries_no_seed() -> None:
    """A specification names a strategy and a label, and nothing secret.

    A deck or fallback seed stored here would travel to every worker with the
    spec, so the field list itself is part of the information boundary.
    """
    assert tuple(item.name for item in fields(AgentSpec)) == ("kind", "name")


def test_spec_rejects_unknown_kinds_and_empty_labels() -> None:
    """A mistyped lineup fails where it is written, not inside a worker."""
    with pytest.raises(ValueError, match="Unknown agent kind"):
        AgentSpec(kind="mcts", name="search")
    with pytest.raises(ValueError, match="non-empty"):
        AgentSpec(kind="random", name="")


def test_factory_builds_every_declared_kind() -> None:
    """Every advertised kind has a builder, and each builds its own class."""
    built = _every_agent()

    assert {type(agent) for agent in built} == {GreedyAgent, RandomAgent}
    assert all(isinstance(agent, Agent) for agent in built)


def test_factory_builds_a_fresh_agent_each_time() -> None:
    """Two builds share no object, so no generator state survives a decision."""
    spec = AgentSpec(kind="random", name="random-1")

    first = build_agent(spec, seed=11)
    second = build_agent(spec, seed=11)

    assert first is not second
    assert first._rng is not second._rng


def test_turn_context_exposes_only_submission_and_time() -> None:
    """The turn is a channel: no legal moves, no state, no selection feedback.

    Legal choices come from the view, so a duplicate on the turn would be a
    second source of truth for the same tuple.
    """
    members = {name for name in vars(TurnContext) if not name.startswith("_")}

    assert members == {"remaining_seconds", "submit"}

    # The fake implements exactly those two members, so the annotation below is
    # also a static check that nothing more is required of an implementation.
    context: TurnContext = FakeTurn()
    assert context.remaining_seconds() == pytest.approx(1.0)


def test_think_receives_only_a_view_and_a_turn() -> None:
    """The interface is the information boundary: no state, ruleset, or runner."""
    parameters = inspect.signature(Agent.think).parameters

    assert tuple(parameters) == ("self", "view", "turn")


def test_a_strategy_must_implement_think() -> None:
    """The abstract method is the whole contract, and it is enforced."""

    class Silent(Agent):
        """An agent that forgot to decide anything."""

    with pytest.raises(TypeError, match="think"):
        Silent(seed=0)


def test_baselines_finalize_exactly_once(picker: DeckPicker) -> None:
    """Every baseline closes its turn with one final legal submission."""
    state = build_play_state(
        picker,
        hands={
            PlayerId(0): picker.many([Rank.FIVE, Rank.SIX]),
            SECOND_SEAT: picker.take(Rank.KING),
        },
    )
    view = _actor_view(state)

    for agent in _every_agent():
        turn = FakeTurn()
        agent.think(view, turn)

        assert [submission.final for submission in turn.submissions] == [True]
        assert turn.selected in view.legal_moves


def test_baselines_choose_only_from_the_view(picker: DeckPicker) -> None:
    """Agents pick from ``view.legal_moves`` rather than deriving legality.

    The view is narrowed to a single option that the engine would not have been
    the only one to offer; an agent that recomputed legality from the cards it
    can see would ignore the narrowing.
    """
    state = build_play_state(
        picker,
        hands={
            PlayerId(0): [*picker.take(Rank.FIVE, 2), *picker.take(Rank.NINE, 2)],
            SECOND_SEAT: picker.take(Rank.KING),
        },
    )
    view = _actor_view(state)
    only = view.legal_moves[-1]
    narrowed = replace(view, legal_moves=(only,))
    assert len(view.legal_moves) > 1

    for agent in _every_agent():
        turn = FakeTurn()
        agent.think(narrowed, turn)

        assert turn.selected == only


def test_baselines_refuse_a_view_with_no_choices(picker: DeckPicker) -> None:
    """Observing as a non-actor yields no moves, and no agent invents one."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.FIVE), SECOND_SEAT: picker.take(Rank.KING)},
    )
    waiting = state.observe(SECOND_SEAT)
    assert waiting.legal_moves == ()

    with pytest.raises(ValueError, match="no legal moves"):
        legal_choices(waiting)
    for agent in _every_agent():
        with pytest.raises(ValueError, match="no legal moves"):
            agent.think(waiting, FakeTurn())


def test_agents_see_no_hidden_cards_in_a_crafted_position(picker: DeckPicker) -> None:
    """The observation an agent decides on carries nothing private.

    The face-down zone is the sharpest case: reveals name slot identifiers, so a
    baseline can act on it without any way to learn what is under a slot.
    """
    hidden = picker.one(Rank.ACE)
    state = build_play_state(
        picker,
        hands={PlayerId(0): [], SECOND_SEAT: picker.take(Rank.KING)},
        face_down={PlayerId(0): {SlotId(0): hidden}},
    )
    view = _actor_view(state)

    assert view.hand == ()
    assert view.me.face_down_slots == (SlotId(0),)
    assert all(card.id != hidden.id for card in view.discard_pile + view.burned_cards)
    for public in view.players:
        assert all(card.id != hidden.id for card in public.face_up)
    for agent in _every_agent():
        turn = FakeTurn()
        agent.think(view, turn)

        assert turn.selected in view.legal_moves


def test_agents_never_ask_the_engine_for_a_zone_they_cannot_see(picker: DeckPicker) -> None:
    """A play always names the active zone the engine chose, never the other one."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.FIVE, 2), SECOND_SEAT: picker.take(Rank.KING)},
        face_up={PlayerId(0): picker.take(Rank.ACE, 3)},
    )
    view = _actor_view(state)

    for agent in _every_agent():
        turn = FakeTurn()
        agent.think(view, turn)
        chosen = turn.selected

        assert isinstance(chosen, Play)
        assert chosen.source is Zone.HAND
