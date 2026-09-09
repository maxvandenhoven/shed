"""Tests for the uniform random baseline.

The agent has one job -- sample the engine's legal-move tuple uniformly and
finalize -- so these tests cover reproducibility, the shape of the sample, and
that it copes with every decision the engine can ask for.
"""

from collections import Counter

import pytest

from shed.agents import AgentSpec, RandomAgent, build_agent
from shed.engine import AtLeast, GameState, Move, PickUp, PlayerId, PlayerView, Rank, Reveal, SlotId
from tests.agents.conftest import FakeTurn
from tests.conftest import DeckPicker, build_play_state, build_setup_state

SECOND_SEAT: PlayerId = PlayerId(1)
"""The opponent in the crafted two-player positions here."""

SPEC = AgentSpec(kind="random", name="random-1")
"""The specification under test; the factory is the only construction path."""


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


def _decide(view: PlayerView, *, seed: int, remaining: float = 1.0) -> Move:
    """Run one decision of a freshly built random agent.

    Args:
        view: The observation to decide on.
        seed: Seed for the agent built for this decision.
        remaining: Fake seconds left in the budget.

    Returns:
        The move the agent finalized.
    """
    turn = FakeTurn(remaining=remaining)
    build_agent(SPEC, seed=seed).think(view, turn)
    return turn.selected


def _hand_view(picker: DeckPicker) -> PlayerView:
    """Build a view whose actor holds two fives and two nines.

    Args:
        picker: Source of the cards.

    Returns:
        The actor's view, offering four legal batches.
    """
    state = build_play_state(
        picker,
        hands={
            PlayerId(0): [*picker.take(Rank.FIVE, 2), *picker.take(Rank.NINE, 2)],
            SECOND_SEAT: picker.take(Rank.KING),
        },
    )
    return _actor_view(state)


def test_sampling_is_reproducible_from_the_seed(picker: DeckPicker) -> None:
    """The same seed on the same view gives the same move, every time."""
    view = _hand_view(picker)

    first = _decide(view, seed=99)
    second = _decide(view, seed=99)

    assert first == second


def test_different_seeds_explore_different_moves(picker: DeckPicker) -> None:
    """Fresh per-decision seeds are what stop every decision repeating itself."""
    view = _hand_view(picker)

    chosen = {_decide(view, seed=seed) for seed in range(30)}

    assert len(chosen) > 1


def test_sampling_is_uniform_over_the_legal_actions(picker: DeckPicker) -> None:
    """Every legal action is reachable, and none dominates the sample.

    The bounds are deliberately loose: this checks the agent samples the
    rank/count tuple it was given rather than favouring, say, its first entry.
    """
    view = _hand_view(picker)
    trials = 400
    assert len(view.legal_moves) == 4

    counts = Counter(_decide(view, seed=seed) for seed in range(trials))

    assert set(counts) == set(view.legal_moves)
    expected = trials / len(view.legal_moves)
    assert all(0.5 * expected <= count <= 1.5 * expected for count in counts.values())


def test_the_sample_ignores_the_clock(picker: DeckPicker) -> None:
    """The single submission is final, even with no time left to improve in."""
    view = _hand_view(picker)
    turn = FakeTurn(remaining=0.0)

    RandomAgent(seed=3).think(view, turn)

    assert [submission.final for submission in turn.submissions] == [True]
    assert turn.selected in view.legal_moves


def test_it_arranges_during_setup(picker: DeckPicker) -> None:
    """Setup offers 20 arrangements and the agent samples among them."""
    state = build_setup_state(
        picker,
        hands={
            PlayerId(0): picker.many([Rank.THREE, Rank.FOUR, Rank.FIVE]),
            SECOND_SEAT: picker.many([Rank.SIX, Rank.SEVEN, Rank.EIGHT]),
        },
        face_up={
            PlayerId(0): picker.many([Rank.TEN, Rank.JACK, Rank.QUEEN]),
            SECOND_SEAT: picker.many([Rank.KING, Rank.ACE, Rank.TWO]),
        },
    )
    view = _actor_view(state)
    assert len(view.legal_moves) == 20

    chosen = {_decide(view, seed=seed) for seed in range(60)}

    assert chosen <= set(view.legal_moves)
    assert len(chosen) > 1


def test_it_plays_from_the_face_up_collection(picker: DeckPicker) -> None:
    """With hand and deck empty the sample space is the face-up batches."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): [], SECOND_SEAT: picker.take(Rank.KING)},
        face_up={PlayerId(0): picker.take(Rank.QUEEN, 2)},
    )
    view = _actor_view(state)

    chosen = {_decide(view, seed=seed) for seed in range(20)}

    assert chosen == set(view.legal_moves)


def test_it_reveals_a_blind_slot(picker: DeckPicker) -> None:
    """Only face-down slots remain, and every slot is a possible sample."""
    blind = {SlotId(slot): card for slot, card in enumerate(picker.take(Rank.ACE, 3))}
    state = build_play_state(
        picker,
        hands={PlayerId(0): [], SECOND_SEAT: picker.take(Rank.KING)},
        face_down={PlayerId(0): blind},
    )
    view = _actor_view(state)

    chosen = {_decide(view, seed=seed) for seed in range(60)}

    assert chosen == {Reveal(SlotId(slot)) for slot in blind}


def test_it_picks_up_when_blocked(picker: DeckPicker) -> None:
    """A forced pickup is the only sample there is."""
    pile = picker.take(Rank.KING)
    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.THREE), SECOND_SEAT: picker.take(Rank.QUEEN)},
        discard=pile,
        constraint=AtLeast(Rank.KING),
    )
    view = _actor_view(state)
    assert view.legal_moves == (PickUp(),)

    assert _decide(view, seed=1) == PickUp()


@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_construction_never_replays_one_stream(picker: DeckPicker, seed: int) -> None:
    """Two agents built with the same seed agree; the seed, not the class, decides.

    This is why the runner must draw a fresh seed per decision: rebuilding an
    agent from its specification alone would replay the same stream forever.
    """
    view = _hand_view(picker)

    repeated = [RandomAgent(seed=seed) for _ in range(3)]
    turns = [FakeTurn() for _ in repeated]
    for agent, turn in zip(repeated, turns, strict=True):
        agent.think(view, turn)

    assert len({turn.selected for turn in turns}) == 1
