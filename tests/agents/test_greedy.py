"""Tests for the greedy baseline's scoring, preferences, and tie-breaking.

The heuristic is fully specified, so these tests pin the exact numbers: the
retention table, the largest-batch-first preference, the cheapest-cards-first
tie-break under it, and the seeded choice among genuinely equal options.
"""

import pytest

from shed.agents import RETENTION_SCORE, AgentSpec, GreedyAgent, build_agent
from shed.engine import (
    Arrange,
    AtLeast,
    Card,
    GameState,
    Move,
    PickUp,
    Play,
    PlayerId,
    PlayerView,
    Rank,
    Reveal,
    SlotId,
    Zone,
)
from tests.agents.conftest import FakeTurn
from tests.conftest import DeckPicker, build_play_state, build_setup_state

SECOND_SEAT: PlayerId = PlayerId(1)
"""The opponent in the crafted two-player positions here."""

SPEC = AgentSpec(kind="greedy", name="greedy-1")
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


def _decide(view: PlayerView, *, seed: int = 0, remaining: float = 1.0) -> Move:
    """Run one decision of a freshly built greedy agent.

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


def _hand_view(picker: DeckPicker, hand: list[Card]) -> PlayerView:
    """Build a view whose actor holds exactly the given hand.

    Args:
        picker: Source of the opponent's filler card.
        hand: The actor's hand.

    Returns:
        The actor's view on an unrestricted pile.
    """
    state = build_play_state(
        picker,
        hands={PlayerId(0): hand, SECOND_SEAT: picker.take(Rank.KING)},
    )
    return _actor_view(state)


def test_the_retention_table_is_the_specified_one() -> None:
    """Ordinary ranks score their value; the powers score above all of them."""
    assert RETENTION_SCORE == {
        Rank.THREE: 3,
        Rank.FOUR: 4,
        Rank.FIVE: 5,
        Rank.SIX: 6,
        Rank.SEVEN: 17,
        Rank.EIGHT: 8,
        Rank.NINE: 20,
        Rank.TEN: 23,
        Rank.JACK: 11,
        Rank.QUEEN: 12,
        Rank.KING: 13,
        Rank.ACE: 14,
        Rank.TWO: 21,
        Rank.JOKER: 22,
    }


def test_setup_keeps_the_highest_scoring_face_up_set(picker: DeckPicker) -> None:
    """The three most valuable cards go face up, wherever they were dealt."""
    keep = picker.many([Rank.TEN, Rank.TWO, Rank.NINE])
    shed = picker.many([Rank.THREE, Rank.FOUR, Rank.FIVE])
    state = build_setup_state(
        picker,
        hands={PlayerId(0): keep, SECOND_SEAT: picker.many([Rank.SIX, Rank.SEVEN, Rank.EIGHT])},
        face_up={PlayerId(0): shed, SECOND_SEAT: picker.many([Rank.KING, Rank.ACE, Rank.JACK])},
        # Arrangements are requested clockwise after the dealer, so dealing from
        # seat one puts the seat under test first.
        dealer=SECOND_SEAT,
    )
    view = _actor_view(state)

    chosen = _decide(view)

    first, second, third = sorted(card.id for card in keep)
    assert chosen == Arrange((first, second, third))


def test_setup_breaks_equal_sums_with_the_seed(picker: DeckPicker) -> None:
    """Equally valuable sets are settled by the generator, not by move order.

    Four twos give four best sets of identical value, which is the only way a
    setup decision can tie: the table scores each rank differently.
    """
    twos = picker.take(Rank.TWO, 4)
    state = build_setup_state(
        picker,
        hands={PlayerId(0): twos[:3], SECOND_SEAT: picker.many([Rank.SIX, Rank.SEVEN, Rank.EIGHT])},
        face_up={
            PlayerId(0): [twos[3], *picker.many([Rank.THREE, Rank.FOUR])],
            SECOND_SEAT: picker.many([Rank.KING, Rank.ACE, Rank.JACK]),
        },
        dealer=SECOND_SEAT,
    )
    view = _actor_view(state)
    two_ids = {card.id for card in twos}

    chosen = {_decide(view, seed=seed) for seed in range(40)}

    assert len(chosen) == 4, "every equally valuable set should be reachable"
    for move in chosen:
        assert isinstance(move, Arrange)
        assert set(move.face_up_cards) < two_ids


def test_it_sheds_the_largest_batch(picker: DeckPicker) -> None:
    """Batch size comes first: more cards gone beats cheaper cards gone."""
    view = _hand_view(picker, [*picker.take(Rank.FIVE, 2), *picker.take(Rank.SIX)])

    assert _decide(view) == Play(Zone.HAND, Rank.FIVE, 2)


def test_batch_size_outranks_the_retention_score(picker: DeckPicker) -> None:
    """A bigger batch of expensive cards beats a single cheap one."""
    view = _hand_view(picker, [*picker.take(Rank.SEVEN, 2), *picker.take(Rank.THREE)])

    assert _decide(view) == Play(Zone.HAND, Rank.SEVEN, 2)


def test_equal_batches_spend_the_cheaper_rank(picker: DeckPicker) -> None:
    """Among equally sized plays the low-retention cards go first."""
    view = _hand_view(picker, [*picker.take(Rank.EIGHT, 2), *picker.take(Rank.NINE, 2)])

    assert _decide(view) == Play(Zone.HAND, Rank.EIGHT, 2)


def test_scoring_uses_the_table_rather_than_rank_order(picker: DeckPicker) -> None:
    """A joker is cheaper to spend than a ten, though its enum value is higher."""
    view = _hand_view(picker, [*picker.take(Rank.TEN), *picker.take(Rank.JOKER)])

    assert _decide(view) == Play(Zone.HAND, Rank.JOKER, 1)


def test_a_unique_best_play_ignores_the_seed(picker: DeckPicker) -> None:
    """Without a tie the heuristic is deterministic, whatever the seed."""
    view = _hand_view(picker, [*picker.take(Rank.FIVE, 2), *picker.take(Rank.NINE, 2)])

    chosen = {_decide(view, seed=seed) for seed in range(25)}

    assert chosen == {Play(Zone.HAND, Rank.FIVE, 2)}


def test_it_respects_the_constraint_the_engine_enforced(picker: DeckPicker) -> None:
    """Only offered batches are considered; the agent recomputes no legality."""
    pile = picker.take(Rank.KING)
    state = build_play_state(
        picker,
        hands={
            PlayerId(0): [*picker.take(Rank.THREE, 2), *picker.take(Rank.ACE)],
            SECOND_SEAT: picker.take(Rank.QUEEN),
        },
        discard=pile,
        constraint=AtLeast(Rank.KING),
    )
    view = _actor_view(state)

    assert _decide(view) == Play(Zone.HAND, Rank.ACE, 1)


def test_it_plays_from_the_face_up_collection(picker: DeckPicker) -> None:
    """The same preferences apply once play moves to the table."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): [], SECOND_SEAT: picker.take(Rank.KING)},
        face_up={PlayerId(0): [*picker.take(Rank.QUEEN, 2), *picker.take(Rank.TWO)]},
    )
    view = _actor_view(state)

    assert _decide(view) == Play(Zone.FACE_UP, Rank.QUEEN, 2)


def test_blind_reveals_are_a_seeded_choice_of_slot(picker: DeckPicker) -> None:
    """Nothing distinguishes one hidden slot from another, so the seed decides."""
    blind = {SlotId(slot): card for slot, card in enumerate(picker.take(Rank.ACE, 3))}
    state = build_play_state(
        picker,
        hands={PlayerId(0): [], SECOND_SEAT: picker.take(Rank.KING)},
        face_down={PlayerId(0): blind},
    )
    view = _actor_view(state)

    chosen = {_decide(view, seed=seed) for seed in range(60)}

    assert chosen == {Reveal(SlotId(slot)) for slot in blind}


def test_it_picks_up_when_that_is_the_only_action(picker: DeckPicker) -> None:
    """A forced pickup needs no scoring; it is simply taken."""
    pile = picker.take(Rank.KING)
    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.THREE), SECOND_SEAT: picker.take(Rank.QUEEN)},
        discard=pile,
        constraint=AtLeast(Rank.KING),
    )
    view = _actor_view(state)
    assert view.legal_moves == (PickUp(),)

    assert _decide(view) == PickUp()


@pytest.mark.parametrize("remaining", [0.0, 2.0])
def test_the_decision_ignores_the_clock(picker: DeckPicker, remaining: float) -> None:
    """One pass, one final submission: the heuristic has nothing to improve."""
    view = _hand_view(picker, [*picker.take(Rank.FIVE, 2), *picker.take(Rank.SIX)])
    turn = FakeTurn(remaining=remaining)

    GreedyAgent(seed=0).think(view, turn)

    assert [submission.final for submission in turn.submissions] == [True]
    assert turn.selected == Play(Zone.HAND, Rank.FIVE, 2)
