"""Tests for the agent adapter: what reaches the agent, and what never does.

The adapter is the one place where observed information is turned into a type the
engine defined, so this is where a fabricated identity would slip in if it were
going to. Most of these tests are therefore negative: no card in a built view stands
for a card nobody looked at, and the suits and identifiers that do appear carry no
claim at all.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from shed.companion.advice import (
    BOOKKEEPING_SUIT,
    GREEDY_CAVEAT,
    build_player_view,
    describe_move,
    recommend,
    view_gaps,
)
from shed.companion.observed import (
    ME,
    OPPONENT,
    ObservationError,
    PlayCards,
    apply_event,
    observed_legal_moves,
)
from shed.engine import AtLeast, Card, PickUp, Play, Rank, Reveal, Suit, Zone
from tests.companion.conftest import craft


def test_a_view_carries_my_hand_and_only_counts_for_theirs() -> None:
    """An opponent's hand reaches the agent as a number, never as cards."""
    state = craft(
        my_hand=(Rank.THREE, Rank.SEVEN),
        my_face_up=(Rank.FOUR,),
        my_face_down=2,
        opponent_hand_known=(Rank.KING,),
        opponent_hand_unknown=3,
        opponent_face_up=(Rank.FIVE,),
        opponent_face_down=2,
        deck_count=5,
    )
    view = build_player_view(state)
    assert [card.rank for card in view.hand] == [Rank.THREE, Rank.SEVEN]
    theirs = view.players[1]
    assert theirs.hand_count == 4
    assert [card.rank for card in theirs.face_up] == [Rank.FIVE]
    assert theirs.face_down_slots == (0, 1)
    assert Rank.KING not in {card.rank for card in theirs.face_up}


def test_a_view_omits_pile_cards_whose_rank_was_never_seen() -> None:
    """Padding the pile with plausible ranks is exactly what this refuses to do."""
    state = craft(
        my_hand=(Rank.THREE,),
        opponent_hand_unknown=1,
        pile=(None, None, Rank.ACE),
        constraint=AtLeast(Rank.ACE),
        deck_count=2,
    )
    view = build_player_view(state)
    assert [card.rank for card in view.discard_pile] == [Rank.ACE]
    assert state.pile_size == 3
    assert any("no recorded rank" in gap for gap in view_gaps(state))


def test_a_view_omits_burned_cards_whose_rank_was_never_seen() -> None:
    """The burned collection is reported the same way, and the gap is named."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1, deck_count=2)
    view = build_player_view(state)
    assert view.burned_cards == ()
    assert len(state.burned) > 0
    assert any("burned" in gap for gap in view_gaps(state))


def test_every_card_in_a_view_carries_the_bookkeeping_suit() -> None:
    """One suit throughout is the clearest signal that the suit means nothing."""
    state = craft(
        my_hand=(Rank.THREE, Rank.SEVEN),
        my_face_up=(Rank.FOUR,),
        opponent_face_up=(Rank.FIVE,),
        opponent_hand_unknown=1,
        deck_count=2,
    )
    view = build_player_view(state)
    cards: list[Card] = [*view.hand, *view.players[0].face_up, *view.players[1].face_up]
    assert BOOKKEEPING_SUIT is Suit.CLUBS
    assert {card.suit for card in cards} == {BOOKKEEPING_SUIT}


def test_a_joker_in_a_view_still_has_no_suit() -> None:
    """The bookkeeping suit does not override the type's own joker rule."""
    state = craft(my_hand=(Rank.JOKER,), opponent_hand_unknown=1, deck_count=2)
    assert build_player_view(state).hand[0].suit is None


def test_bookkeeping_identifiers_are_distinct_within_one_view() -> None:
    """The greedy baseline looks a rank up by identifier, so they must not collide."""
    state = craft(
        my_hand=(Rank.THREE, Rank.THREE, Rank.THREE),
        my_face_up=(Rank.THREE,),
        opponent_face_up=(Rank.FIVE, Rank.FIVE),
        opponent_hand_unknown=1,
        deck_count=2,
    )
    view = build_player_view(state)
    ids = [card.id for card in (*view.hand, *view.players[0].face_up, *view.players[1].face_up)]
    assert len(set(ids)) == len(ids)


def test_a_view_carries_no_history() -> None:
    """The companion keeps its own log; the engine's vocabulary cannot express it."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1, deck_count=2)
    assert build_player_view(state).history == ()


def test_a_view_cannot_be_built_while_my_own_ranks_are_outstanding() -> None:
    """Rather than invent the cards I drew, the adapter refuses."""
    state = craft(
        my_hand=(Rank.THREE, Rank.FOUR, Rank.FIVE),
        my_face_down=1,
        opponent_hand_unknown=1,
        deck_count=4,
    )
    after = apply_event(state, PlayCards(ME, Rank.THREE, 1))
    with pytest.raises(ObservationError, match="without inventing them"):
        build_player_view(after, ME)


def test_a_view_offers_moves_only_to_the_seat_that_is_to_act() -> None:
    """Legal moves belong to the actor, exactly as the engine's views do."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1, deck_count=2)
    assert build_player_view(state, ME).legal_moves
    theirs = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1, deck_count=2, to_act=OPPONENT)
    assert build_player_view(theirs, ME).legal_moves == ()


def test_no_view_can_be_built_for_a_hand_that_was_never_observed() -> None:
    """There is no honest view of a seat whose cards nobody at the table can see."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1, deck_count=2)
    with pytest.raises(ObservationError, match="without inventing them"):
        build_player_view(state, OPPONENT)


def test_a_recommendation_is_one_of_the_observed_legal_moves() -> None:
    """The suggestion is chosen from generated options, never composed."""
    state = craft(
        my_hand=(Rank.THREE, Rank.SEVEN, Rank.TEN),
        my_face_down=1,
        opponent_hand_unknown=2,
        deck_count=5,
    )
    assert recommend(state, seed=3).move in observed_legal_moves(state, ME)


def test_a_recommendation_never_changes_the_tracked_position() -> None:
    """Asking is not recording: the move is applied only when the operator says so."""
    state = craft(
        my_hand=(Rank.THREE, Rank.SEVEN),
        my_face_down=1,
        opponent_hand_unknown=2,
        deck_count=5,
    )
    before = replace(state)
    recommend(state, seed=7)
    assert state == before


def test_the_same_position_and_seed_recommend_the_same_move() -> None:
    """Seeding from the session's own content keeps a suggestion reproducible."""
    state = craft(
        my_hand=(Rank.THREE, Rank.SEVEN, Rank.TEN),
        my_face_down=1,
        opponent_hand_unknown=2,
        deck_count=5,
    )
    assert recommend(state, seed=11).move == recommend(state, seed=11).move


def test_greedy_sheds_the_biggest_batch_it_can() -> None:
    """The first of the agent's two keys, read back out of its own choice."""
    state = craft(
        my_hand=(Rank.FIVE, Rank.FIVE, Rank.ACE),
        my_face_down=1,
        opponent_hand_unknown=2,
        deck_count=5,
    )
    recommendation = recommend(state, seed=1)
    assert recommendation.move == Play(Zone.HAND, Rank.FIVE, 2)
    assert "biggest batch" in recommendation.reasoning


def test_greedy_spends_the_cheapest_rank_among_equal_batches() -> None:
    """The second key: a power card is kept while an ordinary one is available."""
    state = craft(
        my_hand=(Rank.FIVE, Rank.TEN),
        my_face_down=1,
        opponent_hand_unknown=2,
        deck_count=5,
    )
    recommendation = recommend(state, seed=1)
    assert recommendation.move == Play(Zone.HAND, Rank.FIVE, 1)
    assert "retention score" in recommendation.reasoning


def test_a_recommendation_states_a_burn_and_the_retained_turn() -> None:
    """The effect sentence is derived from the engine's own burn rule."""
    state = craft(
        my_hand=(Rank.TEN,),
        my_face_down=1,
        opponent_hand_unknown=2,
        pile=(Rank.KING,),
        constraint=AtLeast(Rank.KING),
        deck_count=5,
    )
    assert "burns the pile" in recommend(state, seed=1).effect


def test_a_forced_pickup_is_recommended_with_the_reason_it_is_forced() -> None:
    """When nothing can go down, the only legal action is explained as such."""
    state = craft(
        my_hand=(Rank.THREE,),
        my_face_down=1,
        opponent_hand_unknown=2,
        pile=(Rank.KING,),
        constraint=AtLeast(Rank.KING),
        deck_count=0,
    )
    recommendation = recommend(state, seed=1)
    assert recommendation.move == PickUp()
    assert "only" in recommendation.reasoning


def test_indistinguishable_face_down_cards_are_treated_as_one_decision() -> None:
    """Every remaining slot is the same action, and the reasoning says so."""
    state = craft(my_face_down=3, opponent_hand_unknown=2, deck_count=0)
    recommendation = recommend(state, seed=5)
    assert isinstance(recommendation.move, Reveal)
    assert "indistinguishable" in recommendation.reasoning
    assert recommendation.considered == 3


def test_every_recommendation_carries_the_standing_caveat() -> None:
    """No caller can present greedy advice without what it is."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=2, deck_count=5)
    recommendation = recommend(state, seed=1)
    assert recommendation.caveat == GREEDY_CAVEAT
    assert "not optimal play" in recommendation.caveat


def test_advice_is_refused_when_it_is_not_my_turn() -> None:
    """The companion advises one seat, the one holding the phone."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=2, deck_count=5, to_act=OPPONENT)
    with pytest.raises(ObservationError, match="your own turn"):
        recommend(state, seed=1)


def test_move_descriptions_name_the_zone_and_the_owner() -> None:
    """The interface's phrasing comes from one place, for both seats."""
    assert describe_move(Play(Zone.HAND, Rank.SEVEN, 2)) == "Play 2 x 7 from your hand"
    assert describe_move(Play(Zone.FACE_UP, Rank.ACE, 1), owner="their") == (
        "Play 1 x A from their face-up cards"
    )
    assert describe_move(PickUp()) == "Pick up the pile"


def test_a_nine_on_an_open_pile_is_described_as_transparent_not_as_a_joker() -> None:
    """Both leave no restriction; only one of them is a joker."""
    state = craft(
        my_hand=(Rank.NINE,),
        my_face_down=1,
        opponent_hand_unknown=2,
        deck_count=5,
    )
    effect = recommend(state, seed=1).effect
    assert "transparent" in effect
    assert "joker" not in effect


def test_a_joker_is_described_as_clearing_the_restriction() -> None:
    """The other side of the same branch."""
    state = craft(
        my_hand=(Rank.JOKER,),
        my_face_down=1,
        opponent_hand_unknown=2,
        pile=(Rank.KING,),
        constraint=AtLeast(Rank.KING),
        deck_count=5,
    )
    assert "joker clears" in recommend(state, seed=1).effect
