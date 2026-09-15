"""Tests for the observed-state model: what it knows, what it refuses to invent.

Each test isolates one rule or one kind of uncertainty. The risks the companion
introduces over the engine are all about *missing* information, so the assertions
are as much about what a position does not claim -- an unknown card that stays
unknown, a rank the companion will not guess -- as about what it resolves.
"""

from __future__ import annotations

import pytest

from shed.companion.observed import (
    ME,
    OPPONENT,
    CorrectState,
    ObservationError,
    ObservedState,
    PendingReason,
    PickUpPile,
    PlayCards,
    RecordCards,
    RevealFaceDown,
    SeatObservation,
    StatePatch,
    advice_blockers,
    apply_event,
    join_game,
    new_game,
    observed_legal_moves,
    seat_active_zone,
    validate_observed,
)
from shed.engine import AtLeast, AtMost, Phase, Play, Rank, Unrestricted, Zone
from tests.companion.conftest import craft


def test_a_new_game_leaves_every_unseen_card_unknown(opening: ObservedState) -> None:
    """The opening records ranks only where somebody looked."""
    assert opening.seat(ME).hand_known == (Rank.THREE, Rank.SEVEN, Rank.TEN)
    assert opening.seat(ME).hand_unknown == 0
    assert opening.seat(OPPONENT).hand_known == ()
    assert opening.seat(OPPONENT).hand_unknown == 3
    assert opening.seat(ME).face_down == 3
    assert opening.seat(OPPONENT).face_down == 3
    assert opening.deck_count == 36
    assert opening.pile == ()


def test_a_new_game_refuses_a_deal_that_is_not_the_profile_s() -> None:
    """A miscounted opening is rejected where it is entered, not later."""
    with pytest.raises(ObservationError, match="3 hand cards"):
        new_game(
            my_hand=(Rank.THREE, Rank.FOUR),
            my_face_up=(Rank.FIVE, Rank.SIX, Rank.SEVEN),
            opponent_face_up=(Rank.EIGHT, Rank.NINE, Rank.TEN),
            starting_player=ME,
        )


def test_a_rank_cannot_be_recorded_more_often_than_the_deck_holds() -> None:
    """Five fours is not a table, however consistent the counts look."""
    with pytest.raises(ObservationError, match="deck only holds 4"):
        craft(my_hand=(Rank.FOUR,) * 4, my_face_up=(Rank.FOUR,), deck_count=0)


def test_cards_that_do_not_add_up_to_the_deck_are_refused() -> None:
    """Card accounting is checked whenever the deck has been counted."""
    state = craft(my_hand=(Rank.FOUR,), my_face_down=1, opponent_hand_unknown=1, deck_count=5)
    with pytest.raises(ObservationError, match="add up to"):
        validate_observed(
            ObservedState(
                rules=state.rules,
                seats=state.seats,
                deck_count=5,
                pile=(),
                burned=(),
                constraint=Unrestricted(),
                to_act=ME,
                phase=Phase.PLAY,
                winner=None,
                pending=(),
            )
        )


def test_an_empty_pile_cannot_carry_a_restriction() -> None:
    """The engine's own invariant holds for an observed position too."""
    with pytest.raises(ObservationError, match="empty pile"):
        craft(my_hand=(Rank.FOUR,), deck_count=0, constraint=AtLeast(Rank.FIVE))


def test_a_legal_batch_resolves_and_sets_the_constraint() -> None:
    """An ordinary rank leaves ``AtLeast`` behind and passes the turn."""
    state = craft(
        my_hand=(Rank.EIGHT, Rank.EIGHT, Rank.FOUR),
        opponent_hand_unknown=1,
        pile=(Rank.THREE,),
        constraint=AtLeast(Rank.THREE),
    )
    after = apply_event(state, PlayCards(ME, Rank.EIGHT, 2))
    assert after.pile == (Rank.THREE, Rank.EIGHT, Rank.EIGHT)
    assert after.constraint == AtLeast(Rank.EIGHT)
    assert after.to_act == OPPONENT
    assert after.seat(ME).hand_known == (Rank.FOUR,)


def test_an_illegal_rank_is_refused_and_changes_nothing() -> None:
    """A rejected batch leaves the caller holding exactly what it passed in."""
    state = craft(
        my_hand=(Rank.FOUR,),
        opponent_hand_unknown=1,
        pile=(Rank.KING,),
        constraint=AtLeast(Rank.KING),
    )
    with pytest.raises(ObservationError, match="cannot be played"):
        apply_event(state, PlayCards(ME, Rank.FOUR, 1))
    assert state.seat(ME).hand_known == (Rank.FOUR,)
    assert state.pile == (Rank.KING,)


def test_a_batch_larger_than_the_hand_is_refused() -> None:
    """Counts are checked against what the zone actually holds."""
    state = craft(my_hand=(Rank.FOUR, Rank.FOUR), opponent_hand_unknown=1)
    with pytest.raises(ObservationError, match="cannot play 3"):
        apply_event(state, PlayCards(ME, Rank.FOUR, 3))


def test_a_face_up_batch_larger_than_the_set_is_refused() -> None:
    """A face-up set is public, so overclaiming it is a flat contradiction."""
    state = craft(my_face_up=(Rank.FOUR, Rank.NINE), opponent_hand_unknown=1)
    with pytest.raises(ObservationError, match="face up, not 2"):
        apply_event(state, PlayCards(ME, Rank.FOUR, 2))


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (Rank.TWO, AtLeast(Rank.TWO)),
        (Rank.SEVEN, AtMost(Rank.SEVEN)),
        (Rank.JOKER, Unrestricted()),
        (Rank.JACK, AtLeast(Rank.JACK)),
    ],
)
def test_special_ranks_leave_the_profile_s_restriction(rank: Rank, expected: object) -> None:
    """Twos, sevens, and jokers set what the engine says they set."""
    state = craft(
        my_hand=(rank,),
        my_face_down=1,
        opponent_hand_unknown=1,
        pile=(Rank.FOUR,),
        constraint=AtLeast(Rank.FOUR),
    )
    assert apply_event(state, PlayCards(ME, rank, 1)).constraint == expected


def test_a_nine_preserves_a_seven_restriction() -> None:
    """A transparent nine leaves ``AtMost(7)`` in force rather than replacing it."""
    state = craft(
        my_hand=(Rank.NINE,),
        my_face_down=1,
        opponent_hand_unknown=1,
        pile=(Rank.SEVEN,),
        constraint=AtMost(Rank.SEVEN),
    )
    assert apply_event(state, PlayCards(ME, Rank.NINE, 1)).constraint == AtMost(Rank.SEVEN)


def test_a_ten_burns_the_pile_and_keeps_the_turn() -> None:
    """The whole pile leaves the game and the same seat decides again."""
    state = craft(
        my_hand=(Rank.TEN, Rank.FIVE),
        my_face_down=1,
        opponent_hand_unknown=1,
        pile=(Rank.KING, Rank.ACE),
        constraint=AtLeast(Rank.ACE),
    )
    after = apply_event(state, PlayCards(ME, Rank.TEN, 1))
    assert after.pile == ()
    assert after.constraint == Unrestricted()
    assert after.to_act == ME
    assert len(after.burned) == len(state.burned) + 3


def test_four_in_one_action_burns_but_four_across_actions_does_not() -> None:
    """The four-card burn is about one action, exactly as the profile says."""
    batch = craft(my_hand=(Rank.FIVE,) * 4, my_face_down=1, opponent_hand_unknown=1)
    assert apply_event(batch, PlayCards(ME, Rank.FIVE, 4)).pile == ()

    accumulated = craft(
        my_hand=(Rank.FIVE,),
        my_face_down=1,
        opponent_hand_unknown=1,
        pile=(Rank.FIVE, Rank.FIVE, Rank.FIVE),
        constraint=AtLeast(Rank.FIVE),
    )
    after = apply_event(accumulated, PlayCards(ME, Rank.FIVE, 1))
    assert after.pile == (Rank.FIVE,) * 4
    assert after.to_act == OPPONENT


def test_a_pickup_takes_the_known_pile_into_the_taker_s_known_hand() -> None:
    """Everybody watched those cards go down, so they are known in that hand."""
    state = craft(
        my_hand=(Rank.THREE,),
        opponent_hand_unknown=2,
        opponent_face_down=1,
        pile=(Rank.KING, Rank.ACE),
        constraint=AtLeast(Rank.ACE),
        to_act=OPPONENT,
    )
    after = apply_event(state, PickUpPile(OPPONENT))
    assert after.seat(OPPONENT).hand_known == (Rank.KING, Rank.ACE)
    assert after.seat(OPPONENT).hand_unknown == 2
    assert after.constraint == Unrestricted()
    assert after.pile == ()
    assert after.to_act == ME


def test_a_pickup_of_unseen_cards_does_not_name_them() -> None:
    """Joining mid-game leaves pile cards unknown, and a pickup keeps them so."""
    state = craft(
        opponent_hand_unknown=1,
        my_hand=(Rank.THREE,),
        pile=(None, None, Rank.ACE),
        constraint=AtLeast(Rank.ACE),
        to_act=OPPONENT,
    )
    after = apply_event(state, PickUpPile(OPPONENT))
    assert after.seat(OPPONENT).hand_known == (Rank.ACE,)
    assert after.seat(OPPONENT).hand_unknown == 3


def test_a_known_opponent_card_is_spent_before_an_unknown_one() -> None:
    """Conservative tracking: what was proven stays proven until it is played."""
    state = craft(
        opponent_hand_known=(Rank.KING, Rank.KING),
        opponent_hand_unknown=1,
        my_hand=(Rank.THREE,),
        to_act=OPPONENT,
    )
    after = apply_event(state, PlayCards(OPPONENT, Rank.KING, 1))
    assert after.seat(OPPONENT).hand_known == (Rank.KING,)
    assert after.seat(OPPONENT).hand_unknown == 1


def test_an_unproven_rank_comes_out_of_the_unknown_count() -> None:
    """Playing a rank nobody saw never invents a known copy of it first."""
    state = craft(
        opponent_hand_known=(Rank.KING,),
        opponent_hand_unknown=2,
        my_hand=(Rank.THREE,),
        to_act=OPPONENT,
    )
    after = apply_event(state, PlayCards(OPPONENT, Rank.FOUR, 1))
    assert after.seat(OPPONENT).hand_known == (Rank.KING,)
    assert after.seat(OPPONENT).hand_unknown == 1


def test_a_pickup_is_refused_when_the_observed_ranks_could_have_been_played() -> None:
    """``standard`` has no voluntary pickup, and this is checkable for my own hand."""
    state = craft(
        my_hand=(Rank.ACE,),
        opponent_hand_unknown=1,
        pile=(Rank.FOUR,),
        constraint=AtLeast(Rank.FOUR),
    )
    with pytest.raises(ObservationError, match="no voluntary pickup"):
        apply_event(state, PickUpPile(ME))


def test_a_pickup_stands_when_the_hand_is_not_fully_observed() -> None:
    """An opponent's unknown hand is never assumed to have held a playable card."""
    state = craft(
        my_hand=(Rank.THREE,),
        opponent_hand_unknown=2,
        pile=(Rank.ACE,),
        constraint=AtLeast(Rank.ACE),
        to_act=OPPONENT,
    )
    assert apply_event(state, PickUpPile(OPPONENT)).to_act == ME


def test_a_refill_queues_my_drawn_ranks_and_takes_them_off_the_deck() -> None:
    """The cards move immediately; only their identities are outstanding."""
    state = craft(
        my_hand=(Rank.THREE, Rank.FOUR, Rank.FIVE),
        my_face_down=1,
        opponent_hand_unknown=1,
        deck_count=4,
    )
    after = apply_event(state, PlayCards(ME, Rank.THREE, 1))
    assert after.deck_count == 3
    assert after.seat(ME).hand_count == 3
    assert after.seat(ME).hand_unknown == 1
    assert [entry.reason for entry in after.pending] == [PendingReason.DRAW]
    recorded = apply_event(after, RecordCards((Rank.NINE,)))
    assert recorded.seat(ME).hand_known == (Rank.FOUR, Rank.FIVE, Rank.NINE)
    assert recorded.pending == ()


def test_a_nearly_empty_deck_refills_only_what_is_left() -> None:
    """One card left draws one card, not the profile's target."""
    state = craft(
        my_hand=(Rank.THREE, Rank.FOUR),
        my_face_down=1,
        opponent_hand_unknown=1,
        deck_count=1,
    )
    after = apply_event(state, PlayCards(ME, Rank.THREE, 1))
    assert after.deck_count == 0
    assert after.pending[0].count == 1
    assert after.seat(ME).hand_count == 2


def test_an_empty_deck_refills_nothing_and_queues_nothing() -> None:
    """With the deck gone there is nothing to draw and nothing to type in."""
    state = craft(
        my_hand=(Rank.THREE, Rank.FOUR),
        my_face_down=1,
        opponent_hand_unknown=1,
        deck_count=0,
    )
    after = apply_event(state, PlayCards(ME, Rank.THREE, 1))
    assert after.pending == ()
    assert after.seat(ME).hand_known == (Rank.FOUR,)


def test_nothing_else_happens_until_drawn_ranks_are_recorded() -> None:
    """An outstanding entry blocks the next observation rather than being guessed."""
    state = craft(
        my_hand=(Rank.THREE, Rank.FOUR, Rank.FIVE),
        my_face_down=1,
        opponent_hand_unknown=1,
        deck_count=4,
    )
    after = apply_event(state, PlayCards(ME, Rank.THREE, 1))
    with pytest.raises(ObservationError, match="Record the 1 card"):
        apply_event(after, PlayCards(OPPONENT, Rank.KING, 1))


def test_recording_the_wrong_number_of_ranks_is_refused() -> None:
    """The count came from a physical transfer, so it is not up for revision."""
    state = craft(
        my_hand=(Rank.THREE, Rank.FOUR, Rank.FIVE),
        my_face_down=1,
        opponent_hand_unknown=1,
        deck_count=4,
    )
    after = apply_event(state, PlayCards(ME, Rank.THREE, 1))
    with pytest.raises(ObservationError, match="exactly 1 rank"):
        apply_event(after, RecordCards((Rank.NINE, Rank.TEN)))


def test_the_active_zone_walks_hand_then_face_up_then_face_down() -> None:
    """Each zone opens only once the ones before it are gone, and the deck with them."""
    hand = craft(
        my_hand=(Rank.FOUR,), my_face_up=(Rank.SIX,), my_face_down=1, opponent_hand_unknown=1
    )
    assert seat_active_zone(hand, ME) is Zone.HAND
    after_hand = apply_event(hand, PlayCards(ME, Rank.FOUR, 1))
    assert seat_active_zone(after_hand, ME) is Zone.FACE_UP
    face_up = craft(my_face_up=(Rank.SIX,), my_face_down=1, opponent_hand_unknown=1)
    after_face_up = apply_event(face_up, PlayCards(ME, Rank.SIX, 1))
    assert seat_active_zone(after_face_up, ME) is Zone.FACE_DOWN


def test_a_hand_play_is_refused_in_the_face_down_phase() -> None:
    """The only action left is a reveal, and the message says so."""
    state = craft(my_face_down=2, opponent_hand_unknown=1)
    with pytest.raises(ObservationError, match="record the reveal"):
        apply_event(state, PlayCards(ME, Rank.FOUR, 1))


def test_a_successful_reveal_resolves_as_a_single_card_play() -> None:
    """The turned card goes down and the restriction follows from its rank."""
    state = craft(
        my_face_down=2,
        opponent_hand_unknown=1,
        pile=(Rank.FOUR,),
        constraint=AtLeast(Rank.FOUR),
    )
    after = apply_event(state, RevealFaceDown(ME, Rank.NINE))
    assert after.pile == (Rank.FOUR, Rank.NINE)
    assert after.constraint == AtLeast(Rank.FOUR)
    assert after.seat(ME).face_down == 1
    assert after.to_act == OPPONENT


def test_a_failed_reveal_takes_the_pile_and_the_card_into_hand() -> None:
    """Legality is judged against the restriction in force before the turn."""
    state = craft(
        my_face_down=2,
        opponent_hand_unknown=1,
        pile=(Rank.KING,),
        constraint=AtLeast(Rank.KING),
    )
    after = apply_event(state, RevealFaceDown(ME, Rank.THREE))
    assert after.seat(ME).hand_known == (Rank.THREE, Rank.KING)
    assert after.pile == ()
    assert after.constraint == Unrestricted()
    assert after.seat(ME).face_down == 1
    assert after.to_act == OPPONENT


def test_a_reveal_that_burns_keeps_the_turn() -> None:
    """A ten off a face-down card burns and decides again, like any other ten."""
    state = craft(
        my_face_down=2,
        opponent_hand_unknown=1,
        pile=(Rank.KING,),
        constraint=AtLeast(Rank.KING),
    )
    after = apply_event(state, RevealFaceDown(ME, Rank.TEN))
    assert after.pile == ()
    assert after.to_act == ME


def test_a_reveal_before_the_face_down_phase_is_refused() -> None:
    """A face-down card comes into play only once hand and table are gone."""
    state = craft(my_hand=(Rank.FOUR,), my_face_down=1, opponent_hand_unknown=1)
    with pytest.raises(ObservationError, match="only comes into play"):
        apply_event(state, RevealFaceDown(ME, Rank.NINE))


def test_the_last_card_wins_after_the_refill_is_resolved() -> None:
    """The win is counted after replenishment, as the engine counts it."""
    state = craft(my_hand=(Rank.FOUR,), opponent_hand_unknown=2, deck_count=0)
    after = apply_event(state, PlayCards(ME, Rank.FOUR, 1))
    assert after.phase is Phase.FINISHED
    assert after.winner == ME
    assert after.to_act is None


def test_shedding_a_last_hand_card_with_a_live_deck_wins_nobody_the_game() -> None:
    """The refill happens first, so the hand is not empty when the win is counted."""
    state = craft(my_hand=(Rank.FOUR,), opponent_hand_unknown=2, deck_count=3)
    after = apply_event(state, PlayCards(ME, Rank.FOUR, 1))
    assert after.phase is Phase.PLAY
    assert after.seat(ME).hand_count == 3


def test_a_finished_game_refuses_further_observations() -> None:
    """The way back from a finished game is Undo or a correction, not another play."""
    state = craft(my_hand=(Rank.FOUR,), opponent_hand_unknown=2, deck_count=0)
    after = apply_event(state, PlayCards(ME, Rank.FOUR, 1))
    with pytest.raises(ObservationError, match="recorded as finished"):
        apply_event(after, PlayCards(OPPONENT, Rank.KING, 1))


def test_joining_without_a_deck_count_blocks_advice_instead_of_guessing() -> None:
    """An uncounted deck is a named missing observation, not a zero."""
    state = join_game(
        my_hand=(Rank.THREE, Rank.FOUR),
        my_face_up=(Rank.FIVE,),
        my_face_down=2,
        opponent_hand_count=4,
        opponent_hand_known=(),
        opponent_face_up=(Rank.SIX,),
        opponent_face_down=2,
        deck_count=None,
        pile=(None, Rank.ACE),
        constraint=AtLeast(Rank.ACE),
        to_act=ME,
    )
    assert state.deck_count is None
    assert [blocker.code for blocker in advice_blockers(state)] == ["deck_unknown"]
    with pytest.raises(ObservationError, match="deck count is unknown"):
        seat_active_zone(state, ME)


def test_joining_derives_the_burned_cards_without_naming_them() -> None:
    """Burned cards are what the 54 cannot otherwise account for, and stay unknown."""
    state = join_game(
        my_hand=(Rank.THREE, Rank.FOUR, Rank.FIVE),
        my_face_up=(Rank.SIX,),
        my_face_down=2,
        opponent_hand_count=3,
        opponent_hand_known=(Rank.KING,),
        opponent_face_up=(Rank.SEVEN,),
        opponent_face_down=2,
        deck_count=10,
        pile=(Rank.EIGHT, Rank.NINE),
        constraint=AtLeast(Rank.NINE),
        to_act=ME,
    )
    assert len(state.burned) == 54 - (10 + 2 + 6 + 6)
    assert set(state.burned) == {None}
    assert state.seat(OPPONENT).hand_known == (Rank.KING,)
    assert state.seat(OPPONENT).hand_unknown == 2


def test_joining_refuses_more_known_opponent_cards_than_they_hold() -> None:
    """A hand cannot be smaller than what was proven to be in it."""
    with pytest.raises(ObservationError, match="they hold 1"):
        join_game(
            my_hand=(Rank.THREE,),
            my_face_up=(),
            my_face_down=0,
            opponent_hand_count=1,
            opponent_hand_known=(Rank.KING, Rank.ACE),
            opponent_face_up=(),
            opponent_face_down=0,
            deck_count=0,
            pile=(),
            constraint=Unrestricted(),
            to_act=ME,
        )


def test_a_correction_sets_only_the_fields_it_names() -> None:
    """Everything else is left exactly as tracked."""
    state = craft(
        my_hand=(Rank.THREE,),
        my_face_down=1,
        opponent_hand_unknown=2,
        opponent_face_down=1,
        deck_count=3,
    )
    after = apply_event(state, CorrectState(StatePatch(deck_count=2, burned_count=47)))
    assert after.deck_count == 2
    assert len(after.burned) == 47
    assert after.seat(ME).hand_known == (Rank.THREE,)
    assert after.to_act == ME


def test_a_correction_to_my_hand_clears_what_was_waiting_to_be_recorded() -> None:
    """Giving the ranks outright leaves nothing outstanding to identify."""
    state = craft(
        my_hand=(Rank.THREE, Rank.FOUR, Rank.FIVE),
        my_face_down=1,
        opponent_hand_unknown=1,
        deck_count=4,
    )
    after = apply_event(state, PlayCards(ME, Rank.THREE, 1))
    assert after.pending
    fixed = apply_event(after, CorrectState(StatePatch(my_hand=(Rank.FOUR, Rank.FIVE, Rank.NINE))))
    assert fixed.pending == ()
    assert fixed.seat(ME).hand_unknown == 0


def test_an_impossible_correction_is_refused_and_the_position_survives() -> None:
    """A correction is validated like any other observation."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1, deck_count=2)
    with pytest.raises(ObservationError, match="add up to"):
        apply_event(state, CorrectState(StatePatch(deck_count=40)))
    assert state.deck_count == 2


def test_an_empty_correction_is_refused() -> None:
    """A recorded entry that changes nothing would only clutter the history."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1)
    with pytest.raises(ObservationError, match="change something"):
        apply_event(state, CorrectState(StatePatch()))


def test_a_correction_can_reopen_a_game_recorded_as_finished() -> None:
    """Setting whose turn it is puts a mistakenly finished game back into play."""
    state = craft(my_hand=(Rank.FOUR,), opponent_hand_unknown=2, deck_count=0)
    finished = apply_event(state, PlayCards(ME, Rank.FOUR, 1))
    reopened = apply_event(
        finished, CorrectState(StatePatch(my_hand=(Rank.FOUR,), burned_count=50, to_act=ME))
    )
    assert reopened.phase is Phase.PLAY
    assert reopened.winner is None
    assert reopened.to_act == ME


def test_observed_moves_never_claim_an_opponent_holds_an_unseen_rank() -> None:
    """Their generated options cover what was proven, never the rest of the hand."""
    state = craft(
        opponent_hand_known=(Rank.KING,),
        opponent_hand_unknown=3,
        my_hand=(Rank.THREE,),
        to_act=OPPONENT,
    )
    moves = observed_legal_moves(state, OPPONENT)
    assert {move.rank for move in moves if isinstance(move, Play)} == {Rank.KING}
    assert all(isinstance(move, Play) for move in moves)


def test_blockers_name_every_missing_observation_in_reading_order() -> None:
    """The panel that replaces a recommendation says what to record next."""
    state = craft(
        my_hand=(Rank.THREE, Rank.FOUR, Rank.FIVE),
        my_face_down=1,
        opponent_hand_unknown=1,
        deck_count=4,
    )
    after = apply_event(state, PlayCards(ME, Rank.THREE, 1))
    assert [blocker.code for blocker in advice_blockers(after)] == [
        "pending_draw",
        "not_my_turn",
    ]


def test_a_seat_observation_normalizes_its_ranks() -> None:
    """Two observations of the same cards compare equal whatever order they arrived."""
    first = SeatObservation(ME, (Rank.KING, Rank.THREE), 0, (Rank.ACE, Rank.FOUR), 1)
    second = SeatObservation(ME, (Rank.THREE, Rank.KING), 0, (Rank.FOUR, Rank.ACE), 1)
    assert first == second
