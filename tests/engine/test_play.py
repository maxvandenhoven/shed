"""Tests for PLAY resolution: transfers, effects, burns, refills, and endings.

Each test crafts the smallest position that isolates one rule, applies exactly
one decision, and inspects the resulting state and the full events. Undo is
covered separately in ``test_undo.py``.
"""

from collections.abc import Sequence

import pytest

from shed.engine import (
    AtLeast,
    AtMost,
    BurnReason,
    Card,
    CardRevealed,
    CardsDrawn,
    CardsPlayed,
    Decision,
    GameEnded,
    GameState,
    IllegalMoveError,
    ObservedEvent,
    Outcome,
    Phase,
    PickUp,
    PileBurned,
    PilePickedUp,
    Play,
    PlayConstraint,
    PlayerId,
    Rank,
    Reveal,
    SlotId,
    Unrestricted,
    Zone,
    filter_events_for,
)
from tests.conftest import (
    FIRST_SEAT,
    UNRESTRICTED,
    DeckPicker,
    assert_cards_conserved,
    build_play_state,
)

SECOND_SEAT: PlayerId = PlayerId(1)
"""The opponent in every two-player fixture here."""


def _duel(
    picker: DeckPicker,
    *,
    hand: list[Card],
    opponent: list[Card] | None = None,
    face_up: list[Card] | None = None,
    face_down: dict[SlotId, Card] | None = None,
    discard: Sequence[Card] = (),
    constraint: PlayConstraint = UNRESTRICTED,
    draw_count: int = 0,
) -> GameState:
    """Craft a two-player PLAY state in which seat 0 is about to decide.

    The opponent exists so that passing the turn reaches a seat that still holds
    cards, which is what a real game guarantees: the profile ends the moment its
    first player runs out.

    Args:
        picker: Source of every card in the position.
        hand: Seat 0's hand.
        opponent: Seat 1's hand; one arbitrary card when omitted.
        face_up: Seat 0's face-up collection.
        face_down: Seat 0's face-down cards, keyed by slot.
        discard: The live pile, oldest first.
        constraint: Restriction the pile currently imposes.
        draw_count: Cards to leave in the draw pile.

    Returns:
        A valid PLAY state with seat 0 scheduled.
    """
    return build_play_state(
        picker,
        hands={
            FIRST_SEAT: hand,
            SECOND_SEAT: opponent if opponent is not None else picker.any_cards(1),
        },
        face_up={FIRST_SEAT: face_up} if face_up is not None else None,
        face_down={FIRST_SEAT: face_down} if face_down is not None else None,
        discard=discard,
        constraint=constraint,
        draw_count=draw_count,
    )


def test_ordinary_play_transfers_the_batch_and_advances(picker: DeckPicker) -> None:
    """A batch reaches the pile, sets its constraint, and passes the turn."""
    sixes = picker.take(Rank.SIX, 2)
    state = _duel(picker, hand=[*sixes, *picker.take(Rank.FOUR, 1)])

    transition = state.apply_move(Play(Zone.HAND, Rank.SIX, 2))

    assert state.discard_pile == sixes
    assert state.constraint == AtLeast(Rank.SIX)
    assert state.current_player == SECOND_SEAT
    assert state.current_ply == 1
    assert state.phase is Phase.PLAY
    assert transition.decision == Decision(FIRST_SEAT, Play(Zone.HAND, Rank.SIX, 2))
    assert transition.events == (
        CardsPlayed(player=FIRST_SEAT, source=Zone.HAND, cards=tuple(sixes)),
    )
    assert_cards_conserved(state)


def test_physical_cards_are_selected_by_ascending_identifier(picker: DeckPicker) -> None:
    """A rank/count play never asks which suits were meant; the engine decides."""
    sixes = picker.take(Rank.SIX, 3)
    state = _duel(picker, hand=list(reversed(sixes)))

    state.apply_move(Play(Zone.HAND, Rank.SIX, 2))

    assert state.discard_pile == sixes[:2]
    assert state.players[FIRST_SEAT].hand == [sixes[2]]


def test_face_up_cards_are_played_once_hand_and_deck_are_empty(picker: DeckPicker) -> None:
    """The face-up collection is an ordinary source, with its own play event."""
    kings = picker.take(Rank.KING, 2)
    state = _duel(picker, hand=[], face_up=kings, face_down={SlotId(0): picker.one(Rank.TWO)})

    transition = state.apply_move(Play(Zone.FACE_UP, Rank.KING, 2))

    assert state.players[FIRST_SEAT].face_up == []
    assert state.discard_pile == kings
    assert transition.events == (
        CardsPlayed(player=FIRST_SEAT, source=Zone.FACE_UP, cards=tuple(kings)),
    )


@pytest.mark.parametrize(
    ("rank", "before", "after"),
    [
        (Rank.EIGHT, AtLeast(Rank.THREE), AtLeast(Rank.EIGHT)),
        (Rank.SEVEN, Unrestricted(), AtMost(Rank.SEVEN)),
        (Rank.TWO, AtLeast(Rank.KING), AtLeast(Rank.TWO)),
        (Rank.NINE, AtMost(Rank.SEVEN), AtMost(Rank.SEVEN)),
        (Rank.NINE, AtLeast(Rank.KING), AtLeast(Rank.KING)),
        (Rank.JOKER, AtMost(Rank.SEVEN), Unrestricted()),
        (Rank.JOKER, AtLeast(Rank.KING), Unrestricted()),
    ],
)
def test_rank_effects_on_the_constraint(
    picker: DeckPicker, rank: Rank, before: PlayConstraint, after: PlayConstraint
) -> None:
    """Each rank leaves the documented constraint behind when nothing burns."""
    state = _duel(picker, hand=[picker.one(rank), picker.one(Rank.NINE)], constraint=before)

    state.apply_move(Play(Zone.HAND, rank, 1))

    assert state.constraint == after


def test_seven_and_nine_chain_keeps_the_restriction_until_a_rank_replaces_it(
    picker: DeckPicker,
) -> None:
    """A nine is transparent: only an ordinary play replaces the seven's bound.

    Both seats keep a face-down card so that shedding a hand never ends the game
    in the middle of the chain.
    """
    blind = {FIRST_SEAT: {SlotId(0): picker.one(Rank.ACE)}}
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: picker.many([Rank.SEVEN, Rank.NINE, Rank.FIVE]),
            SECOND_SEAT: picker.many([Rank.NINE, Rank.NINE, Rank.TWO]),
        },
        face_down={**blind, SECOND_SEAT: {SlotId(0): picker.one(Rank.ACE)}},
    )

    expected: list[tuple[Rank, PlayConstraint]] = [
        (Rank.SEVEN, AtMost(Rank.SEVEN)),  # Seat 0 imposes the restriction.
        (Rank.NINE, AtMost(Rank.SEVEN)),  # Seat 1 plays through it.
        (Rank.NINE, AtMost(Rank.SEVEN)),  # Seat 0 too; nines never replace it.
        (Rank.NINE, AtMost(Rank.SEVEN)),
        (Rank.FIVE, AtLeast(Rank.FIVE)),  # An ordinary rank finally replaces it.
        (Rank.TWO, AtLeast(Rank.TWO)),  # A two is always legal and resets low.
    ]
    for rank, constraint in expected:
        state.apply_move(Play(Zone.HAND, rank, 1))
        assert state.constraint == constraint

    assert state.current_ply == len(expected)


def test_an_eight_is_illegal_under_the_seven_restriction(picker: DeckPicker) -> None:
    """The restriction is enforced when the move is applied, not only listed."""
    state = _duel(picker, hand=picker.many([Rank.EIGHT, Rank.NINE]), constraint=AtMost(Rank.SEVEN))

    with pytest.raises(IllegalMoveError):
        state.apply_move(Play(Zone.HAND, Rank.EIGHT, 1))


def test_a_ten_burns_the_pile_and_keeps_the_turn(picker: DeckPicker) -> None:
    """A ten removes everything, including itself, and the actor decides again."""
    pile = picker.many([Rank.THREE, Rank.KING])
    ten = picker.one(Rank.TEN)
    state = _duel(
        picker,
        hand=[ten, picker.one(Rank.FOUR)],
        discard=pile,
        constraint=AtLeast(Rank.KING),
    )
    burned_before = list(state.burned_cards)

    transition = state.apply_move(Play(Zone.HAND, Rank.TEN, 1))

    assert state.discard_pile == []
    assert state.burned_cards == [*burned_before, *pile, ten]
    assert state.constraint == Unrestricted()
    assert state.current_player == FIRST_SEAT
    assert state.current_ply == 1
    assert transition.events == (
        CardsPlayed(player=FIRST_SEAT, source=Zone.HAND, cards=(ten,)),
        PileBurned(player=FIRST_SEAT, cards=(*pile, ten), reason=BurnReason.TEN),
    )
    assert_cards_conserved(state)


def test_four_played_together_burn_the_pile(picker: DeckPicker) -> None:
    """A four-card batch burns, and the burn reason names that rule."""
    pile = picker.many([Rank.THREE])
    jacks = picker.take(Rank.JACK, 4)
    state = _duel(
        picker,
        hand=[*jacks, picker.one(Rank.FOUR)],
        discard=pile,
        constraint=AtLeast(Rank.THREE),
    )
    burned_before = list(state.burned_cards)

    transition = state.apply_move(Play(Zone.HAND, Rank.JACK, 4))

    assert state.discard_pile == []
    assert state.burned_cards == [*burned_before, *pile, *jacks]
    assert state.constraint == Unrestricted()
    assert state.current_player == FIRST_SEAT
    assert transition.events[-1] == PileBurned(
        player=FIRST_SEAT, cards=(*pile, *jacks), reason=BurnReason.FOUR_OF_A_KIND
    )


def test_a_four_card_batch_must_still_be_legal(picker: DeckPicker) -> None:
    """The burn is an effect of a legal play, never a way around the constraint."""
    state = _duel(
        picker,
        hand=[*picker.take(Rank.KING, 4), picker.one(Rank.TWO)],
        constraint=AtLeast(Rank.ACE),
    )

    with pytest.raises(IllegalMoveError):
        state.apply_move(Play(Zone.HAND, Rank.KING, 4))
    assert state.discard_pile  # The pile that imposed the constraint is untouched.


def test_four_accumulated_across_actions_do_not_burn(picker: DeckPicker) -> None:
    """Only a single action of four burns; a fourth matching card just lands."""
    pile = picker.take(Rank.JACK, 3)
    fourth = picker.one(Rank.JACK)
    state = _duel(
        picker,
        hand=[fourth, picker.one(Rank.TWO)],
        discard=pile,
        constraint=AtLeast(Rank.JACK),
    )
    burned_before = list(state.burned_cards)

    transition = state.apply_move(Play(Zone.HAND, Rank.JACK, 1))

    assert state.discard_pile == [*pile, fourth]
    assert state.burned_cards == burned_before
    assert state.constraint == AtLeast(Rank.JACK)
    assert state.current_player == SECOND_SEAT
    assert transition.events == (CardsPlayed(player=FIRST_SEAT, source=Zone.HAND, cards=(fourth,)),)


def test_a_ten_takes_precedence_over_the_four_card_reason(picker: DeckPicker) -> None:
    """When both burn rules apply, the recorded reason is the ten."""
    tens = picker.take(Rank.TEN, 4)
    state = _duel(picker, hand=[*tens, picker.one(Rank.TWO)])

    transition = state.apply_move(Play(Zone.HAND, Rank.TEN, 4))

    burn = transition.events[-1]
    assert isinstance(burn, PileBurned)
    assert burn.reason is BurnReason.TEN


def test_pickup_transfers_the_pile_and_passes_the_turn(picker: DeckPicker) -> None:
    """A blocked actor takes the whole pile, clearing the restriction."""
    pile = picker.many([Rank.THREE, Rank.KING])
    hand = picker.take(Rank.SIX, 2)
    state = _duel(picker, hand=list(hand), discard=pile, constraint=AtLeast(Rank.KING))

    transition = state.apply_move(PickUp())

    assert state.players[FIRST_SEAT].hand == [*hand, *pile]
    assert state.discard_pile == []
    assert state.constraint == Unrestricted()
    assert state.current_player == SECOND_SEAT
    assert state.current_ply == 1
    assert transition.events == (PilePickedUp(player=FIRST_SEAT, cards=tuple(pile)),)
    assert_cards_conserved(state)


def test_pickup_returns_a_table_player_to_hand_play(picker: DeckPicker) -> None:
    """Cards taken from the pile become a hand again, which is played first."""
    pile = picker.many([Rank.KING])
    state = _duel(
        picker,
        hand=[],
        face_up=picker.take(Rank.SIX, 2),
        discard=pile,
        constraint=AtLeast(Rank.KING),
    )
    assert state.get_legal_moves() == (PickUp(),)

    state.apply_move(PickUp())

    assert state.players[FIRST_SEAT].hand == pile
    assert state.players[FIRST_SEAT].active_zone(len(state.draw_pile)) is Zone.HAND


def test_a_successful_reveal_resolves_as_a_single_card_play(picker: DeckPicker) -> None:
    """One ``CardRevealed`` event covers the whole transfer; no play event."""
    pile = picker.many([Rank.FIVE])
    king = picker.one(Rank.KING)
    state = _duel(
        picker,
        hand=[],
        face_down={SlotId(0): king, SlotId(1): picker.one(Rank.THREE)},
        discard=pile,
        constraint=AtLeast(Rank.FIVE),
    )

    transition = state.apply_move(Reveal(SlotId(0)))

    assert state.discard_pile == [*pile, king]
    assert state.constraint == AtLeast(Rank.KING)
    assert state.current_player == SECOND_SEAT
    assert transition.events == (
        CardRevealed(player=FIRST_SEAT, slot=SlotId(0), card=king, playable=True),
    )
    assert not any(isinstance(event, CardsPlayed) for event in transition.events)


def test_a_failed_reveal_collects_the_pile_and_the_card(picker: DeckPicker) -> None:
    """An unplayable blind card joins the pile in the actor's hand."""
    pile = picker.many([Rank.KING])
    three = picker.one(Rank.THREE)
    state = _duel(
        picker,
        hand=[],
        face_down={SlotId(0): three, SlotId(1): picker.one(Rank.ACE)},
        discard=pile,
        constraint=AtLeast(Rank.KING),
    )

    transition = state.apply_move(Reveal(SlotId(0)))

    assert state.players[FIRST_SEAT].hand == [*pile, three]
    assert state.discard_pile == []
    assert state.constraint == Unrestricted()
    assert state.current_player == SECOND_SEAT
    assert state.current_ply == 1
    assert transition.events == (
        CardRevealed(player=FIRST_SEAT, slot=SlotId(0), card=three, playable=False),
        PilePickedUp(player=FIRST_SEAT, cards=(*pile, three)),
    )
    assert_cards_conserved(state)


def test_a_revealed_ten_burns_and_keeps_the_turn(picker: DeckPicker) -> None:
    """Blind play resolves the same effects an ordinary play would."""
    pile = picker.many([Rank.KING])
    ten = picker.one(Rank.TEN)
    state = _duel(
        picker,
        hand=[],
        face_down={SlotId(0): ten, SlotId(1): picker.one(Rank.ACE)},
        discard=pile,
        constraint=AtLeast(Rank.KING),
    )

    transition = state.apply_move(Reveal(SlotId(0)))

    assert state.discard_pile == []
    assert state.current_player == FIRST_SEAT
    assert transition.events[-1] == PileBurned(
        player=FIRST_SEAT, cards=(*pile, ten), reason=BurnReason.TEN
    )


def test_slot_identifiers_stay_stable_when_a_slot_empties(picker: DeckPicker) -> None:
    """Revealing the middle slot leaves the others under their own IDs."""
    blind = {
        SlotId(0): picker.one(Rank.THREE),
        SlotId(1): picker.one(Rank.ACE),
        SlotId(2): picker.one(Rank.FOUR),
    }
    state = _duel(picker, hand=[], face_down=dict(blind))

    state.apply_move(Reveal(SlotId(1)))
    state.current_player = FIRST_SEAT  # Look at seat 0 again without a second deal.

    assert state.players[FIRST_SEAT].face_down == {
        SlotId(0): blind[SlotId(0)],
        SlotId(2): blind[SlotId(2)],
    }
    assert state.get_legal_moves() == (Reveal(SlotId(0)), Reveal(SlotId(2)))


def test_a_failed_final_reveal_does_not_win(picker: DeckPicker) -> None:
    """Picking the pile up is never a way to run out of cards."""
    pile = picker.many([Rank.KING])
    three = picker.one(Rank.THREE)
    state = _duel(
        picker, hand=[], face_down={SlotId(0): three}, discard=pile, constraint=AtLeast(Rank.KING)
    )

    state.apply_move(Reveal(SlotId(0)))

    assert state.phase is Phase.PLAY
    assert state.outcome is None
    assert not state.is_finished
    assert state.players[FIRST_SEAT].hand == [*pile, three]


def test_the_hand_refills_to_three_while_the_deck_lasts(picker: DeckPicker) -> None:
    """Drawing is automatic and private, and tops the hand up to the target."""
    state = _duel(picker, hand=picker.take(Rank.SIX, 3), draw_count=5)
    top_of_deck = list(reversed(state.draw_pile[-2:]))

    transition = state.apply_move(Play(Zone.HAND, Rank.SIX, 2))

    assert len(state.players[FIRST_SEAT].hand) == 3
    assert len(state.draw_pile) == 3
    assert transition.events[-1] == CardsDrawn(player=FIRST_SEAT, count=2, cards=tuple(top_of_deck))


def test_refill_stops_when_the_deck_runs_out(picker: DeckPicker) -> None:
    """A short deck refills what it can, and the hand may stay below target."""
    state = _duel(picker, hand=picker.take(Rank.SIX, 3), draw_count=1)

    state.apply_move(Play(Zone.HAND, Rank.SIX, 3))

    assert len(state.players[FIRST_SEAT].hand) == 1
    assert state.draw_pile == []
    assert not state.is_finished


def test_a_full_hand_is_never_trimmed_by_the_refill(picker: DeckPicker) -> None:
    """The refill helper only adds cards; a large hand is left alone.

    A hand above the target is what a pickup leaves behind, so this is the
    position the shared refill helper must not disturb.
    """
    swollen = picker.many([Rank.SIX, Rank.SIX, Rank.KING, Rank.QUEEN, Rank.ACE])
    state = _duel(picker, hand=swollen, draw_count=4)

    transition = state.apply_move(Play(Zone.HAND, Rank.SIX, 1))

    assert len(state.players[FIRST_SEAT].hand) == 4
    assert len(state.draw_pile) == 4
    assert not any(isinstance(event, CardsDrawn) for event in transition.events)


def test_shedding_the_hand_does_not_win_while_the_deck_can_refill(picker: DeckPicker) -> None:
    """The win check runs after replenishment, never before it."""
    state = _duel(picker, hand=picker.take(Rank.SIX, 3), draw_count=6)

    state.apply_move(Play(Zone.HAND, Rank.SIX, 3))

    assert not state.is_finished
    assert state.outcome is None
    assert len(state.players[FIRST_SEAT].hand) == 3
    assert state.current_player == SECOND_SEAT


def test_the_last_card_wins_and_ends_the_game(picker: DeckPicker) -> None:
    """Emptying every personal zone finishes the game immediately."""
    pile = picker.many([Rank.KING])
    ace = picker.one(Rank.ACE)
    state = _duel(picker, hand=[ace], discard=pile, constraint=AtLeast(Rank.KING))

    transition = state.apply_move(Play(Zone.HAND, Rank.ACE, 1))

    assert state.phase is Phase.FINISHED
    assert state.is_finished
    assert state.outcome == Outcome(winner=FIRST_SEAT)
    assert state.current_player is None
    assert state.current_ply == 1
    assert transition.events == (
        CardsPlayed(player=FIRST_SEAT, source=Zone.HAND, cards=(ace,)),
        GameEnded(outcome=Outcome(winner=FIRST_SEAT)),
    )
    assert_cards_conserved(state)


def test_a_final_burn_wins_instead_of_granting_another_turn(picker: DeckPicker) -> None:
    """The retained turn never survives the win check."""
    pile = picker.many([Rank.KING])
    ten = picker.one(Rank.TEN)
    state = _duel(picker, hand=[ten], discard=pile, constraint=AtLeast(Rank.KING))

    transition = state.apply_move(Play(Zone.HAND, Rank.TEN, 1))

    assert state.outcome == Outcome(winner=FIRST_SEAT)
    assert state.current_player is None
    assert isinstance(transition.events[-2], PileBurned)
    assert transition.events[-1] == GameEnded(outcome=Outcome(winner=FIRST_SEAT))


def test_a_final_successful_reveal_wins(picker: DeckPicker) -> None:
    """The last face-down card can end the game just like a hand card."""
    ace = picker.one(Rank.ACE)
    state = _duel(picker, hand=[], face_down={SlotId(0): ace})

    state.apply_move(Reveal(SlotId(0)))

    assert state.outcome == Outcome(winner=FIRST_SEAT)
    assert state.players[FIRST_SEAT].remaining_count == 0


def test_a_finished_game_accepts_no_further_move(picker: DeckPicker) -> None:
    """FINISHED schedules nobody and refuses every decision."""
    state = _duel(picker, hand=[picker.one(Rank.ACE)])
    state.apply_move(Play(Zone.HAND, Rank.ACE, 1))

    assert state.get_legal_moves() == ()
    with pytest.raises(IllegalMoveError, match="finished"):
        state.apply_move(PickUp())


def test_setup_decisions_do_not_count_as_plies() -> None:
    """``current_ply`` counts resolved PLAY decisions only."""
    state = GameState.create(3, seed=7)
    while state.phase is Phase.SETUP:
        state.apply_move(state.get_legal_moves()[0])
        assert state.current_ply == 0

    state.apply_move(state.get_legal_moves()[0])
    assert state.current_ply == 1


def test_events_filter_down_to_what_each_recipient_may_know(picker: DeckPicker) -> None:
    """Only the drawing player learns which cards a refill produced."""
    state = _duel(picker, hand=picker.take(Rank.SIX, 3), draw_count=5)
    events: tuple[ObservedEvent, ...] = state.apply_move(Play(Zone.HAND, Rank.SIX, 2)).events

    mine = filter_events_for(events, FIRST_SEAT)
    theirs = filter_events_for(events, SECOND_SEAT)
    drawn_by_me = [event for event in mine if isinstance(event, CardsDrawn)]
    drawn_by_them = [event for event in theirs if isinstance(event, CardsDrawn)]

    assert drawn_by_me and drawn_by_me[0].cards is not None
    assert drawn_by_them and drawn_by_them[0].cards is None
    assert drawn_by_them[0].count == 2
