"""Tests for legal-move generation in PLAY, using crafted valid states.

Play resolution does not exist yet, so these tests build valid PLAY positions
directly and inspect what the generator offers. They never apply a play.
"""

from copy import deepcopy

import pytest

from shed.engine import (
    AtLeast,
    AtMost,
    Card,
    GameState,
    IllegalMoveError,
    Outcome,
    Phase,
    PickUp,
    Play,
    PlayConstraint,
    PlayerId,
    Rank,
    Reveal,
    SlotId,
    StateInvariantError,
    Unrestricted,
    Zone,
    can_play_rank,
)
from tests.conftest import UNRESTRICTED, DeckPicker, build_play_state, plays_in

SPECIAL_RANKS = (Rank.TWO, Rank.NINE, Rank.TEN, Rank.JOKER)
CONSTRAINTS: tuple[PlayConstraint, ...] = (
    Unrestricted(),
    AtLeast(Rank.KING),
    AtMost(Rank.SEVEN),
    AtLeast(Rank.THREE),
)


def _solo_state(
    picker: DeckPicker,
    hand: list[Card],
    *,
    constraint: PlayConstraint = UNRESTRICTED,
) -> GameState:
    """Craft a two-player PLAY state where seat 0 holds the given hand.

    Args:
        picker: Source of cards.
        hand: Seat zero's hand.
        constraint: Restriction the next ordinary rank must satisfy.

    Returns:
        A valid PLAY state with seat 0 to act.
    """
    return build_play_state(
        picker, hands={PlayerId(0): hand, PlayerId(1): []}, constraint=constraint
    )


@pytest.mark.parametrize(
    ("constraint", "rank", "expected"),
    [
        (Unrestricted(), Rank.THREE, True),
        (AtLeast(Rank.EIGHT), Rank.EIGHT, True),
        (AtLeast(Rank.EIGHT), Rank.ACE, True),
        (AtLeast(Rank.EIGHT), Rank.SEVEN, False),
        (AtMost(Rank.SEVEN), Rank.SEVEN, True),
        (AtMost(Rank.SEVEN), Rank.THREE, True),
        (AtMost(Rank.SEVEN), Rank.EIGHT, False),
    ],
)
def test_ordinary_rank_legality(constraint: PlayConstraint, rank: Rank, expected: bool) -> None:
    """Equal and matching ranks pass; ranks outside the bound do not."""
    assert can_play_rank(rank, constraint) is expected


@pytest.mark.parametrize("constraint", CONSTRAINTS)
@pytest.mark.parametrize("rank", SPECIAL_RANKS)
def test_special_ranks_ignore_every_constraint(constraint: PlayConstraint, rank: Rank) -> None:
    """Twos, nines, tens, and jokers are legal whatever the pile demands."""
    assert can_play_rank(rank, constraint) is True


def test_joker_strength_never_comes_from_its_enum_value() -> None:
    """A joker is legal by exception, not because 15 outranks the bound."""
    assert can_play_rank(Rank.JOKER, AtMost(Rank.THREE)) is True
    assert can_play_rank(Rank.ACE, AtMost(Rank.THREE)) is False


def test_every_batch_count_of_a_playable_rank_is_offered(picker: DeckPicker) -> None:
    """A rank held n times yields exactly n rank/count actions."""
    state = _solo_state(picker, picker.take(Rank.FIVE, 3) + picker.take(Rank.KING, 1))
    moves = state.observe(PlayerId(0)).legal_moves
    assert moves == (
        Play(Zone.HAND, Rank.FIVE, 1),
        Play(Zone.HAND, Rank.FIVE, 2),
        Play(Zone.HAND, Rank.FIVE, 3),
        Play(Zone.HAND, Rank.KING, 1),
    )


def test_moves_are_ordered_by_rank_then_count(picker: DeckPicker) -> None:
    """Generation order is deterministic and independent of card order."""
    hand = picker.take(Rank.ACE, 2) + picker.take(Rank.THREE, 2) + picker.take(Rank.TEN, 1)
    state = _solo_state(picker, hand)
    moves = state.observe(PlayerId(0)).legal_moves
    assert [(move.rank, move.count) for move in plays_in(moves)] == [
        (Rank.THREE, 1),
        (Rank.THREE, 2),
        (Rank.TEN, 1),
        (Rank.ACE, 1),
        (Rank.ACE, 2),
    ]
    state.players[PlayerId(0)].hand.reverse()
    assert state.get_legal_moves() == moves


def test_illegal_ranks_are_filtered_but_specials_survive(picker: DeckPicker) -> None:
    """Under ``AtMost(SEVEN)`` an eight is illegal while the specials remain."""
    hand = picker.many([Rank.EIGHT, Rank.SIX, Rank.TWO, Rank.NINE, Rank.TEN, Rank.JOKER])
    state = _solo_state(picker, hand, constraint=AtMost(Rank.SEVEN))
    ranks = {move.rank for move in plays_in(state.get_legal_moves())}
    assert ranks == {Rank.SIX, Rank.TWO, Rank.NINE, Rank.TEN, Rank.JOKER}


def test_at_least_constraint_accepts_equal_and_higher_ranks(picker: DeckPicker) -> None:
    """``AtLeast(TEN)`` accepts ten and above and rejects lower ordinary ranks."""
    hand = picker.many([Rank.NINE, Rank.EIGHT, Rank.TEN, Rank.JACK])
    state = _solo_state(picker, hand, constraint=AtLeast(Rank.TEN))
    ranks = {move.rank for move in plays_in(state.get_legal_moves())}
    assert ranks == {Rank.NINE, Rank.TEN, Rank.JACK}  # The nine is a special, not a comparison.


def test_blocked_hand_forces_pickup(picker: DeckPicker) -> None:
    """With nothing playable, picking up is the only legal action."""
    state = _solo_state(picker, picker.take(Rank.EIGHT, 3), constraint=AtMost(Rank.SEVEN))
    assert state.get_legal_moves() == (PickUp(),)


def test_pickup_is_never_offered_beside_a_playable_batch(picker: DeckPicker) -> None:
    """Voluntary pickup does not exist in this profile."""
    state = _solo_state(
        picker, picker.take(Rank.EIGHT, 2) + picker.take(Rank.TWO, 1), constraint=AtMost(Rank.SEVEN)
    )
    moves = state.get_legal_moves()
    assert PickUp() not in moves
    assert moves == (Play(Zone.HAND, Rank.TWO, 1),)


def test_hand_is_played_before_the_table(picker: DeckPicker) -> None:
    """A non-empty hand is the only active zone, even with face-up cards left."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.FOUR, 1), PlayerId(1): []},
        face_up={PlayerId(0): picker.take(Rank.ACE, 3)},
        face_down={PlayerId(0): {SlotId(0): picker.one(Rank.KING)}},
    )
    assert state.get_legal_moves() == (Play(Zone.HAND, Rank.FOUR, 1),)


def test_face_up_is_active_only_once_hand_and_deck_are_empty(picker: DeckPicker) -> None:
    """The face-up collection becomes playable after the deck runs out."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): [], PlayerId(1): []},
        face_up={PlayerId(0): picker.take(Rank.SEVEN, 2) + picker.take(Rank.ACE, 1)},
        face_down={PlayerId(0): {SlotId(1): picker.one(Rank.KING)}},
    )
    assert state.get_legal_moves() == (
        Play(Zone.FACE_UP, Rank.SEVEN, 1),
        Play(Zone.FACE_UP, Rank.SEVEN, 2),
        Play(Zone.FACE_UP, Rank.ACE, 1),
    )


def test_an_empty_hand_beside_a_stocked_deck_is_an_invariant_error(picker: DeckPicker) -> None:
    """The engine must refill before a decision is requested, not improvise."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): [], PlayerId(1): []},
        face_up={PlayerId(0): picker.take(Rank.SEVEN, 1)},
        draw_count=3,
    )
    with pytest.raises(StateInvariantError, match="refill"):
        state.get_legal_moves()


def test_every_remaining_face_down_slot_is_offered(picker: DeckPicker) -> None:
    """Blind play offers each remaining slot ascending, and only those."""
    blind = {SlotId(0): picker.one(Rank.THREE), SlotId(2): picker.one(Rank.ACE)}
    state = build_play_state(
        picker,
        hands={PlayerId(0): [], PlayerId(1): []},
        face_down={PlayerId(0): blind},
    )
    assert state.get_legal_moves() == (Reveal(SlotId(0)), Reveal(SlotId(2)))


def test_face_down_moves_never_leak_the_hidden_ranks(picker: DeckPicker) -> None:
    """Unplayable blind cards are still offered; the rank is not consulted."""
    blind = {
        SlotId(0): picker.one(Rank.ACE),
        SlotId(1): picker.one(Rank.KING),
        SlotId(2): picker.one(Rank.QUEEN),
    }
    state = build_play_state(
        picker,
        hands={PlayerId(0): [], PlayerId(1): []},
        face_down={PlayerId(0): blind},
        constraint=AtMost(Rank.THREE),
    )
    offered = state.get_legal_moves()
    assert offered == (Reveal(SlotId(0)), Reveal(SlotId(1)), Reveal(SlotId(2)))

    swapped = deepcopy(state)
    slots = swapped.players[PlayerId(0)].face_down
    slots[SlotId(0)], slots[SlotId(1)] = slots[SlotId(1)], slots[SlotId(0)]
    assert swapped.get_legal_moves() == offered


def test_an_exhausted_actor_is_not_a_decision_boundary(picker: DeckPicker) -> None:
    """An actor with no cards means termination was never resolved.

    Empty legal moves stay correct for a finished game or a non-acting viewer,
    but scheduling a player who holds nothing is an engine bug, so the position
    is refused instead of yielding a decision nobody can make.
    """
    exhausted = {PlayerId(0): [], PlayerId(1): []}
    with pytest.raises(StateInvariantError, match="holds no cards"):
        build_play_state(picker, hands=exhausted)

    state = build_play_state(picker, hands=exhausted, validate=False)
    before = deepcopy(state)
    with pytest.raises(StateInvariantError, match="holds no cards"):
        state.observe(PlayerId(0))
    with pytest.raises(StateInvariantError, match="holds no cards"):
        state.get_legal_moves()
    assert state == before  # The rejected observation mutated nothing.


def test_only_the_current_actor_receives_moves(picker: DeckPicker) -> None:
    """Views for other seats generate nothing, whatever those seats hold."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.FOUR, 1), PlayerId(1): picker.take(Rank.FIVE, 2)},
        current_player=PlayerId(0),
    )
    assert state.observe(PlayerId(1)).legal_moves == ()
    assert state.observe(PlayerId(0)).legal_moves


def test_a_finished_game_offers_no_moves(picker: DeckPicker) -> None:
    """FINISHED states never produce decisions."""
    state = build_play_state(
        picker, hands={PlayerId(0): picker.take(Rank.FOUR, 1), PlayerId(1): []}
    )
    state.phase = Phase.FINISHED
    state.current_player = None
    state.outcome = Outcome(winner=PlayerId(1))
    assert state.get_legal_moves() == ()
    assert state.observe(PlayerId(0)).legal_moves == ()


def test_legal_moves_need_no_observation(
    picker: DeckPicker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generation reads the state directly; it builds no view on the way.

    The dependency runs one way -- ``observe`` calls ``get_legal_moves`` -- so
    both are sabotaged here and generation must still succeed.
    """

    def unreachable(*args: object, **kwargs: object) -> None:
        """Fail if legality generation tries to observe or build a view.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("get_legal_moves must not construct an observation")

    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.SIX, 2), PlayerId(1): []},
    )
    monkeypatch.setattr(GameState, "observe", unreachable)
    monkeypatch.setattr("shed.engine.state.PlayerView", unreachable)

    assert state.get_legal_moves() == (
        Play(Zone.HAND, Rank.SIX, 1),
        Play(Zone.HAND, Rank.SIX, 2),
    )


def test_views_carry_the_actors_moves_and_nobody_elses(picker: DeckPicker) -> None:
    """Each view holds exactly the state's legal moves, or none at all."""
    state = build_play_state(
        picker,
        hands={
            PlayerId(0): picker.take(Rank.SIX, 2),
            PlayerId(1): picker.take(Rank.NINE, 1),
            PlayerId(2): picker.take(Rank.ACE, 1),
        },
        current_player=PlayerId(1),
    )
    expected = state.get_legal_moves()
    assert expected  # The actor always has a decision.

    for seat in state.seat_order:
        view = state.observe(seat)
        assert view.legal_moves == (expected if seat == state.current_player else ())


def test_generation_does_not_mutate_the_state(picker: DeckPicker) -> None:
    """Legal-move generation is pure with respect to the authoritative state."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.SIX, 2), PlayerId(1): picker.take(Rank.TWO, 1)},
    )
    before = deepcopy(state)
    state.get_legal_moves()
    assert state == before


def test_play_resolution_is_not_implemented_and_says_so(picker: DeckPicker) -> None:
    """A legal play is refused explicitly rather than faking a transition."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.SIX, 2), PlayerId(1): []},
    )
    before = deepcopy(state)
    move = state.get_legal_moves()[0]
    with pytest.raises(NotImplementedError, match="PLAY resolution is not implemented"):
        state.apply_move(move)
    assert state == before


def test_illegal_play_moves_are_rejected_before_the_unimplemented_path(picker: DeckPicker) -> None:
    """Validation still runs first: an illegal batch is an illegal move."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.SIX, 2), PlayerId(1): []},
        constraint=AtLeast(Rank.KING),
    )
    for move in (
        Play(Zone.HAND, Rank.SIX, 1),
        Play(Zone.HAND, Rank.KING, 1),
        Play(Zone.FACE_UP, Rank.SIX, 1),
        Reveal(SlotId(0)),
    ):
        with pytest.raises(IllegalMoveError):
            state.apply_move(move)
