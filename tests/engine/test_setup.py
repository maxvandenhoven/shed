"""Tests for arrangement generation, private submissions, commitment, and undo."""

from copy import deepcopy
from itertools import combinations

import pytest

from shed.engine import (
    Arrange,
    ArrangementCommitted,
    CardId,
    GameState,
    IllegalMoveError,
    Phase,
    PickUp,
    Play,
    PlayerId,
    Rank,
    Ruleset,
    Zone,
)
from tests.conftest import (
    DeckPicker,
    arrangement,
    arranges_in,
    build_setup_state,
    commit_unchanged,
)


def _owned_ids(state: GameState, player: PlayerId) -> set[CardId]:
    """Return the six hand and face-up card identifiers a player may arrange.

    Args:
        state: A SETUP state.
        player: The seat to inspect.

    Returns:
        The identifiers of that player's arrangeable cards.
    """
    cards = state.players[player]
    return {card.id for card in (*cards.hand, *cards.face_up)}


def test_setup_offers_exactly_twenty_unique_arrangements(ruleset: Ruleset) -> None:
    """Choosing three of six cards yields all 20 combinations, each once."""
    state = ruleset.create_initial_state(3, seed=21)
    moves = state.get_legal_moves()
    assert len(moves) == 20
    assert len(set(moves)) == 20
    assert all(isinstance(move, Arrange) for move in moves)


def test_arrangements_are_canonical_and_deterministically_ordered(ruleset: Ruleset) -> None:
    """Every arrangement has ascending owned IDs, in stable combination order."""
    state = ruleset.create_initial_state(4, seed=22)
    actor = state.current_player
    assert actor is not None
    owned = sorted(_owned_ids(state, actor))
    moves = state.get_legal_moves()

    assert [move.face_up_cards for move in arranges_in(moves)] == list(combinations(owned, 3))
    assert state.get_legal_moves() == moves


def test_keeping_the_dealt_arrangement_is_legal(ruleset: Ruleset) -> None:
    """A player may leave their dealt face-up cards exactly as they are."""
    state = ruleset.create_initial_state(2, seed=23)
    actor = state.current_player
    assert actor is not None
    unchanged = arrangement(card.id for card in state.players[actor].face_up)
    assert unchanged in state.get_legal_moves()
    ruleset.apply_move(state, unchanged)
    assert state.players[actor].face_up  # Still three cards; commit comes later.


def test_arrangements_naming_unowned_cards_are_rejected(ruleset: Ruleset) -> None:
    """Legality checks ownership, not just the shape of the submission."""
    state = ruleset.create_initial_state(2, seed=24)
    actor = state.current_player
    assert actor is not None
    foreign = sorted(set(range(54)) - _owned_ids(state, actor))[:3]
    before = deepcopy(state)
    with pytest.raises(IllegalMoveError, match="not legal"):
        ruleset.apply_move(state, arrangement(CardId(value) for value in foreign))
    assert state == before


@pytest.mark.parametrize("move", [Play(Zone.HAND, Rank.FIVE, 1), PickUp()], ids=["play", "pickup"])
def test_play_moves_are_rejected_during_setup(ruleset: Ruleset, move: object) -> None:
    """Only arrangements resolve in SETUP, and rejection mutates nothing."""
    state = ruleset.create_initial_state(2, seed=25)
    before = deepcopy(state)
    with pytest.raises(IllegalMoveError):
        ruleset.apply_move(state, move)  # ty: ignore[invalid-argument-type]
    assert state == before


def test_a_non_move_object_is_rejected(ruleset: Ruleset) -> None:
    """Decoded rubbish never reaches the transition helpers."""
    state = ruleset.create_initial_state(2, seed=26)
    with pytest.raises(IllegalMoveError, match="is not a move"):
        ruleset.apply_move(state, "arrange")  # ty: ignore[invalid-argument-type]


def test_submissions_stay_private_until_everyone_has_chosen(ruleset: Ruleset) -> None:
    """A stored arrangement changes no visible card and emits no event."""
    state = ruleset.create_initial_state(3, seed=27)
    actor = state.current_player
    assert actor is not None
    views_before = {seat: ruleset.observe(state, seat) for seat in state.seat_order}
    dealt = tuple(sorted(card.id for card in state.players[actor].face_up))
    swap = next(
        move for move in arranges_in(state.get_legal_moves()) if move.face_up_cards != dealt
    )

    transition = ruleset.apply_move(state, swap)

    assert transition.events == ()
    assert state.phase is Phase.SETUP
    assert state.setup is not None
    assert state.setup.submissions[actor] == swap
    assert actor not in state.setup.pending
    for seat in state.seat_order:
        seen = ruleset.observe(state, seat)
        assert seen.players == views_before[seat].players
        assert seen.hand == views_before[seat].hand


def test_view_never_exposes_pending_submissions(ruleset: Ruleset) -> None:
    """Setup submissions are absent from every observation, including the actor's."""
    state = ruleset.create_initial_state(2, seed=28)
    ruleset.apply_move(state, state.get_legal_moves()[7])
    for seat in state.seat_order:
        view = ruleset.observe(state, seat)
        assert not hasattr(view, "setup")
        assert not hasattr(view, "submissions")


def test_commitment_applies_every_arrangement_at_once(ruleset: Ruleset) -> None:
    """The final submission commits all arrangements and emits all events."""
    state = ruleset.create_initial_state(3, seed=29)
    chosen: dict[PlayerId, Arrange] = {}
    events = ()
    while state.phase is Phase.SETUP:
        actor = state.current_player
        assert actor is not None
        move = state.get_legal_moves()[5]
        assert isinstance(move, Arrange)
        chosen[actor] = move
        events = ruleset.apply_move(state, move).events

    committed = [event for event in events if isinstance(event, ArrangementCommitted)]
    assert len(committed) == len(events) == 3
    assert [event.player for event in committed] == [1, 2, 0]  # Clockwise after dealer 0.
    for event in committed:
        face_up = state.players[event.player].face_up
        assert event.face_up == tuple(face_up)
        assert tuple(card.id for card in face_up) == chosen[event.player].face_up_cards
        assert len(state.players[event.player].hand) == 3
        assert not {card.id for card in state.players[event.player].hand} & set(
            chosen[event.player].face_up_cards
        )


def test_commitment_enters_play_and_drops_setup_state(ruleset: Ruleset) -> None:
    """PLAY starts with no setup bookkeeping, ply zero, and a real actor."""
    state = ruleset.create_initial_state(4, seed=30)
    commit_unchanged(ruleset, state)
    assert state.phase is Phase.PLAY
    assert state.setup is None
    assert state.current_ply == 0
    assert state.current_player in state.seat_order
    assert state.get_legal_moves()


def test_opener_is_the_lowest_ordinary_rank_after_the_dealer(picker: DeckPicker) -> None:
    """The first rank found in the 3-to-ace search decides who opens."""
    ruleset = Ruleset()
    state = build_setup_state(
        picker,
        hands={
            PlayerId(0): picker.many([Rank.SIX, Rank.KING, Rank.ACE]),
            PlayerId(1): picker.many([Rank.EIGHT, Rank.QUEEN, Rank.TEN]),
            PlayerId(2): picker.many([Rank.FOUR, Rank.NINE, Rank.TWO]),
        },
        face_up={seat: picker.any_cards(3) for seat in (PlayerId(0), PlayerId(1), PlayerId(2))},
    )
    commit_unchanged(ruleset, state)
    assert state.current_player == 2


def test_opener_ties_break_clockwise_after_the_dealer(picker: DeckPicker) -> None:
    """Two holders of the deciding rank: the one nearer after the dealer opens."""
    ruleset = Ruleset()
    state = build_setup_state(
        picker,
        hands={
            PlayerId(0): picker.many([Rank.THREE, Rank.KING, Rank.ACE]),
            PlayerId(1): picker.many([Rank.EIGHT, Rank.QUEEN, Rank.TEN]),
            PlayerId(2): picker.many([Rank.THREE, Rank.NINE, Rank.JACK]),
        },
        face_up={seat: picker.any_cards(3) for seat in (PlayerId(0), PlayerId(1), PlayerId(2))},
        dealer=PlayerId(1),
    )
    commit_unchanged(ruleset, state)
    assert state.current_player == 2  # Order after dealer 1 is 2, 0, 1.


def test_opener_falls_back_through_the_rank_order_to_twos_and_jokers(
    picker: DeckPicker,
) -> None:
    """Specials come last: a two only opens when no ordinary rank is held."""
    ruleset = Ruleset()
    state = build_setup_state(
        picker,
        hands={
            PlayerId(0): picker.take(Rank.JOKER, 2) + picker.take(Rank.TWO, 1),
            PlayerId(1): picker.take(Rank.TWO, 3),
        },
        face_up={seat: picker.any_cards(3) for seat in (PlayerId(0), PlayerId(1))},
    )
    commit_unchanged(ruleset, state)
    assert state.current_player == 1  # Twos beat jokers; seat 1 follows the dealer.


def test_opener_is_chosen_from_hands_after_arranging_not_before(picker: DeckPicker) -> None:
    """Swapping a low card onto the table moves the opening decision elsewhere."""
    ruleset = Ruleset()
    low, high = picker.one(Rank.THREE), picker.one(Rank.KING)
    state = build_setup_state(
        picker,
        hands={
            PlayerId(0): [low, *picker.take(Rank.QUEEN, 2)],
            PlayerId(1): picker.many([Rank.FOUR, Rank.NINE, Rank.ACE]),
        },
        face_up={
            PlayerId(0): [high, *picker.take(Rank.TEN, 2)],
            PlayerId(1): picker.take(Rank.SIX, 3),
        },
    )
    # Seat 1 keeps its cards; seat 0 buries its three face up, keeping the king.
    ruleset.apply_move(state, arrangement(card.id for card in state.players[PlayerId(1)].face_up))
    hide = [low] + [card for card in state.players[PlayerId(0)].face_up][:2]
    ruleset.apply_move(state, arrangement(card.id for card in hide))

    assert state.phase is Phase.PLAY
    assert low.id in {card.id for card in state.players[PlayerId(0)].face_up}
    assert state.current_player == 1  # Seat 1's four is now the lowest hand rank.


def test_setup_undo_restores_a_pending_submission(ruleset: Ruleset) -> None:
    """Undoing a stored arrangement removes it and restores the actor."""
    state = ruleset.create_initial_state(3, seed=31)
    before = deepcopy(state)
    transition = ruleset.apply_move(state, state.get_legal_moves()[2])
    assert state != before

    ruleset.undo_move(state, transition)
    assert state == before
    assert state.setup is not None
    assert state.setup.submissions == {}


def test_setup_undo_restores_the_collective_commitment(ruleset: Ruleset) -> None:
    """Undoing the final submission returns the game to SETUP, unarranged."""
    state = ruleset.create_initial_state(2, seed=32)
    ruleset.apply_move(state, state.get_legal_moves()[4])
    before_commit = deepcopy(state)
    transition = ruleset.apply_move(state, state.get_legal_moves()[9])
    assert state.phase is Phase.PLAY

    ruleset.undo_move(state, transition)
    assert state == before_commit
    assert state.phase is Phase.SETUP
    assert state.setup is not None
    assert len(state.setup.pending) == 1


def test_undo_snapshots_survive_later_mutation(ruleset: Ruleset) -> None:
    """An undo record stays usable after the state is mutated again."""
    state = ruleset.create_initial_state(2, seed=33)
    before = deepcopy(state)
    transition = ruleset.apply_move(state, state.get_legal_moves()[1])

    ruleset.undo_move(state, transition)
    state.players[PlayerId(0)].hand.clear()
    state.draw_pile.clear()
    ruleset.undo_move(state, transition)
    assert state == before


def test_undo_rewinds_the_caller_s_own_state_object(ruleset: Ruleset) -> None:
    """Undo restores fields in place; holders of the reference see the rewind."""
    state = ruleset.create_initial_state(2, seed=34)
    alias = state
    transition = ruleset.apply_move(state, state.get_legal_moves()[0])
    ruleset.undo_move(state, transition)
    assert alias.setup is not None
    assert alias.setup.submissions == {}
    assert alias.phase is Phase.SETUP


def test_setup_decisions_do_not_advance_the_play_counter(ruleset: Ruleset) -> None:
    """``current_ply`` counts resolved PLAY decisions only."""
    state = ruleset.create_initial_state(5, seed=35)
    while state.phase is Phase.SETUP:
        assert state.current_ply == 0
        ruleset.apply_move(state, state.get_legal_moves()[0])
    assert state.current_ply == 0
