"""Tests for snapshot undo across every transition family.

Undo is LIFO on the state that produced the transition. It restores fields on
the existing object rather than handing back a replacement, and the snapshot it
restores from must survive whatever the caller does to the state afterwards.
"""

from collections.abc import Callable
from copy import deepcopy

import pytest

from shed.engine import (
    AtLeast,
    GameState,
    IllegalMoveError,
    Move,
    Phase,
    PickUp,
    Play,
    PlayerId,
    Rank,
    Reveal,
    SlotId,
    StateInvariantError,
    Zone,
)
from tests.conftest import (
    FIRST_SEAT,
    DeckPicker,
    build_play_state,
    build_setup_state,
    play_seeded_game,
)

SECOND_SEAT: PlayerId = PlayerId(1)
"""The opponent in the crafted two-player positions below."""

Fixture = Callable[[DeckPicker], tuple[GameState, Move]]
"""A crafted position paired with the one legal move a test will apply to it."""


def _ordinary_play(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a plain hand play that passes the turn.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: picker.take(Rank.SIX, 2),
            SECOND_SEAT: picker.take(Rank.FOUR, 1),
        },
    )
    return state, Play(Zone.HAND, Rank.SIX, 2)


def _play_with_refill(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a play whose replenishment draws from the deck.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: picker.take(Rank.SIX, 3),
            SECOND_SEAT: picker.take(Rank.FOUR, 1),
        },
        draw_count=6,
    )
    return state, Play(Zone.HAND, Rank.SIX, 2)


def _ten_burn(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a ten play that burns the pile and retains the turn.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: picker.many([Rank.TEN, Rank.FOUR]),
            SECOND_SEAT: picker.take(Rank.FIVE, 1),
        },
        discard=picker.many([Rank.THREE, Rank.KING]),
        constraint=AtLeast(Rank.KING),
    )
    return state, Play(Zone.HAND, Rank.TEN, 1)


def _batch_burn(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a four-card batch that burns the pile.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: [*picker.take(Rank.JACK, 4), picker.one(Rank.FOUR)],
            SECOND_SEAT: picker.take(Rank.FIVE, 1),
        },
        discard=picker.many([Rank.THREE]),
        constraint=AtLeast(Rank.THREE),
    )
    return state, Play(Zone.HAND, Rank.JACK, 4)


def _pickup(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a blocked hand whose only action is picking the pile up.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: picker.take(Rank.THREE, 2),
            SECOND_SEAT: picker.take(Rank.FIVE, 1),
        },
        discard=picker.many([Rank.FOUR, Rank.KING]),
        constraint=AtLeast(Rank.KING),
    )
    return state, PickUp()


def _successful_reveal(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a blind reveal that satisfies the constraint.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: [], SECOND_SEAT: picker.take(Rank.FIVE, 1)},
        face_down={FIRST_SEAT: {SlotId(0): picker.one(Rank.ACE), SlotId(2): picker.one(Rank.SIX)}},
        discard=picker.many([Rank.FOUR]),
        constraint=AtLeast(Rank.FOUR),
    )
    return state, Reveal(SlotId(0))


def _failed_reveal(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a blind reveal that fails and collects the pile.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: [], SECOND_SEAT: picker.take(Rank.FIVE, 1)},
        face_down={
            FIRST_SEAT: {SlotId(0): picker.one(Rank.THREE), SlotId(1): picker.one(Rank.SIX)}
        },
        discard=picker.many([Rank.KING]),
        constraint=AtLeast(Rank.KING),
    )
    return state, Reveal(SlotId(0))


def _terminal_play(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a last card whose ordinary play wins the game.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: picker.take(Rank.ACE, 1), SECOND_SEAT: picker.take(Rank.FIVE, 1)},
        discard=picker.many([Rank.KING]),
        constraint=AtLeast(Rank.KING),
    )
    return state, Play(Zone.HAND, Rank.ACE, 1)


def _terminal_burn(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a last card whose burn wins instead of retaining the turn.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: picker.take(Rank.TEN, 1), SECOND_SEAT: picker.take(Rank.FIVE, 1)},
        discard=picker.many([Rank.KING]),
        constraint=AtLeast(Rank.KING),
    )
    return state, Play(Zone.HAND, Rank.TEN, 1)


def _setup_submission(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a setup state where one arrangement is still outstanding after this.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state = build_setup_state(
        picker,
        hands={
            FIRST_SEAT: picker.many([Rank.THREE, Rank.FOUR, Rank.FIVE]),
            SECOND_SEAT: picker.many([Rank.SIX, Rank.SEVEN, Rank.EIGHT]),
        },
        face_up={
            FIRST_SEAT: picker.many([Rank.NINE, Rank.TEN, Rank.JACK]),
            SECOND_SEAT: picker.many([Rank.QUEEN, Rank.KING, Rank.ACE]),
        },
    )
    return state, state.get_legal_moves()[0]


def _setup_commit(picker: DeckPicker) -> tuple[GameState, Move]:
    """Build a setup state whose next arrangement commits everybody into PLAY.

    Args:
        picker: Source of every card in the position.

    Returns:
        The state and the move to apply.
    """
    state, first = _setup_submission(picker)
    state.apply_move(first)
    assert state.phase is Phase.SETUP
    return state, state.get_legal_moves()[-1]


FIXTURES: dict[str, Fixture] = {
    "ordinary_play": _ordinary_play,
    "play_with_refill": _play_with_refill,
    "ten_burn": _ten_burn,
    "batch_burn": _batch_burn,
    "pickup": _pickup,
    "successful_reveal": _successful_reveal,
    "failed_reveal": _failed_reveal,
    "terminal_play": _terminal_play,
    "terminal_burn": _terminal_burn,
    "setup_submission": _setup_submission,
    "setup_commit": _setup_commit,
}
"""One crafted position per transition family that undo must handle."""


@pytest.mark.parametrize("build", FIXTURES.values(), ids=list(FIXTURES))
def test_apply_then_undo_restores_the_canonical_state(build: Fixture, picker: DeckPicker) -> None:
    """Every transition family rolls back to exactly the position before it."""
    state, move = build(picker)
    before = deepcopy(state)

    transition = state.apply_move(move)
    assert state != before  # The transition really changed something.

    state.undo_move(transition)
    assert state == before


@pytest.mark.parametrize("build", FIXTURES.values(), ids=list(FIXTURES))
def test_undo_restores_the_object_every_caller_holds(build: Fixture, picker: DeckPicker) -> None:
    """Undo writes onto the existing state; it never returns a replacement."""
    state, move = build(picker)
    before = deepcopy(state)
    holder = state  # A second reference, as the runner and a search caller keep.

    transition = state.apply_move(move)
    result = state.undo_move(transition)

    assert result is None
    assert holder is state
    assert holder == before


@pytest.mark.parametrize("build", FIXTURES.values(), ids=list(FIXTURES))
def test_the_snapshot_survives_later_mutation(build: Fixture, picker: DeckPicker) -> None:
    """Mutating the state after a rollback cannot corrupt the undo record."""
    state, move = build(picker)
    before = deepcopy(state)

    transition = state.apply_move(move)
    state.undo_move(transition)
    # Wreck the restored state, then roll the same record back a second time.
    state.draw_pile.clear()
    state.discard_pile.clear()
    state.players[FIRST_SEAT].hand.clear()
    state.current_ply = 99
    state.undo_move(transition)

    assert state == before


@pytest.mark.parametrize("build", FIXTURES.values(), ids=list(FIXTURES))
def test_the_snapshot_does_not_alias_the_live_state(build: Fixture, picker: DeckPicker) -> None:
    """The record is an independent copy taken before mutation began."""
    state, move = build(picker)
    transition = state.apply_move(move)
    snapshot = deepcopy(transition.undo.before)

    state.draw_pile.clear()
    state.burned_cards.clear()
    for seat in state.seat_order:
        state.players[seat].hand.clear()

    assert transition.undo.before == snapshot


def test_lifo_undo_unwinds_a_whole_seeded_game() -> None:
    """Undoing every decision of a full game returns to the dealt position."""
    state = GameState.create(3, seed=99)
    dealt = deepcopy(state)

    log = play_seeded_game(state, seed=99)
    assert log.transitions

    for transition in reversed(log.transitions):
        state.undo_move(transition)

    assert state == dealt


def test_a_stale_legal_move_cannot_mutate_a_later_state(picker: DeckPicker) -> None:
    """A move tuple observed earlier is not authority over the position now."""
    state, move = _ordinary_play(picker)
    state.apply_move(move)
    after = deepcopy(state)

    with pytest.raises(IllegalMoveError):
        state.apply_move(move)  # Seat 0 already shed those cards, and has passed.
    assert state == after


def test_an_internal_failure_during_resolution_rolls_back(
    picker: DeckPicker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected error after mutation began restores the whole position."""
    state, move = _play_with_refill(picker)
    before = deepcopy(state)

    def explode(self: GameState, actor: PlayerId) -> tuple[object, ...]:
        """Fail the way an engine bug would, once cards have already moved.

        Args:
            self: The state being resolved.
            actor: The player who would have been replenished.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("simulated engine failure")

    monkeypatch.setattr(GameState, "_refill", explode)

    with pytest.raises(RuntimeError, match="simulated engine failure"):
        state.apply_move(move)
    assert state == before


def test_a_failed_postcondition_rolls_back(
    picker: DeckPicker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Postcondition validation is inside the rollback boundary, not after it."""
    state, move = _ordinary_play(picker)
    before = deepcopy(state)

    def reject(state: GameState) -> None:
        """Refuse whatever the transition produced.

        Args:
            state: The state that was just mutated.

        Raises:
            StateInvariantError: Always.
        """
        raise StateInvariantError("simulated postcondition failure")

    monkeypatch.setattr("shed.engine.state.validate_decision_boundary", reject)

    with pytest.raises(StateInvariantError, match="simulated postcondition failure"):
        state.apply_move(move)
    assert state == before
