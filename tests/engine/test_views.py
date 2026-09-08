"""Tests for player views: independence, immutability, and the info boundary."""

from copy import deepcopy
from dataclasses import fields, replace

import pytest

from shed.engine import (
    AtLeast,
    GameState,
    PlayerId,
    PlayerView,
    Rank,
    Ruleset,
    SlotId,
    StateInvariantError,
    Zone,
    build_view,
    filter_events_for,
)
from tests.conftest import DeckPicker, build_play_state, plays_in


def _dealt(ruleset: Ruleset) -> GameState:
    """Deal a reproducible three-player game.

    Args:
        ruleset: Ruleset performing the deal.

    Returns:
        A freshly dealt SETUP state.
    """
    return ruleset.create_initial_state(3, seed=1234)


def test_view_is_an_independent_snapshot(ruleset: Ruleset) -> None:
    """Mutating the state afterwards cannot change an already-built view."""
    state = _dealt(ruleset)
    view = ruleset.observe(state, PlayerId(0))
    hand_before = view.hand
    public_before = view.players

    state.players[PlayerId(0)].hand.clear()
    state.players[PlayerId(1)].face_up.clear()
    state.draw_pile.clear()

    assert view.hand == hand_before
    assert view.players == public_before
    assert view.draw_count == 27


def test_view_collections_are_immutable(ruleset: Ruleset) -> None:
    """Every exposed collection is a tuple of frozen elements."""
    state = _dealt(ruleset)
    view = ruleset.observe(state, PlayerId(0))
    for item in fields(PlayerView):
        value = getattr(view, item.name)
        assert not isinstance(value, list | dict | set)
    assert isinstance(view.hand, tuple)
    assert isinstance(view.players, tuple)
    with pytest.raises(AttributeError):
        view.current_ply = 5  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        view.players[0].hand_count = 0  # ty: ignore[invalid-assignment]


def test_repeated_observation_yields_equal_but_separate_snapshots(ruleset: Ruleset) -> None:
    """Two observations of one state are equal without sharing mutable objects."""
    state = _dealt(ruleset)
    first = ruleset.observe(state, PlayerId(1))
    second = ruleset.observe(state, PlayerId(1))
    assert first == second
    assert first is not second


def test_observation_does_not_mutate_the_state(ruleset: Ruleset) -> None:
    """``observe`` is read-only: it never refills, advances, or repairs."""
    state = _dealt(ruleset)
    before = deepcopy(state)
    for seat in state.seat_order:
        ruleset.observe(state, seat)
    assert state == before


def test_view_hides_opponent_hands_and_face_down_identities(ruleset: Ruleset) -> None:
    """Only the viewer's own hand is identified; blind cards never are."""
    state = _dealt(ruleset)
    view = ruleset.observe(state, PlayerId(0))
    own = {card.id for card in view.hand}
    assert own == {card.id for card in state.players[PlayerId(0)].hand}

    hidden = {card.id for seat in (1, 2) for card in state.players[PlayerId(seat)].hand}
    hidden |= {
        card.id for seat in state.seat_order for card in state.players[seat].face_down.values()
    }
    exposed = own | {card.id for public in view.players for card in public.face_up}
    assert not hidden & exposed

    for public in view.players:
        assert public.face_down_slots == (0, 1, 2)
        assert public.hand_count == 3


def test_view_exposes_deck_size_but_not_deck_order(ruleset: Ruleset) -> None:
    """Agents learn how many cards remain, never which ones or in what order."""
    state = _dealt(ruleset)
    view = ruleset.observe(state, PlayerId(2))
    assert view.draw_count == len(state.draw_pile)
    assert not any(isinstance(getattr(view, item.name), list) for item in fields(PlayerView))
    assert "draw_pile" not in {item.name for item in fields(PlayerView)}


def test_me_returns_the_viewers_public_state(ruleset: Ruleset) -> None:
    """``me`` finds the viewer's own seat, and a foreign viewer is an error."""
    state = _dealt(ruleset)
    view = ruleset.observe(state, PlayerId(2))
    assert view.me.player == 2
    assert view.me.hand_count == len(view.hand)

    stranger = build_view(state, PlayerId(0))
    impostor = replace(stranger, viewer=PlayerId(9))
    with pytest.raises(StateInvariantError, match="no seat"):
        _ = impostor.me


def test_observing_an_unseated_player_is_an_invariant_error(ruleset: Ruleset) -> None:
    """Observation is only defined for seats that exist."""
    state = _dealt(ruleset)
    with pytest.raises(StateInvariantError, match="no seat"):
        ruleset.observe(state, PlayerId(7))


def test_observation_rejects_a_state_that_still_owes_a_refill(
    ruleset: Ruleset, picker: DeckPicker
) -> None:
    """An empty hand beside a non-empty deck is not a decision boundary."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): [], PlayerId(1): picker.take(Rank.FIVE, 2)},
        face_up={PlayerId(0): picker.take(Rank.KING, 2)},
        draw_count=4,
    )
    with pytest.raises(StateInvariantError, match="refill"):
        ruleset.observe(state, PlayerId(1))


def test_hidden_assignments_cannot_change_what_a_viewer_sees(
    ruleset: Ruleset, picker: DeckPicker
) -> None:
    """Swapping face-down identities leaves views and legal moves identical."""
    hand = picker.take(Rank.SIX, 2)
    blind = picker.many([Rank.THREE, Rank.ACE, Rank.JOKER])
    face_down = {PlayerId(0): {SlotId(index): card for index, card in enumerate(blind)}}
    original = build_play_state(
        picker,
        hands={PlayerId(0): hand, PlayerId(1): []},
        face_down=face_down,
        constraint=AtLeast(Rank.FOUR),
    )
    swapped = deepcopy(original)
    slots = swapped.players[PlayerId(0)].face_down
    slots[SlotId(0)], slots[SlotId(2)] = slots[SlotId(2)], slots[SlotId(0)]

    assert original.players[PlayerId(0)].face_down != slots
    assert ruleset.observe(original, PlayerId(0)) == ruleset.observe(swapped, PlayerId(0))
    assert ruleset.observe(original, PlayerId(1)) == ruleset.observe(swapped, PlayerId(1))
    assert original.get_legal_moves() == swapped.get_legal_moves()


def test_history_is_passed_through_untouched(ruleset: Ruleset) -> None:
    """The view carries exactly the filtered history the caller supplies.

    ``observe`` deliberately does not filter: the caller owns history, so it
    must hand over events already filtered for this viewer. Passing raw engine
    events here would put opponents' hand identities into the observation.
    """
    state = _dealt(ruleset)
    filtered = filter_events_for(ruleset.initial_events(state), PlayerId(0))
    view = ruleset.observe(state, PlayerId(0), history=filtered)
    assert view.history == filtered
    assert isinstance(view.history, tuple)
    assert ruleset.observe(state, PlayerId(0)).history == ()


def test_view_reports_the_active_zone_through_legal_moves(
    ruleset: Ruleset, picker: DeckPicker
) -> None:
    """A view is enough to derive the actor's zone without the hidden state."""
    state = build_play_state(
        picker,
        hands={PlayerId(0): picker.take(Rank.FOUR, 1), PlayerId(1): []},
    )
    view = ruleset.observe(state, PlayerId(0))
    assert {move.source for move in plays_in(view.get_legal_moves())} == {Zone.HAND}
