"""Tests for the canonical deck deal: zones, order, determinism, and isolation."""

import random
from copy import deepcopy

import pytest

from shed.engine import (
    DEFAULT_RULES,
    Card,
    GameState,
    Phase,
    PlayerId,
    Ruleset,
    SlotId,
    Unrestricted,
    build_deck,
    deal_initial_state,
    dealing_order,
    shuffled_deck,
    validate_decision_boundary,
)


def _all_cards(state: GameState) -> list[Card]:
    """Collect every physical card in a state, whichever zone holds it.

    Args:
        state: State to walk.

    Returns:
        All cards in deck, pile, burned, and personal zones.
    """
    cards = [*state.draw_pile, *state.discard_pile, *state.burned_cards]
    for player in state.players.values():
        cards.extend([*player.hand, *player.face_up, *player.face_down.values()])
    return cards


@pytest.mark.parametrize("player_count", [2, 3, 4, 5])
def test_deal_gives_every_supported_table_the_right_zones(
    ruleset: Ruleset, player_count: int
) -> None:
    """Each seat gets three hand, three face-up, and three face-down cards."""
    state = ruleset.create_initial_state(player_count, seed=11)
    assert state.phase is Phase.SETUP
    assert state.seat_order == tuple(range(player_count))
    for seat in state.seat_order:
        player = state.players[seat]
        assert len(player.hand) == DEFAULT_RULES.initial_hand_size
        assert len(player.face_up) == DEFAULT_RULES.initial_face_up_count
        assert sorted(player.face_down) == [0, 1, 2]
        assert player.remaining_count == 9
    assert len(state.draw_pile) == 54 - 9 * player_count
    assert state.discard_pile == []
    assert state.burned_cards == []
    assert isinstance(state.constraint, Unrestricted)
    assert state.outcome is None
    assert state.current_ply == 0


@pytest.mark.parametrize("player_count", [2, 3, 4, 5])
def test_every_card_is_dealt_exactly_once(ruleset: Ruleset, player_count: int) -> None:
    """All 54 identified cards occur once across every zone."""
    state = ruleset.create_initial_state(player_count, seed=3)
    cards = _all_cards(state)
    assert len(cards) == 54
    assert sorted(cards, key=lambda card: card.id) == list(build_deck())
    validate_decision_boundary(state)


def test_deal_is_deterministic_for_a_seed(ruleset: Ruleset) -> None:
    """The same seed reproduces the same deal; a different seed does not."""
    first = ruleset.create_initial_state(4, seed=2024)
    second = ruleset.create_initial_state(4, seed=2024)
    other = ruleset.create_initial_state(4, seed=2025)
    assert first == second
    assert first != other


def test_deal_follows_the_dealer_relative_round_order(ruleset: Ruleset) -> None:
    """Cards come off the deck end, clockwise after the dealer, zone by zone."""
    deck = shuffled_deck(seed=99)
    state = deal_initial_state(deck, player_count=3, dealer=PlayerId(1))
    order = dealing_order(state.seat_order, PlayerId(1))
    assert order == (2, 0, 1)

    expected = list(deck)
    for slot in range(3):
        for seat in order:
            assert state.players[seat].face_down[SlotId(slot)] == expected.pop()
    for index in range(3):
        for seat in order:
            assert state.players[seat].face_up[index] == expected.pop()
    for index in range(3):
        for seat in order:
            assert state.players[seat].hand[index] == expected.pop()
    assert state.draw_pile == expected


def test_dealer_choice_changes_who_arranges_first(ruleset: Ruleset) -> None:
    """Arrangements are requested clockwise starting after the dealer."""
    for dealer in range(4):
        state = ruleset.create_initial_state(4, seed=5, dealer=PlayerId(dealer))
        assert state.dealer == dealer
        assert state.current_player == (dealer + 1) % 4
        assert state.setup is not None
        assert state.setup.pending == [(dealer + 1 + offset) % 4 for offset in range(4)]


def test_face_down_slots_are_stable_identifiers(ruleset: Ruleset) -> None:
    """Removing one slot leaves the identifiers of the others untouched."""
    state = ruleset.create_initial_state(2, seed=8)
    player = state.players[PlayerId(0)]
    kept = {slot: card for slot, card in player.face_down.items() if slot != 1}
    player.face_down = kept
    assert sorted(player.face_down) == [0, 2]


def test_shuffle_uses_an_isolated_generator(ruleset: Ruleset) -> None:
    """Dealing never advances module-global randomness or leaks between deals."""
    random.seed(1234)
    before = random.getstate()
    first = ruleset.create_initial_state(3, seed=77)
    assert random.getstate() == before

    random.random()  # Global randomness moves on; the deck must not care.
    second = ruleset.create_initial_state(3, seed=77)
    assert first == second


def test_deal_helper_rebuilds_a_state_from_a_recorded_deck_order(ruleset: Ruleset) -> None:
    """A replay can rebuild the opening position from the deck order alone."""
    deck = shuffled_deck(seed=404)
    from_seed = ruleset.create_initial_state(5, seed=404)
    from_deck = deal_initial_state(list(deck), player_count=5)
    assert from_seed == from_deck


@pytest.mark.parametrize("player_count", [0, 1, 6])
def test_unsupported_table_sizes_are_rejected(ruleset: Ruleset, player_count: int) -> None:
    """The profile supports two through five players and nothing else."""
    with pytest.raises(ValueError, match="players"):
        ruleset.create_initial_state(player_count, seed=1)


def test_dealer_must_hold_a_seat(ruleset: Ruleset) -> None:
    """A dealer outside the table is rejected before anything is dealt."""
    with pytest.raises(ValueError, match="Dealer"):
        ruleset.create_initial_state(3, seed=1, dealer=PlayerId(3))


def test_deck_must_be_a_permutation_of_the_canonical_deck() -> None:
    """A recorded deck with missing or foreign cards cannot be dealt."""
    deck = list(build_deck())
    with pytest.raises(ValueError, match="canonical deck"):
        deal_initial_state(deck[:-1], player_count=2)
    duplicated = deepcopy(deck)
    duplicated[0] = duplicated[1]
    with pytest.raises(ValueError, match="canonical deck"):
        deal_initial_state(duplicated, player_count=2)
