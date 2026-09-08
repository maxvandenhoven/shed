"""Seeded invariant checks over many deals and every setup decision.

Play resolution does not exist yet, so these sweeps cover the deal and the whole
SETUP phase: card conservation, a legal move at every decision boundary, and
LIFO undo back to the exact opening position.
"""

import random

import pytest

from shed.engine import (
    Arrange,
    GameState,
    Phase,
    PlayerId,
    Ruleset,
    Transition,
    Unrestricted,
    validate_decision_boundary,
)

SEEDS = range(12)


def _play_out_setup(ruleset: Ruleset, state: GameState, rng: random.Random) -> list[Transition]:
    """Submit a random legal arrangement for every seat, checking invariants.

    Args:
        ruleset: Ruleset applying the moves.
        state: A SETUP state, mutated until it enters PLAY.
        rng: Generator choosing among the legal arrangements.

    Returns:
        The transitions in submission order, ready for LIFO undo.
    """
    transitions: list[Transition] = []
    while state.phase is Phase.SETUP:
        moves = state.get_legal_moves()
        assert len(moves) == 20
        assert all(isinstance(move, Arrange) for move in moves)
        transitions.append(ruleset.apply_move(state, rng.choice(moves)))
        validate_decision_boundary(state)
    return transitions


@pytest.mark.parametrize("player_count", [2, 3, 4, 5])
def test_seeded_setups_conserve_cards_and_reach_a_legal_opening(player_count: int) -> None:
    """Every seeded deal arranges cleanly into a playable opening position."""
    ruleset = Ruleset()
    rng = random.Random(player_count)
    for seed in SEEDS:
        dealer = PlayerId(seed % player_count)
        state = ruleset.create_initial_state(player_count, seed=seed, dealer=dealer)
        _play_out_setup(ruleset, state, rng)

        assert state.phase is Phase.PLAY
        assert state.setup is None
        assert state.current_ply == 0
        assert isinstance(state.constraint, Unrestricted)
        assert state.discard_pile == []
        validate_decision_boundary(state)

        for seat in state.seat_order:
            player = state.players[seat]
            assert len(player.hand) == 3
            assert len(player.face_up) == 3
            assert len(player.face_down) == 3
        assert state.get_legal_moves()  # The opener always has a decision.


@pytest.mark.parametrize("player_count", [2, 3, 4, 5])
def test_lifo_undo_returns_to_the_exact_dealt_position(player_count: int) -> None:
    """Undoing every setup decision restores the freshly dealt state."""
    ruleset = Ruleset()
    rng = random.Random(100 + player_count)
    for seed in SEEDS:
        state = ruleset.create_initial_state(player_count, seed=seed)
        dealt = ruleset.create_initial_state(player_count, seed=seed)
        transitions = _play_out_setup(ruleset, state, rng)

        for transition in reversed(transitions):
            ruleset.undo_move(state, transition)
        assert state == dealt
        validate_decision_boundary(state)


@pytest.mark.parametrize("player_count", [2, 3, 4, 5])
def test_each_seat_observes_only_its_own_hand_throughout_setup(player_count: int) -> None:
    """No observation during setup ever names a card another player holds."""
    ruleset = Ruleset()
    rng = random.Random(200 + player_count)
    state = ruleset.create_initial_state(player_count, seed=7)

    while True:
        for seat in state.seat_order:
            view = ruleset.observe(state, seat)
            visible = {card.id for card in view.hand}
            visible |= {card.id for public in view.players for card in public.face_up}
            for other in state.seat_order:
                if other == seat:
                    continue
                private = {card.id for card in state.players[other].hand}
                private |= {card.id for card in state.players[other].face_down.values()}
                assert not visible & private
        if state.phase is not Phase.SETUP:
            break
        ruleset.apply_move(state, rng.choice(state.get_legal_moves()))
