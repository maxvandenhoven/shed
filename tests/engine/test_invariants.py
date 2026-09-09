"""Seeded invariant checks over many deals and complete games.

These sweeps drive whole games with :func:`tests.conftest.play_seeded_game`, the
synchronous stand-in for the match runner, and check what must hold at every
decision boundary: card conservation, a legal move for every live actor,
agreement between what the engine offers and what it accepts, a ply counter that
advances exactly once per PLAY decision, and LIFO undo back to the dealt
position. A game that hits the test-only action bound is a truncation and is
never reported as a win.
"""

import random
from copy import deepcopy

import pytest

from shed.engine import (
    Arrange,
    GameEnded,
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
    Transition,
    Unrestricted,
    Zone,
    filter_events_for,
    validate_decision_boundary,
)
from tests.conftest import assert_cards_conserved, play_seeded_game

SEEDS = range(12)

_CANDIDATE_MOVES: tuple[Move, ...] = (
    PickUp(),
    Play(Zone.FACE_UP, Rank.ACE, 1),
    Play(Zone.HAND, Rank.JOKER, 4),
    Reveal(SlotId(0)),
)
"""Moves the probe tries as an unoffered action; at least one always is one."""


def _play_out_setup(state: GameState, rng: random.Random) -> list[Transition]:
    """Submit a random legal arrangement for every seat, checking invariants.

    Args:
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
        transitions.append(state.apply_move(rng.choice(moves)))
        validate_decision_boundary(state)
    return transitions


@pytest.mark.parametrize("player_count", [2, 3, 4, 5])
def test_seeded_setups_conserve_cards_and_reach_a_legal_opening(player_count: int) -> None:
    """Every seeded deal arranges cleanly into a playable opening position."""
    rng = random.Random(player_count)
    for seed in SEEDS:
        dealer = PlayerId(seed % player_count)
        state = GameState.create(player_count, seed=seed, dealer=dealer)
        _play_out_setup(state, rng)

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
    rng = random.Random(100 + player_count)
    for seed in SEEDS:
        state = GameState.create(player_count, seed=seed)
        dealt = GameState.create(player_count, seed=seed)
        transitions = _play_out_setup(state, rng)

        for transition in reversed(transitions):
            state.undo_move(transition)
        assert state == dealt
        validate_decision_boundary(state)


@pytest.mark.parametrize("player_count", [2, 3, 4, 5])
def test_each_seat_observes_only_its_own_hand_throughout_setup(player_count: int) -> None:
    """No observation during setup ever names a card another player holds."""
    rng = random.Random(200 + player_count)
    state = GameState.create(player_count, seed=7)

    while True:
        for seat in state.seat_order:
            view = state.observe(seat)
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
        state.apply_move(rng.choice(state.get_legal_moves()))


@pytest.mark.parametrize("player_count", [2, 3, 4, 5])
def test_seeded_games_conserve_cards_and_finish_or_truncate(player_count: int) -> None:
    """Complete seeded games hold the deck together and end honestly.

    Every decision is checked inside the helper. What is asserted here is the
    shape of the ending: a finished game names a winner who holds nothing and
    schedules nobody, while a game stopped by the action bound claims no winner
    at all.
    """
    for seed in SEEDS:
        state = GameState.create(player_count, seed=seed, dealer=PlayerId(seed % player_count))
        log = play_seeded_game(state, seed=seed)

        assert_cards_conserved(state)
        if log.truncated:
            assert state.outcome is None
            assert not state.is_finished
            assert state.phase is Phase.PLAY
            continue

        assert state.is_finished
        assert state.outcome is not None
        assert state.current_player is None
        assert state.players[state.outcome.winner].remaining_count == 0
        assert isinstance(log.events[-1], GameEnded)
        validate_decision_boundary(state)


PROBE_DECISIONS = 25
"""How many decisions the exhaustive apply/undo probe covers per game.

The probe applies *every* offered move at each boundary, so its cost grows with
the branching factor. A prefix of each game is enough to exercise setup, the
opening, and ordinary play; the seeded sweeps above cover the rest.
"""


def test_every_offered_move_is_accepted_and_nothing_else_is() -> None:
    """Legality and application agree at each boundary of a seeded prefix.

    Every move the engine offers is applied and undone, which shows the two
    never disagree; a move it did not offer is refused. Undo makes the probe
    non-destructive, so the real game continues from the same position.
    """
    for seed in SEEDS:
        state = GameState.create(3, seed=seed)
        for _ in range(PROBE_DECISIONS):
            if state.is_finished:
                break
            offered = state.get_legal_moves()
            assert offered
            before = deepcopy(state)
            for move in offered:
                state.undo_move(state.apply_move(move))
                assert state == before

            unoffered = next(move for move in _CANDIDATE_MOVES if move not in offered)
            with pytest.raises(IllegalMoveError):
                state.apply_move(unoffered)
            assert state == before

            state.apply_move(offered[len(offered) // 2])


def test_lifo_undo_unwinds_a_complete_game_to_the_deal() -> None:
    """Undo covers PLAY as well as SETUP, all the way back to the deal."""
    for seed in SEEDS:
        state = GameState.create(4, seed=seed)
        dealt = GameState.create(4, seed=seed)
        log = play_seeded_game(state, seed=seed)

        for transition in reversed(log.transitions):
            state.undo_move(transition)
        assert state == dealt


def test_hidden_assignments_cannot_change_a_viewer_view_or_moves() -> None:
    """Two positions differing only in unknown cards look identical to a viewer.

    The observable information is held fixed and the hidden assignments are
    permuted: an opponent's hand is reordered, and the face-down cards are
    swapped between slots and between opponents. Neither the viewer's
    observation nor their legal moves may notice.
    """
    for seed in SEEDS:
        state = GameState.create(3, seed=seed)
        play_seeded_game(state, seed=seed, action_limit=12)
        if state.is_finished:
            continue
        viewer = state.current_player
        assert viewer is not None

        twin = deepcopy(state)
        others = [seat for seat in twin.seat_order if seat != viewer]
        for seat in others:
            twin.players[seat].hand.reverse()
        blind = [
            (seat, slot)
            for seat in twin.seat_order
            for slot in sorted(twin.players[seat].face_down)
        ]
        for (left_seat, left_slot), (right_seat, right_slot) in zip(
            blind, reversed(blind), strict=True
        ):
            left = twin.players[left_seat].face_down[left_slot]
            twin.players[left_seat].face_down[left_slot] = state.players[right_seat].face_down[
                right_slot
            ]
            assert left is not None

        history = filter_events_for(state.initial_events(), viewer)
        assert state.observe(viewer, history=history) == twin.observe(viewer, history=history)
        assert state.get_legal_moves() == twin.get_legal_moves()


def test_a_truncated_game_never_claims_a_winner() -> None:
    """The action bound is a test-only truncation, not a rules-level result."""
    state = GameState.create(4, seed=3)
    log = play_seeded_game(state, seed=3, action_limit=6)

    assert log.truncated
    assert len(log.transitions) == 6
    assert state.outcome is None
    assert not state.is_finished
    assert_cards_conserved(state)
