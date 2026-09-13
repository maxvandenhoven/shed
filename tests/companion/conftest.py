"""Small fixtures for the companion tests.

The companion tracks a game it did not deal, so a fixture here is an *observation*
rather than a position built out of physical cards: ranks for what was seen and
counts for what was not. :func:`craft` keeps that honest by deriving the burned
collection from whatever the 54 cards cannot otherwise account for, so every
crafted position satisfies the card accounting
:func:`~shed.companion.observed.validate_observed` enforces -- the same discipline
``tests/conftest.py`` applies to the engine's crafted states.
"""

from __future__ import annotations

import pytest

from shed.companion.observed import (
    ME,
    OPPONENT,
    ObservedState,
    SeatObservation,
    new_game,
    validate_observed,
)
from shed.companion.session import COMPANION_SCHEMA_VERSION, Session
from shed.engine import DEFAULT_RULES, Phase, PlayConstraint, PlayerId, Rank, Unrestricted


def craft(
    *,
    my_hand: tuple[Rank, ...] = (),
    my_face_up: tuple[Rank, ...] = (),
    my_face_down: int = 0,
    opponent_hand_known: tuple[Rank, ...] = (),
    opponent_hand_unknown: int = 0,
    opponent_face_up: tuple[Rank, ...] = (),
    opponent_face_down: int = 0,
    deck_count: int | None = 0,
    pile: tuple[Rank | None, ...] = (),
    constraint: PlayConstraint | None = None,
    to_act: PlayerId = ME,
) -> ObservedState:
    """Build a live position, filling the burned collection to conserve 54 cards.

    Args:
        my_hand: My hand ranks.
        my_face_up: My face-up ranks.
        my_face_down: My remaining face-down cards.
        opponent_hand_known: Ranks proven to be in their hand.
        opponent_hand_unknown: Their hand cards with no observed rank.
        opponent_face_up: Their face-up ranks.
        opponent_face_down: Their remaining face-down cards.
        deck_count: Cards left in the deck, or ``None`` for uncounted -- in which
            case nothing is burned, because the subtraction is unavailable.
        pile: The pile bottom to top, ``None`` for a card whose rank was never seen.
        constraint: The restriction in force; unrestricted by default.
        to_act: Who is to act.

    Returns:
        The validated position.

    Raises:
        ValueError: If the cards named already exceed the deck, which would make the
            fixture itself describe no table.
    """
    seats = (
        SeatObservation(ME, my_hand, 0, my_face_up, my_face_down),
        SeatObservation(
            OPPONENT,
            opponent_hand_known,
            opponent_hand_unknown,
            opponent_face_up,
            opponent_face_down,
        ),
    )
    burned = 0
    if deck_count is not None:
        accounted = deck_count + len(pile) + sum(seat.remaining_count for seat in seats)
        burned = DEFAULT_RULES.deck_size - accounted
        if burned < 0:
            raise ValueError(
                f"crafted position names {accounted} cards, more than {DEFAULT_RULES.deck_size}"
            )
    state = ObservedState(
        rules=DEFAULT_RULES,
        seats=seats,
        deck_count=deck_count,
        pile=pile,
        burned=(None,) * burned,
        constraint=constraint if constraint is not None else Unrestricted(),
        to_act=to_act,
        phase=Phase.PLAY,
        winner=None,
        pending=(),
    )
    validate_observed(state)
    return state


def session_for(state: ObservedState) -> Session:
    """Wrap a position as a session whose log is still empty.

    Args:
        state: The position tracking starts from.

    Returns:
        The document.
    """
    return Session(schema_version=COMPANION_SCHEMA_VERSION, initial=state, events=())


@pytest.fixture
def opening() -> ObservedState:
    """Return a fresh opening with my turn to act.

    Every rank named is distinct, so a test can refer to any of them without
    ambiguity about which physical card it meant.

    Returns:
        The opening position of a new game.
    """
    return new_game(
        my_hand=(Rank.THREE, Rank.SEVEN, Rank.TEN),
        my_face_up=(Rank.FOUR, Rank.NINE, Rank.ACE),
        opponent_face_up=(Rank.FIVE, Rank.SIX, Rank.KING),
        starting_player=ME,
    )
