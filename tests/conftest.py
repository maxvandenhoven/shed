"""Small deterministic fixtures shared by the engine tests.

The helpers here build crafted positions from the canonical deck. Every card a
test uses is taken from one :class:`DeckPicker`, and whatever is left over is
placed in the draw pile or the burned collection, so a crafted state always
conserves all 54 physical cards and passes the decision-boundary invariants.
"""

from collections.abc import Iterable, Sequence

import pytest

from shed.engine import (
    DEFAULT_RULES,
    Arrange,
    Card,
    CardId,
    GameState,
    Move,
    Phase,
    Play,
    PlayConstraint,
    PlayerId,
    PlayerState,
    Rank,
    Ruleset,
    SetupState,
    SlotId,
    Transition,
    Unrestricted,
    build_deck,
    dealing_order,
    validate_decision_boundary,
)

FIRST_SEAT: PlayerId = PlayerId(0)
"""Seat zero, the default actor and dealer for crafted states."""

UNRESTRICTED: PlayConstraint = Unrestricted()
"""Shared empty-pile constraint; frozen, so one instance is safe to reuse."""


class DeckPicker:
    """Hands out distinct canonical cards so crafted states stay consistent.

    Attributes:
        _available: Canonical cards not yet handed to a test, in deck order.
    """

    def __init__(self) -> None:
        """Start with the full canonical deck available."""
        self._available: list[Card] = list(build_deck())

    def take(self, rank: Rank, count: int = 1) -> list[Card]:
        """Take distinct unused cards of one rank.

        Args:
            rank: Rank to take.
            count: How many cards of that rank are needed.

        Returns:
            The taken cards, in canonical deck order.

        Raises:
            AssertionError: If the deck cannot supply that many.
        """
        taken = [card for card in self._available if card.rank is rank][:count]
        assert len(taken) == count, f"Only {len(taken)} unused {rank.name} cards remain"
        for card in taken:
            self._available.remove(card)
        return taken

    def one(self, rank: Rank) -> Card:
        """Take a single unused card of the given rank.

        Args:
            rank: Rank to take.

        Returns:
            One card of that rank.
        """
        return self.take(rank, 1)[0]

    def many(self, ranks: Iterable[Rank]) -> list[Card]:
        """Take one unused card for each rank in order.

        Args:
            ranks: Ranks to take, one card each.

        Returns:
            The taken cards in the requested order.
        """
        return [self.one(rank) for rank in ranks]

    def any_cards(self, count: int) -> list[Card]:
        """Take unused cards without caring about their ranks.

        Useful for filler zones a test does not reason about, such as face-up
        cards in a test that only inspects hands.

        Args:
            count: How many cards are needed.

        Returns:
            The taken cards, in canonical deck order.

        Raises:
            AssertionError: If too few cards remain.
        """
        assert len(self._available) >= count, "The deck cannot supply that many cards"
        taken = self._available[:count]
        for card in taken:
            self._available.remove(card)
        return taken

    def rest(self) -> list[Card]:
        """Take every card that is still unused.

        Returns:
            The remaining cards, leaving the picker empty.
        """
        remaining = list(self._available)
        self._available.clear()
        return remaining


def _player_states(
    seats: Sequence[PlayerId],
    hands: dict[PlayerId, list[Card]],
    face_up: dict[PlayerId, list[Card]] | None,
    face_down: dict[PlayerId, dict[SlotId, Card]] | None,
) -> dict[PlayerId, PlayerState]:
    """Assemble per-player card ownership for a crafted state.

    Args:
        seats: Seats to build states for.
        hands: Hand cards per seat.
        face_up: Face-up cards per seat; missing seats get none.
        face_down: Face-down cards keyed by slot per seat; missing seats get
            none.

    Returns:
        A player-state mapping suitable for :class:`GameState`.
    """
    return {
        seat: PlayerState(
            hand=list(hands.get(seat, [])),
            face_up=list((face_up or {}).get(seat, [])),
            face_down=dict((face_down or {}).get(seat, {})),
        )
        for seat in seats
    }


def build_play_state(
    picker: DeckPicker,
    *,
    hands: dict[PlayerId, list[Card]],
    face_up: dict[PlayerId, list[Card]] | None = None,
    face_down: dict[PlayerId, dict[SlotId, Card]] | None = None,
    discard: Sequence[Card] = (),
    constraint: PlayConstraint = UNRESTRICTED,
    draw_count: int = 0,
    current_player: PlayerId = FIRST_SEAT,
    dealer: PlayerId = FIRST_SEAT,
    validate: bool = True,
) -> GameState:
    """Craft a valid PLAY state for testing legality before play resolution.

    Cards not handed out by the test fill the draw pile up to ``draw_count`` and
    then the burned collection, so card accounting always holds. When a
    constraint is given without a discard pile, one matching card is placed on
    the pile: an empty pile must always be unrestricted.

    Args:
        picker: Source of every card used, including the filler cards.
        hands: Hand cards per seat; its keys define the seats.
        face_up: Face-up cards per seat.
        face_down: Face-down cards keyed by slot per seat.
        discard: The live pile, oldest first.
        constraint: Restriction the next ordinary rank must satisfy.
        draw_count: Cards to leave in the draw pile.
        current_player: The actor to schedule.
        dealer: Dealing seat.
        validate: Whether to assert the decision-boundary invariants. Pass
            ``False`` only to build a deliberately invalid position for a test
            that checks how the engine rejects it.

    Returns:
        A PLAY state, by default one that satisfies the decision-boundary
        invariants.
    """
    pile = list(discard)
    if not pile and not isinstance(constraint, Unrestricted):
        pile = [picker.one(constraint.rank)]
    seats = tuple(PlayerId(seat) for seat in sorted(hands))
    leftover = picker.rest()
    state = GameState(
        rules=DEFAULT_RULES,
        players=_player_states(seats, hands, face_up, face_down),
        seat_order=seats,
        dealer=dealer,
        phase=Phase.PLAY,
        current_player=current_player,
        current_ply=0,
        draw_pile=leftover[:draw_count],
        discard_pile=pile,
        burned_cards=leftover[draw_count:],
        constraint=constraint,
        setup=None,
        outcome=None,
    )
    if validate:
        validate_decision_boundary(state)
    return state


def build_setup_state(
    picker: DeckPicker,
    *,
    hands: dict[PlayerId, list[Card]],
    face_up: dict[PlayerId, list[Card]],
    face_down: dict[PlayerId, dict[SlotId, Card]] | None = None,
    dealer: PlayerId = FIRST_SEAT,
) -> GameState:
    """Craft a SETUP state with chosen hand and face-up cards.

    Leftover cards become the draw pile, matching a real deal.

    Args:
        picker: Source of every card used.
        hands: Hand cards per seat; its keys define the seats.
        face_up: Face-up cards per seat.
        face_down: Face-down cards keyed by slot per seat; defaults to none,
            which is legal because setup never inspects them.
        dealer: Dealing seat; arrangements are requested clockwise after it.

    Returns:
        A SETUP state awaiting an arrangement from every seat.
    """
    seats = tuple(PlayerId(seat) for seat in sorted(hands))
    order = dealing_order(seats, dealer)
    state = GameState(
        rules=DEFAULT_RULES,
        players=_player_states(seats, hands, face_up, face_down),
        seat_order=seats,
        dealer=dealer,
        phase=Phase.SETUP,
        current_player=order[0],
        current_ply=0,
        draw_pile=picker.rest(),
        discard_pile=[],
        burned_cards=[],
        constraint=Unrestricted(),
        setup=SetupState(pending=list(order)),
        outcome=None,
    )
    validate_decision_boundary(state)
    return state


def arrangement(card_ids: Iterable[CardId]) -> Arrange:
    """Build an arrangement move from exactly three card identifiers.

    The unpacking keeps the fixed arity explicit, which the ``Arrange``
    annotation requires and a variable-length tuple cannot express.

    Args:
        card_ids: The three chosen identifiers, in any order.

    Returns:
        The canonical arrangement, with identifiers sorted ascending.
    """
    first, second, third = sorted(card_ids)
    return Arrange((first, second, third))


def plays_in(moves: Sequence[Move]) -> tuple[Play, ...]:
    """Narrow generated moves to plays, asserting nothing else was offered.

    Args:
        moves: A generated legal-move tuple.

    Returns:
        The same moves, typed as plays.
    """
    assert all(isinstance(move, Play) for move in moves), moves
    return tuple(move for move in moves if isinstance(move, Play))


def arranges_in(moves: Sequence[Move]) -> tuple[Arrange, ...]:
    """Narrow generated moves to arrangements, asserting nothing else was offered.

    Args:
        moves: A generated legal-move tuple.

    Returns:
        The same moves, typed as arrangements.
    """
    assert all(isinstance(move, Arrange) for move in moves), moves
    return tuple(move for move in moves if isinstance(move, Arrange))


def commit_unchanged(ruleset: Ruleset, state: GameState) -> list[Transition]:
    """Have every pending player keep the arrangement they were dealt.

    Keeping the original arrangement is legal, so this is the shortest route
    from a crafted SETUP state to the opening PLAY position.

    Args:
        ruleset: Ruleset applying the moves.
        state: A SETUP state; mutated in place until it enters PLAY.

    Returns:
        The transitions in submission order, newest last, ready for LIFO undo.
    """
    transitions: list[Transition] = []
    while state.phase is Phase.SETUP:
        actor = state.current_player
        assert actor is not None
        keep = arrangement(card.id for card in state.players[actor].face_up)
        transitions.append(ruleset.apply_move(state, keep))
    return transitions


@pytest.fixture
def picker() -> DeckPicker:
    """Provide a fresh canonical-deck picker for one test."""
    return DeckPicker()


@pytest.fixture
def ruleset() -> Ruleset:
    """Provide a ruleset running the fixed ``shed-v1`` profile."""
    return Ruleset()
