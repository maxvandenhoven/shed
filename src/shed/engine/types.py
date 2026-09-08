"""Immutable supporting types shared by every other Shed engine module.

This module owns the vocabulary of the fixed ``shed-v1`` profile: identifier
aliases, physical cards, enums, play constraints, the canonical move union, the
frozen rules configuration, and the engine's two error types. Everything here is
immutable, hashable, and holds no game state, so any other module may import it
without creating a cycle.

Moves canonicalize themselves on construction. Equivalent decoded submissions
therefore compare equal, which is what lets the ruleset validate a move by
membership in the generated legal-move tuple.
"""

from dataclasses import dataclass, fields
from enum import Enum, IntEnum
from typing import NewType

__all__ = [
    "DEFAULT_DEALER",
    "DEFAULT_RULES",
    "ORDINARY_RANKS",
    "SUIT_ORDER",
    "Arrange",
    "AtLeast",
    "AtMost",
    "Card",
    "CardId",
    "IllegalMoveError",
    "Move",
    "Phase",
    "PickUp",
    "Play",
    "PlayConstraint",
    "PlayerId",
    "Rank",
    "Reveal",
    "RulesConfig",
    "SlotId",
    "StateInvariantError",
    "Suit",
    "Unrestricted",
    "Zone",
    "build_deck",
]

PlayerId = NewType("PlayerId", int)
CardId = NewType("CardId", int)
SlotId = NewType("SlotId", int)


class IllegalMoveError(ValueError):
    """Raised when a submitted move is not legal in the current state.

    Raising this error must leave the authoritative state untouched: validation
    always completes before any mutation begins.
    """


class StateInvariantError(RuntimeError):
    """Raised when authoritative state violates an engine invariant.

    This signals an engine bug or a hand-built state, never ordinary agent
    misbehaviour; illegal agent input raises :class:`IllegalMoveError` instead.
    """


class Suit(Enum):
    """Suit of an ordinary card. Suits never affect legality or strength."""

    CLUBS = "clubs"
    DIAMONDS = "diamonds"
    HEARTS = "hearts"
    SPADES = "spades"


class Rank(IntEnum):
    """Rank of a physical card, ordered by ordinary playing strength.

    ``JOKER`` carries the highest enum value purely so ranks stay totally
    ordered for deterministic iteration. Its value must never be compared
    against a constraint: jokers are handled by the always-playable exception.
    """

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    JOKER = 15  # Identity only; always handled specially.


class Zone(Enum):
    """A personal card location owned by one player."""

    HAND = "hand"
    FACE_UP = "face_up"
    FACE_DOWN = "face_down"


class Phase(Enum):
    """Stored phase of a game. Reveals, draws, burns, and pickups are atomic."""

    SETUP = "setup"
    PLAY = "play"
    FINISHED = "finished"


SUIT_ORDER: tuple[Suit, ...] = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
"""Canonical suit order used when building the unshuffled deck."""

ORDINARY_RANKS: tuple[Rank, ...] = tuple(rank for rank in Rank if rank is not Rank.JOKER)
"""Ordinary ranks ascending, two through ace; jokers are excluded."""


def _require_int(value: object, name: str) -> int:
    """Return ``value`` as an ``int``, rejecting booleans and other types.

    Decoded JSON and agent submissions can contain ``True``/``False`` where an
    integer is expected; ``bool`` is a subclass of ``int``, so it must be
    rejected explicitly rather than silently accepted as ``0``/``1``.

    Args:
        value: Candidate value taken from a move, card, or decoded payload.
        name: Field name used in the error message.

    Returns:
        The value as a plain integer.

    Raises:
        ValueError: If the value is a boolean or is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    return int(value)


def _require_index(value: object, name: str) -> int:
    """Return ``value`` as a non-negative integer index.

    Args:
        value: Candidate identifier such as a card ID or a face-down slot.
        name: Field name used in the error message.

    Returns:
        The value as a non-negative integer.

    Raises:
        ValueError: If the value is not an integer or is negative.
    """
    number = _require_int(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, got {number}")
    return number


def _require_rank(value: object, name: str) -> "Rank":
    """Return ``value`` as a :class:`Rank`, accepting its integer value.

    Args:
        value: A ``Rank`` or the integer value of one.
        name: Field name used in the error message.

    Returns:
        The corresponding ``Rank`` member.

    Raises:
        ValueError: If the value is not an integer or names no rank.
    """
    if isinstance(value, Rank):
        return value
    return Rank(_require_int(value, name))


@dataclass(frozen=True, slots=True)
class Card:
    """One of the 54 physical cards, identified for its whole lifetime.

    Attributes:
        id: Stable identifier assigned by the canonical deck before shuffling.
        rank: Playing rank; ``Rank.JOKER`` marks a joker.
        suit: Suit of an ordinary card, or ``None`` for a joker.
    """

    id: CardId
    rank: Rank
    suit: Suit | None

    def __post_init__(self) -> None:
        """Canonicalize the identifier and enforce joker/suit consistency.

        Raises:
            ValueError: If the identifier is not a non-negative integer, the
                rank is unknown, a joker carries a suit, or an ordinary card
                lacks one.
        """
        object.__setattr__(self, "id", CardId(_require_index(self.id, "Card.id")))
        object.__setattr__(self, "rank", _require_rank(self.rank, "Card.rank"))
        if self.rank is Rank.JOKER:
            if self.suit is not None:
                raise ValueError("A joker has no suit")
        elif not isinstance(self.suit, Suit):
            raise ValueError(f"An ordinary card needs a suit, got {self.suit!r}")


@dataclass(frozen=True, slots=True)
class Unrestricted:
    """Constraint accepting every ordinary rank; the state of an empty pile."""


@dataclass(frozen=True, slots=True)
class AtLeast:
    """Constraint accepting ordinary ranks greater than or equal to ``rank``.

    Attributes:
        rank: Minimum acceptable ordinary rank.
    """

    rank: Rank

    def __post_init__(self) -> None:
        """Canonicalize the rank and reject a joker as a comparison bound.

        Raises:
            ValueError: If the rank is unknown or is ``Rank.JOKER``, whose enum
                value must never determine game strength.
        """
        object.__setattr__(self, "rank", _require_rank(self.rank, "AtLeast.rank"))
        if self.rank is Rank.JOKER:
            raise ValueError("A joker never sets a rank bound; it resets to Unrestricted")


@dataclass(frozen=True, slots=True)
class AtMost:
    """Constraint accepting ordinary ranks less than or equal to ``rank``.

    Attributes:
        rank: Maximum acceptable ordinary rank; only a seven sets this profile's
            ``AtMost`` constraint.
    """

    rank: Rank

    def __post_init__(self) -> None:
        """Canonicalize the rank and reject a joker as a comparison bound.

        Raises:
            ValueError: If the rank is unknown or is ``Rank.JOKER``.
        """
        object.__setattr__(self, "rank", _require_rank(self.rank, "AtMost.rank"))
        if self.rank is Rank.JOKER:
            raise ValueError("A joker never sets a rank bound; it resets to Unrestricted")


type PlayConstraint = Unrestricted | AtLeast | AtMost


@dataclass(frozen=True, slots=True)
class Arrange:
    """Setup decision choosing which three cards become the final face-up set.

    The three identifiers are normalized ascending, so two submissions naming
    the same physical cards in different orders compare equal.

    Attributes:
        face_up_cards: Ascending identifiers of the three chosen cards.
    """

    face_up_cards: tuple[CardId, CardId, CardId]

    def __post_init__(self) -> None:
        """Validate and canonicalize the chosen identifiers.

        Raises:
            ValueError: If the submission does not contain exactly three
                distinct non-negative integer identifiers.
        """
        try:
            given = tuple(self.face_up_cards)
        except TypeError as error:
            raise ValueError("Arrange.face_up_cards must be a sequence of three IDs") from error
        if len(given) != 3:
            raise ValueError(f"Arrange requires exactly three card IDs, got {len(given)}")
        ids = tuple(_require_index(value, "Arrange.face_up_cards") for value in given)
        if len(set(ids)) != 3:
            raise ValueError(f"Arrange requires three distinct card IDs, got {ids}")
        object.__setattr__(self, "face_up_cards", tuple(sorted(CardId(value) for value in ids)))


@dataclass(frozen=True, slots=True)
class Play:
    """Play decision naming a source zone, a rank, and a batch size.

    The engine selects the physical cards by ascending card ID, so suits never
    multiply the action space.

    Attributes:
        source: Zone the batch comes from; hand or face-up only.
        rank: Rank shared by every card in the batch.
        count: Number of cards in the batch; at least one.
    """

    source: Zone
    rank: Rank
    count: int

    def __post_init__(self) -> None:
        """Validate the source zone, canonicalize the rank, and check the count.

        Raises:
            ValueError: If the source is not a playable zone, the rank is
                unknown, or the count is not a positive integer.
        """
        if self.source not in (Zone.HAND, Zone.FACE_UP):
            raise ValueError(f"Play.source must be HAND or FACE_UP, got {self.source!r}")
        object.__setattr__(self, "rank", _require_rank(self.rank, "Play.rank"))
        count = _require_int(self.count, "Play.count")
        if count < 1:
            raise ValueError(f"Play.count must be positive, got {count}")
        object.__setattr__(self, "count", count)


@dataclass(frozen=True, slots=True)
class Reveal:
    """Blind decision turning over one remaining face-down slot.

    Attributes:
        slot: Identifier of the slot to reveal; slot identifiers stay stable
            when other slots are emptied.
    """

    slot: SlotId

    def __post_init__(self) -> None:
        """Canonicalize the slot identifier.

        Raises:
            ValueError: If the slot is not a non-negative integer.
        """
        object.__setattr__(self, "slot", SlotId(_require_index(self.slot, "Reveal.slot")))


@dataclass(frozen=True, slots=True)
class PickUp:
    """Forced decision taking the whole discard pile into hand.

    Voluntary pickup is not legal in ``shed-v1``: this move is generated only
    when no playable batch exists.
    """


type Move = Arrange | Play | Reveal | PickUp


@dataclass(frozen=True, slots=True)
class RulesConfig:
    """The fixed ``shed-v1`` rules profile.

    The first release supports exactly one profile. The fields document its
    numbers rather than offering configuration: :meth:`validate` rejects any
    changed value so a modified profile can never claim to be ``shed-v1``.

    Attributes:
        id: Profile identifier recorded in replays.
        min_players: Smallest supported table size.
        max_players: Largest supported table size.
        joker_count: Jokers added to the 52 ordinary cards.
        initial_hand_size: Cards dealt to each hand.
        initial_face_up_count: Cards dealt face up to each player.
        initial_face_down_count: Face-down slots dealt to each player.
        refill_target: Hand size a player is replenished to while the deck
            lasts.
    """

    id: str = "shed-v1"
    min_players: int = 2
    max_players: int = 5
    joker_count: int = 2
    initial_hand_size: int = 3
    initial_face_up_count: int = 3
    initial_face_down_count: int = 3
    refill_target: int = 3

    @property
    def deck_size(self) -> int:
        """Number of physical cards in the deck, including jokers."""
        return 52 + self.joker_count

    def validate(self) -> None:
        """Check that this configuration is the unmodified ``shed-v1`` profile.

        Raises:
            ValueError: If the identifier is unknown or any field differs from
                the fixed profile. Add a new profile alongside explicit rules
                and tests instead of tuning these values.
        """
        reference = RulesConfig()
        if self.id != reference.id:
            raise ValueError(f"Unknown rules profile {self.id!r}; only {reference.id!r} exists")
        changed = sorted(
            item.name
            for item in fields(self)
            if getattr(self, item.name) != getattr(reference, item.name)
        )
        if changed:
            raise ValueError(f"Profile {self.id!r} is fixed; changed fields: {', '.join(changed)}")


DEFAULT_DEALER: PlayerId = PlayerId(0)
"""Default dealing seat.

A module-level constant rather than a ``PlayerId(0)`` call in a default
argument, so signatures stay free of calls while the documented default value is
unchanged.
"""

DEFAULT_RULES = RulesConfig()
"""The single supported profile, used as the default wherever config is taken.

A module constant rather than a call in a default argument: the profile is
frozen, so one shared instance is safe and keeps signatures side-effect free.
"""


def build_deck(config: RulesConfig = DEFAULT_RULES) -> tuple[Card, ...]:
    """Build the canonical unshuffled deck with stable identifiers.

    Ordinary cards come first, suit by suit in clubs, diamonds, hearts, spades
    order, with ranks ascending two through ace inside each suit; the jokers
    follow. Identifiers are assigned ``0..deck_size-1`` in that order and never
    change, which is what makes recorded deals reproducible.

    Args:
        config: Rules profile supplying the joker count.

    Returns:
        The 54 canonical cards in identifier order.
    """
    cards: list[Card] = []
    for suit in SUIT_ORDER:
        for rank in ORDINARY_RANKS:
            cards.append(Card(id=CardId(len(cards)), rank=rank, suit=suit))
    for _ in range(config.joker_count):
        cards.append(Card(id=CardId(len(cards)), rank=Rank.JOKER, suit=None))
    return tuple(cards)
