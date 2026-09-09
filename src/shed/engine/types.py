"""Immutable supporting types shared by every other Shed engine module.

This module owns the vocabulary of the fixed ``shed-v1`` profile: identifier
aliases, physical cards, enums, play constraints, the canonical move union, the
frozen rules configuration, and the engine's two error types. Everything here is
immutable, hashable, and holds no game state, so any other module may import it
without creating a cycle.

Every constructor here takes correctly typed domain objects and assumes its
annotations hold: ``Play`` takes a ``Rank``, never an integer it converts.
``__post_init__`` checks domain invariants only -- non-negative identifiers,
positive counts, playable sources, distinct arrangement IDs, joker/suit
consistency -- and never coerces or type-checks its inputs. ``ty`` enforces the
annotations for engine callers; untyped external data belongs to the future
replay and transport layers, which validate it where it is decoded and construct
these objects before calling the engine.

Moves still canonicalize their *values*: an arrangement sorts its identifiers, so
two submissions naming the same physical cards compare equal. That is semantic
canonicalization, not input conversion, and it is what lets the ruleset validate
a move by membership in the generated legal-move tuple.
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
    "Outcome",
    "Phase",
    "PickUp",
    "Play",
    "PlayConstraint",
    "PlayerId",
    "PublicPlayerState",
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


def _reject_joker_bound(rank: Rank, name: str) -> None:
    """Refuse a joker as a rank comparison bound.

    A joker resets the pile to :class:`Unrestricted`; its enum value must never
    act as a threshold. The rule is shared by both bounded constraints, so it is
    named once here.

    Args:
        rank: Bound the constraint was built with.
        name: Field name used in the error message.

    Raises:
        ValueError: If the bound is ``Rank.JOKER``.
    """
    if rank is Rank.JOKER:
        raise ValueError(f"{name} cannot be a joker; a joker resets to Unrestricted")


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
        """Enforce a usable identifier and joker/suit consistency.

        Raises:
            ValueError: If the identifier is negative, a joker carries a suit,
                or an ordinary card lacks one.
        """
        if self.id < 0:
            raise ValueError(f"Card.id must be non-negative, got {self.id}")
        if self.rank is Rank.JOKER and self.suit is not None:
            raise ValueError("A joker has no suit")
        if self.rank is not Rank.JOKER and self.suit is None:
            raise ValueError("An ordinary card needs a suit")


@dataclass(frozen=True, slots=True)
class Outcome:
    """Result of a finished game.

    Attributes:
        winner: The first player to empty every personal zone.
    """

    winner: PlayerId


@dataclass(frozen=True, slots=True)
class PublicPlayerState:
    """What everybody knows about one player.

    Shared by observations and by the events that announce a deal, so it lives
    with the other value types rather than in either consumer.

    Attributes:
        player: The described seat.
        hand_count: Number of cards in hand; identities stay private.
        face_up: Public face-up cards, sorted by card ID.
        face_down_slots: Remaining face-down slot IDs ascending; identities are
            never included.
    """

    player: PlayerId
    hand_count: int
    face_up: tuple[Card, ...]
    face_down_slots: tuple[SlotId, ...]


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
        """Reject a joker as a comparison bound.

        Raises:
            ValueError: If the bound is ``Rank.JOKER``, whose enum value must
                never determine game strength.
        """
        _reject_joker_bound(self.rank, "AtLeast.rank")


@dataclass(frozen=True, slots=True)
class AtMost:
    """Constraint accepting ordinary ranks less than or equal to ``rank``.

    Attributes:
        rank: Maximum acceptable ordinary rank; only a seven sets this profile's
            ``AtMost`` constraint.
    """

    rank: Rank

    def __post_init__(self) -> None:
        """Reject a joker as a comparison bound.

        Raises:
            ValueError: If the bound is ``Rank.JOKER``.
        """
        _reject_joker_bound(self.rank, "AtMost.rank")


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
        """Check the identifiers name three distinct cards, then sort them.

        Raises:
            ValueError: If an identifier is negative or the three are not
                distinct. The arity itself is carried by the annotation.
        """
        ids = self.face_up_cards
        if any(card_id < 0 for card_id in ids):
            raise ValueError(f"Arrange requires non-negative card IDs, got {ids}")
        if len(set(ids)) != 3:
            raise ValueError(f"Arrange requires three distinct card IDs, got {ids}")
        object.__setattr__(self, "face_up_cards", tuple(sorted(ids)))


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
        """Check the source zone is playable and the batch is not empty.

        Raises:
            ValueError: If the source is the face-down zone or the count is not
                positive.
        """
        if self.source not in (Zone.HAND, Zone.FACE_UP):
            raise ValueError(f"Play.source must be HAND or FACE_UP, got {self.source}")
        if self.count < 1:
            raise ValueError(f"Play.count must be positive, got {self.count}")


@dataclass(frozen=True, slots=True)
class Reveal:
    """Blind decision turning over one remaining face-down slot.

    Attributes:
        slot: Identifier of the slot to reveal; slot identifiers stay stable
            when other slots are emptied.
    """

    slot: SlotId

    def __post_init__(self) -> None:
        """Check the slot identifier is usable.

        Raises:
            ValueError: If the slot is negative.
        """
        if self.slot < 0:
            raise ValueError(f"Reveal.slot must be non-negative, got {self.slot}")


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

    Like every type here, the profile assumes its annotations: the fields are
    integers, and a decoder is responsible for rejecting external data that only
    looks like one (``3.0`` is not ``initial_hand_size``). Validation covers the
    domain question instead -- whether a correctly typed profile is the one this
    release implements.

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

        Every public entry point that consumes a profile calls this, so a
        correctly typed but unsupported configuration is refused rather than
        quietly producing a non-standard game.

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
        config: Rules profile supplying the joker count. It is validated here,
            so an unsupported profile cannot produce a non-standard deck through
            this entry point or through :func:`shed.engine.state.shuffled_deck`.

    Returns:
        The 54 canonical cards in identifier order.

    Raises:
        ValueError: If the profile is not the fixed ``shed-v1`` profile.
    """
    config.validate()
    cards: list[Card] = []
    for suit in SUIT_ORDER:
        for rank in ORDINARY_RANKS:
            cards.append(Card(id=CardId(len(cards)), rank=rank, suit=suit))
    for _ in range(config.joker_count):
        cards.append(Card(id=CardId(len(cards)), rank=Rank.JOKER, suit=None))
    return tuple(cards)
