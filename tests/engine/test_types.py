"""Tests for cards, moves, constraints, and the fixed rules profile."""

from collections import Counter

import pytest

from shed.engine import (
    ORDINARY_RANKS,
    SUIT_ORDER,
    Arrange,
    AtLeast,
    AtMost,
    Card,
    CardId,
    Phase,
    PickUp,
    Play,
    Rank,
    Reveal,
    RulesConfig,
    Ruleset,
    SlotId,
    Suit,
    Zone,
    build_deck,
)


def test_canonical_deck_has_54_uniquely_identified_cards() -> None:
    """The deck holds 54 cards with contiguous identifiers and no duplicates."""
    deck = build_deck()
    assert len(deck) == 54
    assert [card.id for card in deck] == list(range(54))
    assert len({card.id for card in deck}) == 54
    assert len({(card.rank, card.suit, card.id) for card in deck}) == 54


def test_canonical_deck_is_built_in_the_specified_order() -> None:
    """Suits run clubs to spades with ranks ascending, then the two jokers."""
    deck = build_deck()
    expected = [(rank, suit) for suit in SUIT_ORDER for rank in ORDINARY_RANKS]
    assert [(card.rank, card.suit) for card in deck[:52]] == expected
    assert deck[0] == Card(id=CardId(0), rank=Rank.TWO, suit=Suit.CLUBS)
    assert deck[51] == Card(id=CardId(51), rank=Rank.ACE, suit=Suit.SPADES)
    assert [card.suit for card in deck[52:]] == [None, None]
    assert {card.rank for card in deck[52:]} == {Rank.JOKER}


def test_canonical_deck_has_four_of_each_ordinary_rank_and_two_jokers() -> None:
    """Rank multiplicities match a standard deck plus two jokers."""
    counts = Counter(card.rank for card in build_deck())
    assert all(counts[rank] == 4 for rank in ORDINARY_RANKS)
    assert counts[Rank.JOKER] == 2


def test_jokers_and_ordinary_cards_must_agree_on_suits() -> None:
    """A joker carries no suit and an ordinary card must carry one."""
    with pytest.raises(ValueError, match="joker has no suit"):
        Card(id=CardId(0), rank=Rank.JOKER, suit=Suit.CLUBS)
    with pytest.raises(ValueError, match="needs a suit"):
        Card(id=CardId(0), rank=Rank.ACE, suit=None)


@pytest.mark.parametrize("bad_id", [True, False, -1, "3", 1.0])
def test_card_identifiers_must_be_non_negative_integers(bad_id: object) -> None:
    """Booleans, floats, strings, and negatives are rejected as card IDs."""
    with pytest.raises(ValueError, match="Card.id"):
        Card(id=bad_id, rank=Rank.ACE, suit=Suit.CLUBS)  # ty: ignore[invalid-argument-type]


def test_cards_are_immutable_and_hashable() -> None:
    """Cards are frozen values usable in sets and dictionary keys."""
    card = build_deck()[0]
    with pytest.raises(AttributeError):
        card.rank = Rank.ACE  # ty: ignore[invalid-assignment]
    assert {card, build_deck()[0]} == {card}


def test_arrange_canonicalizes_identifier_order() -> None:
    """Equivalent arrangements compare and hash equal regardless of order."""
    ordered = Arrange((CardId(2), CardId(9), CardId(4)))
    assert ordered.face_up_cards == (2, 4, 9)
    assert ordered == Arrange((CardId(9), CardId(4), CardId(2)))
    assert len({ordered, Arrange((CardId(4), CardId(9), CardId(2)))}) == 1


@pytest.mark.parametrize(
    ("card_ids", "message"),
    [
        ((1, 2), "exactly three"),
        ((1, 2, 3, 4), "exactly three"),
        ((1, 2, 2), "distinct"),
        ((1, 2, True), "must be an int"),
        ((1, 2, -3), "non-negative"),
    ],
)
def test_arrange_rejects_malformed_submissions(card_ids: tuple[int, ...], message: str) -> None:
    """Wrong counts, duplicates, booleans, and negatives are all rejected."""
    with pytest.raises(ValueError, match=message):
        Arrange(card_ids)  # ty: ignore[invalid-argument-type]


def test_play_accepts_only_hand_and_face_up_sources() -> None:
    """A play never names the face-down zone; blind cards use ``Reveal``."""
    assert Play(Zone.HAND, Rank.FIVE, 2).source is Zone.HAND
    assert Play(Zone.FACE_UP, Rank.FIVE, 1).source is Zone.FACE_UP
    with pytest.raises(ValueError, match="HAND or FACE_UP"):
        Play(Zone.FACE_DOWN, Rank.FIVE, 1)


@pytest.mark.parametrize("bad_count", [0, -1, True, 1.5, "1"])
def test_play_requires_a_positive_integer_count(bad_count: object) -> None:
    """Counts must be real positive integers, never booleans or floats."""
    with pytest.raises(ValueError, match="Play.count"):
        Play(Zone.HAND, Rank.FIVE, bad_count)  # ty: ignore[invalid-argument-type]


def test_play_canonicalizes_a_decoded_integer_rank() -> None:
    """A rank decoded as a plain integer becomes the matching enum member."""
    decoded = Play(Zone.HAND, 7, 2)  # ty: ignore[invalid-argument-type]
    assert decoded.rank is Rank.SEVEN
    assert decoded == Play(Zone.HAND, Rank.SEVEN, 2)
    with pytest.raises(ValueError):
        Play(Zone.HAND, 99, 1)  # ty: ignore[invalid-argument-type]


def test_reveal_requires_a_non_negative_integer_slot() -> None:
    """Slot identifiers are validated the same way card identifiers are."""
    assert Reveal(SlotId(2)).slot == 2
    with pytest.raises(ValueError, match="Reveal.slot"):
        Reveal(True)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="non-negative"):
        Reveal(SlotId(-1))


def test_moves_are_immutable_and_hashable() -> None:
    """Every move type is a frozen, hashable value object."""
    moves = [
        Arrange((CardId(0), CardId(1), CardId(2))),
        Play(Zone.HAND, Rank.TWO, 1),
        Reveal(SlotId(0)),
        PickUp(),
    ]
    assert len(set(moves)) == 4
    with pytest.raises(AttributeError):
        moves[1].count = 2  # ty: ignore[invalid-assignment]
    assert PickUp() == PickUp()


def test_constraints_reject_a_joker_as_a_rank_bound() -> None:
    """A joker's enum value must never act as a comparison bound."""
    for constraint in (AtLeast, AtMost):
        with pytest.raises(ValueError, match="never sets a rank bound"):
            constraint(Rank.JOKER)


def test_default_rules_profile_validates() -> None:
    """The unmodified profile is accepted and describes a 54-card deck."""
    config = RulesConfig()
    config.validate()
    assert config.id == "shed-v1"
    assert config.deck_size == 54


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": "shed-v2"}, "Unknown rules profile"),
        ({"max_players": 6}, "max_players"),
        ({"refill_target": 4}, "refill_target"),
        ({"joker_count": 0, "initial_hand_size": 2}, "initial_hand_size, joker_count"),
    ],
)
def test_modified_profiles_are_rejected(changes: dict[str, object], message: str) -> None:
    """A changed profile must never silently claim to be ``shed-v1``."""
    config = RulesConfig(**changes)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match=message):
        config.validate()
    with pytest.raises(ValueError, match=message):
        Ruleset(config)


def test_phases_and_zones_are_distinct_values() -> None:
    """The stored enums cover exactly the documented phases and zones."""
    assert {phase.value for phase in Phase} == {"setup", "play", "finished"}
    assert {zone.value for zone in Zone} == {"hand", "face_up", "face_down"}
