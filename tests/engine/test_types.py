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
    GameState,
    Move,
    Phase,
    PickUp,
    Play,
    Rank,
    Reveal,
    RulesConfig,
    SlotId,
    Suit,
    Zone,
    build_deck,
    shuffled_deck,
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


def test_card_identifiers_must_be_non_negative() -> None:
    """A negative identifier names no physical card.

    Non-integer input is a decoding concern, not an engine one: the annotation
    declares ``CardId`` and the future decoder enforces it.
    """
    with pytest.raises(ValueError, match="Card.id must be non-negative"):
        Card(id=CardId(-1), rank=Rank.ACE, suit=Suit.CLUBS)


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
        ((CardId(1), CardId(2), CardId(2)), "distinct"),
        ((CardId(1), CardId(2), CardId(-3)), "non-negative"),
    ],
)
def test_arrange_rejects_invalid_identifier_sets(
    card_ids: tuple[CardId, CardId, CardId], message: str
) -> None:
    """Duplicate and negative identifiers name no valid three-card choice.

    The arity is carried by the annotation, so a wrong-length submission is a
    type error rather than a runtime check.
    """
    with pytest.raises(ValueError, match=message):
        Arrange(card_ids)


def test_play_accepts_only_hand_and_face_up_sources() -> None:
    """A play never names the face-down zone; blind cards use ``Reveal``."""
    assert Play(Zone.HAND, Rank.FIVE, 2).source is Zone.HAND
    assert Play(Zone.FACE_UP, Rank.FIVE, 1).source is Zone.FACE_UP
    with pytest.raises(ValueError, match="HAND or FACE_UP"):
        Play(Zone.FACE_DOWN, Rank.FIVE, 1)


@pytest.mark.parametrize("bad_count", [0, -1])
def test_play_requires_a_positive_count(bad_count: int) -> None:
    """A batch of no cards is not a decision."""
    with pytest.raises(ValueError, match="Play.count must be positive"):
        Play(Zone.HAND, Rank.FIVE, bad_count)


def test_reveal_requires_a_non_negative_slot() -> None:
    """A negative slot names no face-down position."""
    assert Reveal(SlotId(2)).slot == 2
    with pytest.raises(ValueError, match="Reveal.slot must be non-negative"):
        Reveal(SlotId(-1))


def test_moves_are_immutable_and_hashable() -> None:
    """Every move type is a frozen, hashable value object."""
    moves: list[Move] = [
        Arrange((CardId(0), CardId(1), CardId(2))),
        Play(Zone.HAND, Rank.TWO, 1),
        Reveal(SlotId(0)),
        PickUp(),
    ]
    assert len(set(moves)) == 4
    batch = Play(Zone.HAND, Rank.TWO, 1)
    with pytest.raises(AttributeError):
        batch.count = 2  # ty: ignore[invalid-assignment]
    assert PickUp() == PickUp()


def test_constraints_reject_a_joker_as_a_rank_bound() -> None:
    """A joker's enum value must never act as a comparison bound."""
    for constraint in (AtLeast, AtMost):
        with pytest.raises(ValueError, match="cannot be a joker"):
            constraint(Rank.JOKER)


def test_default_rules_profile_validates() -> None:
    """The unmodified profile is accepted and describes a 54-card deck."""
    config = RulesConfig()
    config.validate()
    assert config.id == "shed-v1"
    assert config.deck_size == 54


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (RulesConfig(id="shed-v2"), "Unknown rules profile"),
        (RulesConfig(max_players=6), "max_players"),
        (RulesConfig(refill_target=4), "refill_target"),
        (RulesConfig(joker_count=0, initial_hand_size=2), "initial_hand_size, joker_count"),
    ],
    ids=["unknown_id", "max_players", "refill_target", "two_fields"],
)
def test_correctly_typed_but_unsupported_profiles_are_rejected(
    config: RulesConfig, message: str
) -> None:
    """A changed profile must never silently claim to be ``shed-v1``."""
    with pytest.raises(ValueError, match=message):
        config.validate()
    with pytest.raises(ValueError, match=message):
        GameState.create(2, seed=1, rules=config)


@pytest.mark.parametrize(
    "config",
    [RulesConfig(joker_count=0), RulesConfig(id="shed-v2")],
    ids=["no_jokers", "unknown_id"],
)
def test_deck_construction_rejects_unsupported_profiles(config: RulesConfig) -> None:
    """A public deck builder never returns a deck the profile cannot support.

    ``joker_count=0`` is correctly typed and would otherwise quietly yield a
    52-card deck through both exported entry points.
    """
    with pytest.raises(ValueError):
        build_deck(config)
    with pytest.raises(ValueError):
        shuffled_deck(seed=1, config=config)


def test_phases_and_zones_are_distinct_values() -> None:
    """The stored enums cover exactly the documented phases and zones."""
    assert {phase.value for phase in Phase} == {"setup", "play", "finished"}
    assert {zone.value for zone in Zone} == {"hand", "face_up", "face_down"}
