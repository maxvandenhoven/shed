"""Tests for initial events and the per-recipient information filter."""

from dataclasses import fields

import pytest

from shed.engine import (
    BurnReason,
    CardRevealed,
    CardsDrawn,
    CardsPlayed,
    GameEnded,
    GameStarted,
    GameState,
    HandDealt,
    ObservedEvent,
    Outcome,
    PileBurned,
    PilePickedUp,
    PlayerId,
    Rank,
    SlotId,
    Zone,
    build_deck,
    filter_event_for,
    filter_events_for,
)


def test_initial_events_open_with_the_public_deal() -> None:
    """A public opening event is followed by one hand deal per seat."""
    state = GameState.create(3, seed=51)
    events = state.initial_events()
    assert isinstance(events[0], GameStarted)
    assert [type(event) for event in events[1:]] == [HandDealt] * 3
    assert [event.player for event in events[1:] if isinstance(event, HandDealt)] == [1, 2, 0]


def test_game_started_records_the_cards_visible_before_arranging() -> None:
    """Face-up cards and slot counts are public; hands are only counted."""
    state = GameState.create(4, seed=52, dealer=PlayerId(2))
    started = state.initial_events()[0]
    assert isinstance(started, GameStarted)
    assert started.dealer == 2
    assert started.seat_order == state.seat_order
    for public in started.players:
        dealt = state.players[public.player]
        assert public.face_up == tuple(sorted(dealt.face_up, key=lambda card: card.id))
        assert public.face_down_slots == (0, 1, 2)
        assert public.hand_count == 3


def test_hand_deals_carry_identities_only_for_their_recipient() -> None:
    """A private deal keeps its count for everybody and its cards for one seat."""
    state = GameState.create(3, seed=53)
    events = state.initial_events()

    for seat in state.seat_order:
        filtered = filter_events_for(events, seat)
        deals = [event for event in filtered if isinstance(event, HandDealt)]
        assert len(deals) == 3
        for deal in deals:
            assert deal.count == 3
            if deal.player == seat:
                assert deal.cards == tuple(
                    sorted(state.players[seat].hand, key=lambda card: card.id)
                )
            else:
                assert deal.cards is None


def test_filtered_history_never_reveals_another_hand() -> None:
    """No opponent card identity survives into a player's own history."""
    state = GameState.create(2, seed=54)
    history = filter_events_for(state.initial_events(), PlayerId(0))
    view = state.observe(PlayerId(0), history=history)

    opponent = {card.id for card in state.players[PlayerId(1)].hand}
    seen = {
        card.id
        for event in view.history
        if isinstance(event, HandDealt) and event.cards is not None
        for card in event.cards
    }
    assert not seen & opponent
    assert seen == {card.id for card in state.players[PlayerId(0)].hand}


def test_drawn_cards_are_private_to_the_drawing_player() -> None:
    """``CardsDrawn`` follows the same rule as a hand deal."""
    cards = build_deck()[:2]
    event = CardsDrawn(player=PlayerId(1), count=2, cards=cards)
    assert filter_event_for(event, PlayerId(1)) == event
    hidden = filter_event_for(event, PlayerId(0))
    assert hidden == CardsDrawn(player=PlayerId(1), count=2, cards=None)
    assert hidden.count == 2


@pytest.mark.parametrize(
    "event",
    [
        CardsPlayed(player=PlayerId(0), source=Zone.HAND, cards=build_deck()[:2]),
        CardRevealed(player=PlayerId(0), slot=SlotId(1), card=build_deck()[5], playable=False),
        PilePickedUp(player=PlayerId(0), cards=build_deck()[:3]),
        PileBurned(player=PlayerId(0), cards=build_deck()[:4], reason=BurnReason.TEN),
        GameEnded(outcome=Outcome(winner=PlayerId(0))),
    ],
    ids=["played", "revealed", "picked_up", "burned", "ended"],
)
def test_public_events_reach_every_viewer_unchanged(event: ObservedEvent) -> None:
    """Filtering a public event returns it as it is, for any recipient."""
    for viewer in (PlayerId(0), PlayerId(1)):
        assert filter_event_for(event, viewer) is event


def test_events_are_frozen_values() -> None:
    """Events are immutable, so sharing a public one between viewers is safe."""
    event = PileBurned(player=PlayerId(0), cards=build_deck()[:4], reason=BurnReason.FOUR_OF_A_KIND)
    with pytest.raises(AttributeError):
        event.reason = BurnReason.TEN  # ty: ignore[invalid-assignment]
    assert {event, event} == {event}


def test_burn_reasons_cover_both_rules() -> None:
    """The recorded burn reason distinguishes a ten from a four-card batch."""
    assert {reason.value for reason in BurnReason} == {"ten", "four_of_a_kind"}


def test_revealed_cards_record_their_playability() -> None:
    """A reveal is a single public event, including whether it succeeded."""
    card = next(item for item in build_deck() if item.rank is Rank.ACE)
    event = CardRevealed(player=PlayerId(2), slot=SlotId(0), card=card, playable=True)
    assert {item.name for item in fields(CardRevealed)} == {"player", "slot", "card", "playable"}
    assert event.playable is True
