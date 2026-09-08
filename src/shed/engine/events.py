"""Observed events, per-recipient filtering, and the transition/undo shape.

Events are the only record of what happened. The engine produces *full* events
that retain hidden identities; those are for trusted callers such as the match
runner and the replay writer. Before an event reaches an agent it passes through
:func:`filter_events_for`, which strips identities the recipient may not know
while keeping the counts everybody can see.

History itself lives in the runner, outside ``GameState``: growing history must
never enter an undo snapshot or a replay's state records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from shed.engine.state import GameState, Outcome, PublicPlayerState
from shed.engine.types import Card, Move, PlayerId, SlotId, Zone

__all__ = [
    "ArrangementCommitted",
    "BurnReason",
    "CardRevealed",
    "CardsDrawn",
    "CardsPlayed",
    "Decision",
    "GameEnded",
    "GameStarted",
    "HandDealt",
    "ObservedEvent",
    "PileBurned",
    "PilePickedUp",
    "Transition",
    "UndoRecord",
    "filter_event_for",
    "filter_events_for",
]


class BurnReason(Enum):
    """Why a pile was burned. A ten takes precedence over a four-card batch."""

    TEN = "ten"
    FOUR_OF_A_KIND = "four_of_a_kind"


@dataclass(frozen=True, slots=True)
class GameStarted:
    """Public opening announcement, emitted once per game.

    Attributes:
        dealer: The dealing seat.
        seat_order: Seats in clockwise order.
        players: Public state of every seat straight after the deal, which lets
            agents remember the cards that were visible before arrangements.
    """

    dealer: PlayerId
    seat_order: tuple[PlayerId, ...]
    players: tuple[PublicPlayerState, ...]


@dataclass(frozen=True, slots=True)
class HandDealt:
    """Cards dealt to one hand; identities are private to the recipient.

    Attributes:
        player: Who received the cards.
        count: How many cards, public to everybody.
        cards: The identities, or ``None`` in a copy filtered for anybody but
            the recipient.
    """

    player: PlayerId
    count: int
    cards: tuple[Card, ...] | None


@dataclass(frozen=True, slots=True)
class ArrangementCommitted:
    """One player's final face-up set, emitted only at the collective commit.

    Attributes:
        player: Whose arrangement this is.
        face_up: The three final face-up cards, sorted by card ID.
    """

    player: PlayerId
    face_up: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class CardsPlayed:
    """A batch moved from a personal zone onto the pile; public.

    Attributes:
        player: The actor.
        source: Zone the batch came from.
        cards: The selected physical cards, in transfer order.
    """

    player: PlayerId
    source: Zone
    cards: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class CardRevealed:
    """A face-down slot turned over; public, including the failure case.

    A successful reveal is represented by this event alone. No ``CardsPlayed``
    event is emitted for the same physical transfer.

    Attributes:
        player: The actor.
        slot: The revealed slot.
        card: The revealed card, now known to everybody.
        playable: Whether the card satisfied the pre-reveal constraint.
    """

    player: PlayerId
    slot: SlotId
    card: Card
    playable: bool


@dataclass(frozen=True, slots=True)
class CardsDrawn:
    """Replenishment draw; identities are private to the drawing player.

    Attributes:
        player: Who drew.
        count: How many cards, public to everybody.
        cards: The identities, or ``None`` in a filtered copy.
    """

    player: PlayerId
    count: int
    cards: tuple[Card, ...] | None


@dataclass(frozen=True, slots=True)
class PilePickedUp:
    """The pile taken into a hand; public, so everybody can track those cards.

    Attributes:
        player: Who picked up.
        cards: Everything transferred, including a failed reveal's card when
            applicable.
    """

    player: PlayerId
    cards: tuple[Card, ...]


@dataclass(frozen=True, slots=True)
class PileBurned:
    """The pile removed from the game; public.

    Attributes:
        player: The actor whose action burned the pile.
        cards: Everything burned, including the cards just played.
        reason: Which rule caused the burn.
    """

    player: PlayerId
    cards: tuple[Card, ...]
    reason: BurnReason


@dataclass(frozen=True, slots=True)
class GameEnded:
    """Final result; public.

    Attributes:
        outcome: The winner.
    """

    outcome: Outcome


type ObservedEvent = (
    GameStarted
    | HandDealt
    | ArrangementCommitted
    | CardsPlayed
    | CardRevealed
    | CardsDrawn
    | PilePickedUp
    | PileBurned
    | GameEnded
)


def filter_event_for(event: ObservedEvent, viewer: PlayerId) -> ObservedEvent:
    """Return the copy of ``event`` that ``viewer`` is allowed to observe.

    Private events keep their public count but lose their identities for anybody
    other than the player they concern. Every other event is already public and
    is returned unchanged, which is safe because events are frozen.

    Args:
        event: A full internal event.
        viewer: The recipient.

    Returns:
        The event as this viewer may see it.
    """
    match event:
        case HandDealt(player=player) if player != viewer:
            return HandDealt(player=player, count=event.count, cards=None)
        case CardsDrawn(player=player) if player != viewer:
            return CardsDrawn(player=player, count=event.count, cards=None)
        case _:
            return event


def filter_events_for(
    events: tuple[ObservedEvent, ...], viewer: PlayerId
) -> tuple[ObservedEvent, ...]:
    """Filter a run of full events for one recipient.

    Args:
        events: Full internal events in resolution order.
        viewer: The recipient.

    Returns:
        The events as this viewer may see them, in the same order.
    """
    return tuple(filter_event_for(event, viewer) for event in events)


@dataclass(frozen=True, slots=True)
class Decision:
    """The move one player actually took.

    Attributes:
        player: Who decided.
        move: The canonical move that was applied.
    """

    player: PlayerId
    move: Move


@dataclass(slots=True)
class UndoRecord:
    """An independent snapshot of the state from before a move was applied.

    The first implementation snapshots the whole state rather than computing
    deltas; benchmark before replacing it.

    Attributes:
        before: Deep copy of the state as it was before mutation. It is never
            serialized into a replay and never reaches an agent.
    """

    before: GameState


@dataclass(slots=True)
class Transition:
    """Everything produced by applying one move.

    Attributes:
        decision: The applied decision.
        undo: Snapshot supporting LIFO undo on the originating state.
        events: Full events with hidden identities intact; the runner filters
            them per recipient before they reach an agent.
    """

    decision: Decision
    undo: UndoRecord
    events: tuple[ObservedEvent, ...]
