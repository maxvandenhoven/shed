"""Public interface of the Shed engine.

The engine owns cards, authoritative state, observations, legality, and
transitions. It depends only on the standard library and never imports agents,
timing, scripts, or multiprocessing, nor any JSON, transport, or replay
decoding.

``GameState`` is the entry point: create a game with :meth:`GameState.create`,
then use ``get_legal_moves``, ``observe``, ``initial_events``, ``apply_move``,
and ``undo_move`` on it. Modules depend one way -- ``types`` defines the value
types, ``events`` records what happened with them, and ``state`` builds the
game on both -- and engine code imports from those modules directly rather than
through this file.

The deterministic engine is complete: the deck, dealing, setup, observations,
legality, the full PLAY transition -- effects, burns, pickups, blind reveals,
replenishment, and termination -- and snapshot undo for all of them. Timing,
processes, agents, replay, and the gauntlet are separate layers built on top and
are not part of this package.

The profile's individual rules are exported as well as applied: ``can_play_rank``,
``constraint_after``, ``burn_reason``, ``legal_batches``, and
``active_zone_for_counts`` are the engine's only implementations of rank
legality, the constraint transition, the burn rule, batch generation, and the
active-zone ordering. A caller that tracks a game the engine does not own --
:mod:`shed.companion`, following a physical game from observed ranks -- calls
these instead of restating them, so the two cannot drift. Each takes counts,
ranks, and constraints rather than ``GameState``, which is what makes them usable
without one.
"""

from shed.engine.events import (
    ArrangementCommitted,
    BurnReason,
    CardRevealed,
    CardsDrawn,
    CardsPlayed,
    Decision,
    GameEnded,
    GameStarted,
    HandDealt,
    ObservedEvent,
    PileBurned,
    PilePickedUp,
    Transition,
    UndoRecord,
    filter_event_for,
    filter_events_for,
)
from shed.engine.state import (
    ALWAYS_PLAYABLE,
    BURN_BATCH_SIZE,
    GameState,
    PlayerState,
    PlayerView,
    SetupState,
    active_zone_for_counts,
    burn_reason,
    can_play_rank,
    constraint_after,
    deal_initial_state,
    dealing_order,
    legal_batches,
    shuffled_deck,
    validate_decision_boundary,
)
from shed.engine.types import (
    DEFAULT_DEALER,
    DEFAULT_RULES,
    ORDINARY_RANKS,
    SUIT_ORDER,
    Arrange,
    AtLeast,
    AtMost,
    Card,
    CardId,
    IllegalMoveError,
    Move,
    Outcome,
    Phase,
    PickUp,
    Play,
    PlayConstraint,
    PlayerId,
    PublicPlayerState,
    Rank,
    Reveal,
    RulesConfig,
    SlotId,
    StateInvariantError,
    Suit,
    Unrestricted,
    Zone,
    build_deck,
)

__all__ = [
    "ALWAYS_PLAYABLE",
    "BURN_BATCH_SIZE",
    "DEFAULT_DEALER",
    "DEFAULT_RULES",
    "ORDINARY_RANKS",
    "SUIT_ORDER",
    "Arrange",
    "ArrangementCommitted",
    "AtLeast",
    "AtMost",
    "BurnReason",
    "Card",
    "CardId",
    "CardRevealed",
    "CardsDrawn",
    "CardsPlayed",
    "Decision",
    "GameEnded",
    "GameStarted",
    "GameState",
    "HandDealt",
    "IllegalMoveError",
    "Move",
    "ObservedEvent",
    "Outcome",
    "Phase",
    "PickUp",
    "PileBurned",
    "PilePickedUp",
    "Play",
    "PlayConstraint",
    "PlayerId",
    "PlayerState",
    "PlayerView",
    "PublicPlayerState",
    "Rank",
    "Reveal",
    "RulesConfig",
    "SetupState",
    "SlotId",
    "StateInvariantError",
    "Suit",
    "Transition",
    "UndoRecord",
    "Unrestricted",
    "Zone",
    "active_zone_for_counts",
    "build_deck",
    "burn_reason",
    "can_play_rank",
    "constraint_after",
    "deal_initial_state",
    "dealing_order",
    "filter_event_for",
    "filter_events_for",
    "legal_batches",
    "shuffled_deck",
    "validate_decision_boundary",
]
