"""Authoritative game state, immutable observations, and the deal helper.

``GameState`` holds every physical card, including hidden assignments and deck
order, and is only ever handed to trusted code. ``PlayerView`` is the immutable
projection an agent receives: it exposes public cards, the viewer's own hand,
and nothing else.

The deal helpers live here because they build state. They are split into a
shuffle step and a pure dealing step so a replay can rebuild the identical
opening position from a recorded deck order, without depending on the shuffle
implementation staying byte-for-byte stable.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from shed.engine.types import (
    DEFAULT_DEALER,
    DEFAULT_RULES,
    Arrange,
    Card,
    Move,
    Phase,
    PlayConstraint,
    PlayerId,
    RulesConfig,
    SlotId,
    StateInvariantError,
    Unrestricted,
    Zone,
    build_deck,
)

if TYPE_CHECKING:  # Type-only edge; events.py imports this module at runtime.
    from shed.engine.events import ObservedEvent

__all__ = [
    "GameState",
    "Outcome",
    "PlayerState",
    "PlayerView",
    "PublicPlayerState",
    "SetupState",
    "build_view",
    "deal_initial_state",
    "dealing_order",
    "derive_active_zone",
    "public_player_state",
    "shuffled_deck",
    "validate_decision_boundary",
]


def _by_id(cards: Iterable[Card]) -> tuple[Card, ...]:
    """Return the given cards sorted by identifier.

    Args:
        cards: Any iterable of cards.

    Returns:
        A tuple in ascending card-ID order, the stable representation used for
        hands and face-up collections in observations.
    """
    return tuple(sorted(cards, key=lambda card: card.id))


def derive_active_zone(
    hand_count: int,
    face_up_count: int,
    face_down_count: int,
    draw_count: int,
) -> Zone | None:
    """Derive the zone a player must decide from, per the profile's ordering.

    Hand play comes first, the face-up collection is only reachable once both
    hand and deck are empty, and face-down slots come last.

    Args:
        hand_count: Cards currently in hand.
        face_up_count: Cards in the face-up collection.
        face_down_count: Remaining face-down slots.
        draw_count: Cards left in the draw pile.

    Returns:
        The zone to decide from, or ``None`` when the player has no cards left
        anywhere and is therefore finished.

    Raises:
        StateInvariantError: If the hand is empty while the draw pile still has
            cards. The engine must replenish before a decision is requested;
            this is never a legal decision boundary.
    """
    if hand_count > 0:
        return Zone.HAND
    if draw_count > 0:
        raise StateInvariantError(
            "Hand is empty while the draw pile is not; refill before requesting a decision"
        )
    if face_up_count > 0:
        return Zone.FACE_UP
    if face_down_count > 0:
        return Zone.FACE_DOWN
    return None


@dataclass(slots=True)
class PlayerState:
    """Every card one player owns, including their hidden face-down cards.

    Face-down identities are unknown even to their owner; they are stored here
    because the engine is the authority, and must never reach a ``PlayerView``.

    Attributes:
        hand: Private hand, in insertion order.
        face_up: Public face-up collection; it does not cover any particular
            face-down slot.
        face_down: Remaining face-down cards keyed by their stable slot ID.
    """

    hand: list[Card] = field(default_factory=list)
    face_up: list[Card] = field(default_factory=list)
    face_down: dict[SlotId, Card] = field(default_factory=dict)

    @property
    def remaining_count(self) -> int:
        """Total cards this player still holds across all personal zones."""
        return len(self.hand) + len(self.face_up) + len(self.face_down)

    def active_zone(self, draw_count: int) -> Zone | None:
        """Return the zone this player must decide from.

        Args:
            draw_count: Cards left in the draw pile, which decides whether an
                empty hand means "refill first" or "play from the table".

        Returns:
            The active zone, or ``None`` when the player has finished.

        Raises:
            StateInvariantError: If a refill is still pending.
        """
        return derive_active_zone(
            len(self.hand), len(self.face_up), len(self.face_down), draw_count
        )


@dataclass(slots=True)
class SetupState:
    """Arrangement submissions collected but deliberately not yet applied.

    Submissions stay private here until every player has chosen, so nobody can
    react to an opponent's arrangement while still making their own.

    Attributes:
        pending: Players still to submit, in clockwise request order.
        submissions: Stored arrangements keyed by player.
    """

    pending: list[PlayerId]
    submissions: dict[PlayerId, Arrange] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Outcome:
    """Result of a finished game.

    Attributes:
        winner: The first player to empty every personal zone.
    """

    winner: PlayerId


@dataclass(slots=True)
class GameState:
    """The authoritative position: all cards, hidden assignments, and deck order.

    Never hand this to an agent. Agents receive :class:`PlayerView` instead.

    Attributes:
        rules: The fixed profile this game runs under.
        players: Per-player card ownership keyed by player.
        seat_order: Seats ``0..player_count-1`` in clockwise order.
        dealer: Seat that dealt; deal, arrangement, and tie-break order all
            start clockwise after it.
        phase: Stored phase; reveals, draws, burns, and pickups are atomic.
        current_player: Next arrangement submitter in SETUP, the actor in PLAY,
            and ``None`` in FINISHED.
        current_ply: Count of resolved PLAY decisions; setup does not increment
            it.
        draw_pile: Remaining deck; the end of the list is the next card drawn.
        discard_pile: The live pile, in play order.
        burned_cards: Cards removed from the game by burns.
        constraint: Restriction the next ordinary rank must satisfy.
        setup: Pending arrangement bookkeeping; present only during SETUP.
        outcome: Result; present only once FINISHED.
    """

    rules: RulesConfig
    players: dict[PlayerId, PlayerState]
    seat_order: tuple[PlayerId, ...]
    dealer: PlayerId
    phase: Phase
    current_player: PlayerId | None
    current_ply: int
    draw_pile: list[Card]
    discard_pile: list[Card]
    burned_cards: list[Card]
    constraint: PlayConstraint
    setup: SetupState | None
    outcome: Outcome | None

    def get_legal_moves(self) -> tuple[Move, ...]:
        """Return the current actor's legal moves.

        Builds the actor's view without history and delegates to the single
        legal-move generator, so this convenience never becomes a second rule
        engine.

        Returns:
            The legal moves, or an empty tuple when no actor is scheduled.
        """
        from shed.engine.rules import legal_moves  # Local import avoids a cycle.

        if self.current_player is None:
            return ()
        return legal_moves(build_view(self, self.current_player))


@dataclass(frozen=True, slots=True)
class PublicPlayerState:
    """What everybody knows about one player.

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
class PlayerView:
    """One player's immutable snapshot of the game.

    Every exposed collection is a tuple of frozen elements, and no field aliases
    a mutable structure inside :class:`GameState`. The view deliberately omits
    opponents' hands, all face-down identities, the shuffle seed, RNG state,
    deck order, and pending setup submissions.

    Attributes:
        rules: The public rules profile.
        viewer: The player this view belongs to.
        seat_order: Seats in clockwise order.
        dealer: The dealing seat.
        phase: Current phase.
        current_player: Whoever must decide next, or ``None`` when finished.
        current_ply: Resolved PLAY decisions so far.
        hand: The viewer's own hand, sorted by card ID.
        players: Public state for every seat, in seat order.
        draw_count: Cards left in the draw pile; order is not exposed.
        discard_pile: The live pile in play order.
        burned_cards: Cards removed from the game.
        constraint: Restriction the next ordinary rank must satisfy.
        outcome: Result once the game has finished.
        history: Events already filtered for this viewer.
    """

    rules: RulesConfig
    viewer: PlayerId
    seat_order: tuple[PlayerId, ...]
    dealer: PlayerId
    phase: Phase
    current_player: PlayerId | None
    current_ply: int
    hand: tuple[Card, ...]
    players: tuple[PublicPlayerState, ...]
    draw_count: int
    discard_pile: tuple[Card, ...]
    burned_cards: tuple[Card, ...]
    constraint: PlayConstraint
    outcome: Outcome | None
    history: tuple[ObservedEvent, ...]

    @property
    def me(self) -> PublicPlayerState:
        """Return the viewer's own public state.

        Returns:
            The public entry for :attr:`viewer`.

        Raises:
            StateInvariantError: If the viewer has no seat in this view.
        """
        for public in self.players:
            if public.player == self.viewer:
                return public
        raise StateInvariantError(f"Viewer {self.viewer} has no seat in this view")

    def get_legal_moves(self) -> tuple[Move, ...]:
        """Return this viewer's legal moves.

        Returns:
            The legal moves, empty when it is not the viewer's decision. The
            runner precomputes this tuple once per decision rather than caching
            it on the view.
        """
        from shed.engine.rules import legal_moves  # Local import avoids a cycle.

        return legal_moves(self)


def public_player_state(player: PlayerId, state: PlayerState) -> PublicPlayerState:
    """Project one player's cards onto what everybody may see.

    Args:
        player: The seat being described.
        state: That player's authoritative cards.

    Returns:
        A frozen public snapshot: hand size, sorted face-up cards, and ascending
        face-down slot IDs with no identities attached.
    """
    return PublicPlayerState(
        player=player,
        hand_count=len(state.hand),
        face_up=_by_id(state.face_up),
        face_down_slots=tuple(sorted(state.face_down)),
    )


def build_view(
    state: GameState,
    player: PlayerId,
    *,
    history: tuple[ObservedEvent, ...] = (),
) -> PlayerView:
    """Build one player's immutable observation of ``state``.

    This is strictly read-only: it never refills a hand, advances an actor, or
    otherwise repairs the position. Automatic work belongs in initialization or
    in ``apply_move``.

    Args:
        state: Authoritative state to observe.
        player: The viewing seat.
        history: Events already filtered for this viewer; callers must not pass
            another player's private identities.

    Returns:
        An independent snapshot sharing no mutable object with ``state``.

    Raises:
        StateInvariantError: If the player has no seat, or the state is not at a
            valid decision boundary because the actor still owes a refill.
    """
    if player not in state.players:
        raise StateInvariantError(f"Player {player} has no seat in this game")
    if state.phase is Phase.PLAY and state.current_player is not None:
        # Read-only enforcement of the actor's decision-boundary invariants: a
        # pending refill raises inside active_zone, and an actor with nothing
        # left means termination was never resolved.
        actor = state.players[state.current_player]
        if actor.active_zone(len(state.draw_pile)) is None:
            raise StateInvariantError(
                f"Actor {state.current_player} holds no cards; resolve termination "
                "before requesting another decision"
            )
    return PlayerView(
        rules=state.rules,
        viewer=player,
        seat_order=state.seat_order,
        dealer=state.dealer,
        phase=state.phase,
        current_player=state.current_player,
        current_ply=state.current_ply,
        hand=_by_id(state.players[player].hand),
        players=tuple(public_player_state(seat, state.players[seat]) for seat in state.seat_order),
        draw_count=len(state.draw_pile),
        discard_pile=tuple(state.discard_pile),
        burned_cards=tuple(state.burned_cards),
        constraint=state.constraint,
        outcome=state.outcome,
        history=history,
    )


def dealing_order(seat_order: tuple[PlayerId, ...], dealer: PlayerId) -> tuple[PlayerId, ...]:
    """Return the seats clockwise, starting with the one after the dealer.

    The same order governs dealing, arrangement requests, and opener tie-breaks.

    Args:
        seat_order: Seats in clockwise order.
        dealer: The dealing seat.

    Returns:
        A rotation of ``seat_order`` beginning after the dealer.

    Raises:
        StateInvariantError: If the dealer is not one of the seats.
    """
    if dealer not in seat_order:
        raise StateInvariantError(f"Dealer {dealer} is not a seat in {seat_order}")
    start = seat_order.index(dealer) + 1
    count = len(seat_order)
    return tuple(seat_order[(start + offset) % count] for offset in range(count))


def shuffled_deck(*, seed: int, config: RulesConfig = DEFAULT_RULES) -> list[Card]:
    """Shuffle the canonical deck with a dedicated generator.

    A private ``random.Random`` keeps deck randomness isolated from agent and
    fallback seed streams and never touches module-global randomness.

    Args:
        seed: Deck seed; trusted match metadata only, never shown to agents.
        config: Rules profile supplying the deck composition.

    Returns:
        The shuffled deck. The end of the list is the next card to be drawn.
    """
    deck = list(build_deck(config))
    random.Random(seed).shuffle(deck)
    return deck


def deal_initial_state(
    deck: list[Card] | tuple[Card, ...],
    *,
    player_count: int,
    dealer: PlayerId = DEFAULT_DEALER,
    config: RulesConfig = DEFAULT_RULES,
) -> GameState:
    """Deal a shuffled deck into a fresh SETUP state.

    Dealing runs clockwise starting after the dealer: three rounds into
    face-down slots ``0,1,2``, then three rounds face up, then three rounds into
    hands, each round giving one card to every seat. Cards come off the end of
    the deck list, which is the draw position.

    Taking the deck as an argument keeps this deterministic and reusable: a
    replay reconstructs the opening position from the recorded deck order
    instead of re-running the shuffle.

    Args:
        deck: The shuffled deck; must be a permutation of the canonical deck.
        player_count: Number of seats, within the profile's supported range.
        dealer: Dealing seat.
        config: Rules profile.

    Returns:
        A SETUP state with arrangements pending for every seat, an empty discard
        pile, no burned cards, and an unrestricted constraint.

    Raises:
        ValueError: If the player count is unsupported, the dealer is not a
            seat, or the deck is not the canonical 54 cards.
    """
    config.validate()
    if not config.min_players <= player_count <= config.max_players:
        raise ValueError(
            f"{config.id} supports {config.min_players}-{config.max_players} players, "
            f"got {player_count}"
        )
    if dealer not in range(player_count):
        raise ValueError(f"Dealer {dealer} is not a seat in a {player_count}-player game")
    canonical = build_deck(config)
    if sorted(deck, key=lambda card: card.id) != list(canonical):
        raise ValueError("Deck is not a permutation of the canonical deck")

    seat_order = tuple(PlayerId(seat) for seat in range(player_count))
    players = {seat: PlayerState() for seat in seat_order}
    order = dealing_order(seat_order, dealer)
    remaining = list(deck)

    for slot in range(config.initial_face_down_count):
        for seat in order:
            players[seat].face_down[SlotId(slot)] = remaining.pop()
    for _ in range(config.initial_face_up_count):
        for seat in order:
            players[seat].face_up.append(remaining.pop())
    for _ in range(config.initial_hand_size):
        for seat in order:
            players[seat].hand.append(remaining.pop())

    state = GameState(
        rules=config,
        players=players,
        seat_order=seat_order,
        dealer=dealer,
        phase=Phase.SETUP,
        current_player=order[0],
        current_ply=0,
        draw_pile=remaining,
        discard_pile=[],
        burned_cards=[],
        constraint=Unrestricted(),
        setup=SetupState(pending=list(order)),
        outcome=None,
    )
    validate_decision_boundary(state)
    return state


def validate_decision_boundary(state: GameState) -> None:
    """Check the invariants that must hold whenever a decision is requested.

    Covers card conservation against the canonical deck, seat and actor
    consistency, the empty-pile constraint rule, and the phase-specific setup
    and finished conditions. Refill and active-zone consistency is enforced by
    :func:`derive_active_zone` when the actor's view is built.

    Args:
        state: Authoritative state to check.

    Raises:
        StateInvariantError: If any decision-boundary invariant is violated.
    """
    canonical = build_deck(state.rules)
    seen: list[Card] = [*state.draw_pile, *state.discard_pile, *state.burned_cards]
    for seat in state.seat_order:
        player = state.players[seat]
        seen.extend(player.hand)
        seen.extend(player.face_up)
        seen.extend(player.face_down.values())
    identifiers = [card.id for card in seen]
    if len(set(identifiers)) != len(identifiers):
        raise StateInvariantError("A physical card occurs in more than one zone")
    if sorted(seen, key=lambda card: card.id) != list(canonical):
        raise StateInvariantError("Cards in play do not match the canonical deck")
    if set(state.players) != set(state.seat_order):
        raise StateInvariantError("Player states and seat order disagree")
    if state.current_player is not None and state.current_player not in state.seat_order:
        raise StateInvariantError(f"Actor {state.current_player} is not a seat")
    if not state.discard_pile and not isinstance(state.constraint, Unrestricted):
        raise StateInvariantError("An empty discard pile must leave the constraint unrestricted")

    match state.phase:
        case Phase.SETUP:
            setup = state.setup
            if setup is None or not setup.pending:
                raise StateInvariantError("SETUP requires outstanding arrangement submissions")
            if state.current_player != setup.pending[0]:
                raise StateInvariantError("The SETUP actor must be the next pending submitter")
            if set(setup.submissions) & set(setup.pending):
                raise StateInvariantError("A pending player must not have a stored submission")
            if state.outcome is not None:
                raise StateInvariantError("SETUP cannot have an outcome")
        case Phase.PLAY:
            if state.setup is not None:
                raise StateInvariantError("PLAY must not retain setup state")
            if state.current_player is None:
                raise StateInvariantError("PLAY requires an actor")
            if state.outcome is not None:
                raise StateInvariantError("PLAY cannot have an outcome")
            if not state.players[state.current_player].remaining_count:
                raise StateInvariantError(
                    f"Actor {state.current_player} holds no cards; resolve termination "
                    "before requesting another decision"
                )
        case Phase.FINISHED:
            if state.setup is not None:
                raise StateInvariantError("FINISHED must not retain setup state")
            if state.current_player is not None:
                raise StateInvariantError("FINISHED must not schedule an actor")
            if state.outcome is None:
                raise StateInvariantError("FINISHED requires an outcome")
            if state.players[state.outcome.winner].remaining_count:
                raise StateInvariantError("The winner must hold no cards")
