"""Authoritative game state, immutable observations, and the game operations.

``GameState`` is the engine's entry point. It holds every physical card,
including hidden assignments and deck order, and exposes the operations a
trusted caller needs: create a game, list the actor's legal moves, observe it as
one player, apply a move, and undo it. ``PlayerView`` is the immutable
projection an agent receives -- public cards, the viewer's own hand, and the
moves that viewer may make.

The deal helpers are split into a shuffle step and a pure dealing step so a
replay can rebuild the identical opening position from a recorded deck order,
without depending on the shuffle implementation staying byte-for-byte stable.

Dependencies point one way: :mod:`shed.engine.types` defines the value types,
:mod:`shed.engine.events` records what happened using those types, and this
module builds state and operations on both. Nothing here imports agents, clocks,
processes, or serialization.

A decision is atomic. :meth:`GameState.apply_move` validates the move, snapshots
the position, then resolves the whole chain -- transfer or reveal, burn or rank
effect, pickup, replenishment, termination, and the next actor -- before
returning. Any failure inside that boundary, including the closing invariant
check, restores the snapshot, so a caller only ever sees the position before the
decision or the position after it completes.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields
from itertools import combinations

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
)
from shed.engine.types import (
    DEFAULT_DEALER,
    DEFAULT_RULES,
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
    Unrestricted,
    Zone,
    build_deck,
)

__all__ = [
    "GameState",
    "PlayerState",
    "PlayerView",
    "SetupState",
    "can_play_rank",
    "deal_initial_state",
    "dealing_order",
    "shuffled_deck",
    "validate_decision_boundary",
]

_ALWAYS_PLAYABLE: frozenset[Rank] = frozenset({Rank.TWO, Rank.NINE, Rank.TEN, Rank.JOKER})
"""Ranks exempt from the current constraint, checked before any comparison."""

_BURN_BATCH_SIZE = 4
"""Batch size that burns the pile when played in one action."""

_OPENING_RANK_ORDER: tuple[Rank, ...] = (
    Rank.THREE,
    Rank.FOUR,
    Rank.FIVE,
    Rank.SIX,
    Rank.SEVEN,
    Rank.EIGHT,
    Rank.NINE,
    Rank.TEN,
    Rank.JACK,
    Rank.QUEEN,
    Rank.KING,
    Rank.ACE,
    Rank.TWO,
    Rank.JOKER,
)
"""Ranks searched when choosing the opener; specials are deliberately last."""


def can_play_rank(rank: Rank, constraint: PlayConstraint) -> bool:
    """Report whether a rank may be played against a constraint.

    The always-playable exceptions are applied before any ordinary comparison,
    so a joker's enum value never determines its strength.

    Args:
        rank: Rank of the batch or revealed card.
        constraint: Restriction currently imposed by the pile.

    Returns:
        ``True`` if the rank is legal to play.

    Raises:
        ValueError: If the constraint is not a known constraint type. The match
            below is exhaustive for ``PlayConstraint``; this guards a constraint
            added without updating it.
    """
    if rank in _ALWAYS_PLAYABLE:
        return True
    match constraint:
        case Unrestricted():
            return True
        case AtLeast(rank=minimum):
            return rank >= minimum
        case AtMost(rank=maximum):
            return rank <= maximum
    raise ValueError(f"Unknown play constraint {constraint!r}")


def _burn_reason(rank: Rank, count: int) -> BurnReason | None:
    """Report which rule, if any, burns the pile after a successful play.

    A ten burns whatever the batch size, and four cards of one rank burn when
    they are played in a *single* action; four equal ranks that merely
    accumulated across separate actions never burn. When both apply the ten is
    recorded, as the profile requires.

    Args:
        rank: Rank just played, or the rank of a successfully revealed card.
        count: Cards in that single batch; a reveal is always one.

    Returns:
        The burn reason, or ``None`` when the pile survives.
    """
    if rank is Rank.TEN:
        return BurnReason.TEN
    if count == _BURN_BATCH_SIZE:
        return BurnReason.FOUR_OF_A_KIND
    return None


def _constraint_after(rank: Rank, current: PlayConstraint) -> PlayConstraint:
    """Return the constraint a successful, non-burning play leaves behind.

    The seven restriction lives in the constraint rather than in a countdown of
    players, so a transparent nine simply preserves whatever is already there:
    seven then nine still demands at most a seven, while seven then five leaves
    at least a five.

    Args:
        rank: Rank just played or revealed. A ten never reaches here because it
            always burns.
        current: Constraint in force before this play.

    Returns:
        The constraint the next ordinary rank must satisfy.

    Raises:
        StateInvariantError: If a ten reaches here, which means burn resolution
            was skipped.
    """
    match rank:
        case Rank.TEN:
            raise StateInvariantError("A ten always burns; it never sets a constraint")
        case Rank.NINE:
            return current
        case Rank.JOKER:
            return Unrestricted()
        case Rank.SEVEN:
            return AtMost(Rank.SEVEN)
        case _:
            return AtLeast(rank)


def _by_id(cards: Iterable[Card]) -> tuple[Card, ...]:
    """Return the given cards sorted by identifier.

    Args:
        cards: Any iterable of cards.

    Returns:
        A tuple in ascending card-ID order, the stable representation used for
        hands and face-up collections in observations.
    """
    return tuple(sorted(cards, key=lambda card: card.id))


def _active_zone(
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
        return _active_zone(len(self.hand), len(self.face_up), len(self.face_down), draw_count)

    def public(self, player: PlayerId) -> PublicPlayerState:
        """Project these cards onto what everybody may see.

        Args:
            player: The seat being described.

        Returns:
            A frozen public snapshot: hand size, sorted face-up cards, and
            ascending face-down slot IDs with no identities attached.
        """
        return PublicPlayerState(
            player=player,
            hand_count=len(self.hand),
            face_up=_by_id(self.face_up),
            face_down_slots=tuple(sorted(self.face_down)),
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
        legal_moves: What this viewer may do now, empty unless it is their
            decision. It describes the state observed here; the engine
            revalidates on every ``apply_move``, so holding a stale tuple is
            never authority to mutate a later state.
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
    legal_moves: tuple[Move, ...]

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


@dataclass(slots=True)
class GameState:
    """The authoritative position, and the operations that advance it.

    Never hand this to an agent: it carries every hidden assignment and the deck
    order. Agents receive :meth:`observe` output instead.

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

    @classmethod
    def create(
        cls,
        player_count: int,
        *,
        seed: int,
        dealer: PlayerId = DEFAULT_DEALER,
        rules: RulesConfig = DEFAULT_RULES,
    ) -> GameState:
        """Shuffle and deal a fresh game.

        Args:
            player_count: Number of seats, within the profile's range.
            seed: Deck seed for the dedicated shuffle generator. It is trusted
                metadata: never expose it to an agent.
            dealer: Dealing seat; deal, arrangement, and tie-break order all
                start clockwise after it.
            rules: Rules profile; only the fixed ``shed-v1`` profile is
                supported.

        Returns:
            A SETUP state awaiting the first arrangement.

        Raises:
            ValueError: If the profile, player count, or dealer is invalid.
        """
        deck = shuffled_deck(seed=seed, config=rules)
        return deal_initial_state(deck, player_count=player_count, dealer=dealer, config=rules)

    @property
    def is_finished(self) -> bool:
        """Whether the game has ended; derived from the phase, never stored."""
        return self.phase is Phase.FINISHED

    def get_legal_moves(self) -> tuple[Move, ...]:
        """Generate every legal move for the current actor.

        This is the engine's single legality implementation. It reads only the
        actor's own cards, the public constraint, and the size of the draw pile,
        so legality can never depend on an opponent's hidden cards or on deck
        order. No observation is built: :meth:`observe` calls this, not the
        other way round.

        Returns:
            A deterministically ordered tuple: the 20 arrangements in SETUP;
            in PLAY every legal rank/count batch from the one active zone, or
            every remaining face-down slot, or a forced pickup; and nothing once
            the game is finished.

        Raises:
            StateInvariantError: If the position is not a valid live decision
                boundary -- no actor scheduled, a refill still owed, or an actor
                who holds no cards at all.
        """
        if self.phase is Phase.FINISHED:
            return ()
        actor = self.current_player
        if actor is None:
            raise StateInvariantError(f"No actor is scheduled in phase {self.phase.value}")
        player = self.players[actor]
        if self.phase is Phase.SETUP:
            return _arrangements(player)

        zone = player.active_zone(len(self.draw_pile))
        if zone is None:
            raise StateInvariantError(
                f"Actor {actor} holds no cards; resolve termination "
                "before requesting another decision"
            )
        if zone is Zone.FACE_DOWN:
            # Every remaining slot is legal; hidden ranks are never inspected.
            return tuple(Reveal(slot) for slot in sorted(player.face_down))
        return _batches(player.hand if zone is Zone.HAND else player.face_up, zone, self.constraint)

    def observe(
        self,
        player: PlayerId,
        *,
        history: tuple[ObservedEvent, ...] = (),
    ) -> PlayerView:
        """Build one player's immutable observation.

        Strictly read-only: it never refills a hand, advances an actor, or
        otherwise repairs the position. Automatic work belongs in creation or in
        :meth:`apply_move`. Generating the actor's moves also checks that this
        is a valid decision boundary, so an unresolved position is refused
        whoever asks to observe it.

        Args:
            player: The viewing seat.
            history: That player's already-filtered history. ``observe`` does
                not filter: passing another player's unfiltered events would
                leak hidden information into the observation.

        Returns:
            An independent snapshot sharing no mutable object with this state,
            carrying the viewer's legal moves and nothing they may not know.

        Raises:
            StateInvariantError: If the player has no seat, or the state is not
                at a valid decision boundary.
        """
        if player not in self.players:
            raise StateInvariantError(f"Player {player} has no seat in this game")
        actor_moves = self.get_legal_moves()
        return PlayerView(
            rules=self.rules,
            viewer=player,
            seat_order=self.seat_order,
            dealer=self.dealer,
            phase=self.phase,
            current_player=self.current_player,
            current_ply=self.current_ply,
            hand=_by_id(self.players[player].hand),
            players=tuple(self.players[seat].public(seat) for seat in self.seat_order),
            draw_count=len(self.draw_pile),
            discard_pile=tuple(self.discard_pile),
            burned_cards=tuple(self.burned_cards),
            constraint=self.constraint,
            outcome=self.outcome,
            history=history,
            legal_moves=actor_moves if player == self.current_player else (),
        )

    def initial_events(self) -> tuple[ObservedEvent, ...]:
        """Describe the deal as full events.

        The public opening event records the face-up cards visible before any
        arrangement, which is what lets agents remember where those cards
        started. Hand deals are private events: filtering replaces their
        identities for every player but the recipient.

        Returns:
            The opening event followed by one hand deal per seat, in dealing
            order. Identities are intact; filter before showing them to agents.
        """
        events: list[ObservedEvent] = [
            GameStarted(
                dealer=self.dealer,
                seat_order=self.seat_order,
                players=tuple(self.players[seat].public(seat) for seat in self.seat_order),
            )
        ]
        for seat in dealing_order(self.seat_order, self.dealer):
            hand = _by_id(self.players[seat].hand)
            events.append(HandDealt(player=seat, count=len(hand), cards=hand))
        return tuple(events)

    def apply_move(self, move: Move) -> Transition:
        """Validate and apply one decision, mutating this state in place.

        The move is revalidated against the position as it is now: a legal-move
        tuple observed earlier carries no authority to mutate a later state.
        Validation completes before any mutation, so a rejected move leaves the
        state exactly as it was.

        Args:
            move: The submitted decision.

        Returns:
            The transition: the applied decision, an undo snapshot, and the full
            events in physical resolution order.

        Raises:
            IllegalMoveError: If the move is out of phase or is not among the
                actor's current legal moves.
            StateInvariantError: If the state is not at a valid decision
                boundary, or resolution left it in one that is not.
        """
        if self.phase is Phase.FINISHED:
            raise IllegalMoveError("The game has finished; no move can be applied")
        actor = self.current_player
        if actor is None:
            raise StateInvariantError("No actor is scheduled to decide")

        legal = self.get_legal_moves()
        if move not in legal:
            raise IllegalMoveError(
                f"{move!r} is not legal for player {actor} in phase {self.phase.value}"
            )
        canonical = legal[legal.index(move)]

        undo = UndoRecord(before=deepcopy(self))
        try:
            if self.phase is Phase.SETUP:
                events = self._commit_arrangement(actor, canonical)
            else:
                events = self._resolve_play(actor, canonical)
            # The postcondition is part of the transition: a state that fails it
            # must be rolled back too, not left half-applied behind an error.
            validate_decision_boundary(self)
        except Exception:
            self._restore(undo.before)
            raise
        return Transition(decision=Decision(player=actor, move=canonical), undo=undo, events=events)

    def undo_move(self, transition: Transition) -> None:
        """Restore the state captured before ``transition`` was applied.

        Fields are restored on this object, so every holder of the reference
        observes the rollback. The snapshot is copied again on the way back,
        keeping the record reusable if the state is mutated after the undo. Undo
        is LIFO on the originating state; arbitrary out-of-order undo is not
        supported.

        Args:
            transition: The transition to roll back.
        """
        self._restore(transition.undo.before)

    def _commit_arrangement(self, actor: PlayerId, move: Move) -> tuple[ObservedEvent, ...]:
        """Store one arrangement, committing every arrangement once all arrive.

        A stored arrangement changes no visible card: hands and face-up
        collections stay exactly as dealt until the collective commit, so nobody
        can react to an opponent's choice while still making their own.

        Args:
            actor: The submitting player.
            move: The submitted, already validated arrangement.

        Returns:
            No events while submissions are still outstanding, otherwise one
            public commitment event per player in arrangement order.

        Raises:
            StateInvariantError: If setup bookkeeping is missing or inconsistent.
        """
        if not isinstance(move, Arrange):
            raise StateInvariantError(f"SETUP resolved a non-arrangement move {move!r}")
        setup = self.setup
        if setup is None:
            raise StateInvariantError("SETUP phase without setup state")

        setup.submissions[actor] = move
        setup.pending.remove(actor)
        if setup.pending:
            self.current_player = setup.pending[0]
            return ()

        events: list[ObservedEvent] = []
        for seat in dealing_order(self.seat_order, self.dealer):
            arrangement = setup.submissions[seat]
            player = self.players[seat]
            pool: dict[CardId, Card] = {card.id: card for card in (*player.hand, *player.face_up)}
            chosen = set(arrangement.face_up_cards)
            if not chosen <= set(pool):
                raise StateInvariantError(f"Player {seat} arranged cards they do not own")
            player.face_up = [pool[card_id] for card_id in sorted(chosen)]
            player.hand = [pool[card_id] for card_id in sorted(set(pool) - chosen)]
            events.append(ArrangementCommitted(player=seat, face_up=tuple(player.face_up)))

        self.setup = None
        self.phase = Phase.PLAY
        self.current_ply = 0
        self.current_player = self._select_opener()
        return tuple(events)

    def _select_opener(self) -> PlayerId:
        """Choose who opens, from the hands players hold after arranging.

        Ranks are searched with the specials last; the first rank present in
        anybody's hand decides, and ties among its holders break clockwise after
        the dealer.

        Returns:
            The opening player.

        Raises:
            StateInvariantError: If no seat holds any card, which cannot happen
                after a valid deal.
        """
        order = dealing_order(self.seat_order, self.dealer)
        for rank in _OPENING_RANK_ORDER:
            for seat in order:
                if any(card.rank is rank for card in self.players[seat].hand):
                    return seat
        raise StateInvariantError("No player holds a card after arrangement")

    def _resolve_play(self, actor: PlayerId, move: Move) -> tuple[ObservedEvent, ...]:
        """Resolve one PLAY decision completely, in physical order.

        The chain is fixed: move the cards, resolve a burn or the rank's effect
        (or the pickup that replaces both), replenish the actor's hand, count the
        ply, then either end the game or schedule the actor chosen earlier. The
        win check deliberately runs after replenishment, so shedding a last hand
        card while the deck can still refill does not finish anybody, and a
        burning last card wins instead of granting the retained turn.

        Args:
            actor: The deciding player.
            move: The already validated decision.

        Returns:
            The full events in resolution order, identities intact.

        Raises:
            StateInvariantError: If a non-PLAY move reaches here, or the actor
                cannot supply the batch their own legal move named.
        """
        player = self.players[actor]
        events: list[ObservedEvent] = []

        match move:
            case Play(source=source, rank=rank, count=count):
                batch = self._take_batch(actor, source, rank, count)
                self.discard_pile.extend(batch)
                events.append(CardsPlayed(player=actor, source=source, cards=batch))
                next_actor = self._resolve_pile(actor, rank, count, events)
            case Reveal(slot=slot):
                # Legality was decided before the reveal, so the pre-reveal
                # constraint is still the one this card must satisfy.
                card = player.face_down.pop(slot)
                playable = can_play_rank(card.rank, self.constraint)
                events.append(CardRevealed(player=actor, slot=slot, card=card, playable=playable))
                if playable:
                    self.discard_pile.append(card)
                    next_actor = self._resolve_pile(actor, card.rank, 1, events)
                else:
                    self._collect_pile(actor, extra=card, events=events)
                    next_actor = self._next_seat(actor)
            case PickUp():
                self._collect_pile(actor, extra=None, events=events)
                next_actor = self._next_seat(actor)
            case _:
                raise StateInvariantError(f"PLAY resolved a non-play move {move!r}")

        events.extend(self._refill(actor))
        self.current_ply += 1
        if player.remaining_count:
            self.current_player = next_actor
        else:
            outcome = Outcome(winner=actor)
            self.phase = Phase.FINISHED
            self.outcome = outcome
            self.current_player = None
            events.append(GameEnded(outcome=outcome))
        return tuple(events)

    def _take_batch(
        self, actor: PlayerId, source: Zone, rank: Rank, count: int
    ) -> tuple[Card, ...]:
        """Remove the physical cards a rank/count play names from one zone.

        Suits never affect the outcome, so the engine picks the batch by
        ascending card ID rather than asking the agent which physical cards it
        meant. That keeps the action space small and the transition reproducible.

        Args:
            actor: The deciding player.
            source: Zone to take from; hand or face-up.
            rank: Rank shared by the batch.
            count: How many cards to take.

        Returns:
            The removed cards in ascending card-ID order, which is also the
            order they reach the pile.

        Raises:
            StateInvariantError: If the zone holds fewer cards of that rank than
                the validated move claims.
        """
        player = self.players[actor]
        zone = player.hand if source is Zone.HAND else player.face_up
        batch = _by_id(card for card in zone if card.rank is rank)[:count]
        if len(batch) != count:
            raise StateInvariantError(
                f"Player {actor} cannot supply {count} {rank.name} cards from {source.value}"
            )
        for card in batch:
            zone.remove(card)
        return batch

    def _resolve_pile(
        self, actor: PlayerId, rank: Rank, count: int, events: list[ObservedEvent]
    ) -> PlayerId:
        """Burn the pile or apply the rank's effect, and pick the next actor.

        Args:
            actor: The deciding player, whose cards are already on the pile.
            rank: Rank just played or successfully revealed.
            count: Cards played in this single action; a reveal is one.
            events: Resolution log, appended to in place.

        Returns:
            Who should decide next if the game continues: the same actor after a
            burn, otherwise the next seat clockwise.
        """
        reason = _burn_reason(rank, count)
        if reason is None:
            self.constraint = _constraint_after(rank, self.constraint)
            return self._next_seat(actor)

        burned = tuple(self.discard_pile)
        self.burned_cards.extend(burned)
        self.discard_pile.clear()
        self.constraint = Unrestricted()
        events.append(PileBurned(player=actor, cards=burned, reason=reason))
        return actor

    def _collect_pile(
        self, actor: PlayerId, *, extra: Card | None, events: list[ObservedEvent]
    ) -> None:
        """Move the whole pile into the actor's hand and clear the constraint.

        Shared by a forced pickup and a failed blind reveal; the reveal passes
        the card it turned over, which joins the same transfer. The pile keeps
        its play order and the failed card follows it, so the hand's order stays
        deterministic even though nothing depends on it.

        Args:
            actor: The player taking the cards.
            extra: A failed reveal's card, or ``None`` for an ordinary pickup.
            events: Resolution log, appended to in place.
        """
        taken = tuple(self.discard_pile) if extra is None else (*self.discard_pile, extra)
        self.discard_pile.clear()
        self.players[actor].hand.extend(taken)
        self.constraint = Unrestricted()
        events.append(PilePickedUp(player=actor, cards=taken))

    def _refill(self, actor: PlayerId) -> tuple[ObservedEvent, ...]:
        """Replenish one hand to the profile's target while the deck lasts.

        Drawing is automatic rather than an agent decision, and never removes
        cards from a hand that already holds at least the target. Cards come off
        the end of the draw pile, the draw position.

        Args:
            actor: The player to replenish.

        Returns:
            One private draw event, or nothing when no card was drawn.
        """
        hand = self.players[actor].hand
        drawn: list[Card] = []
        while len(hand) < self.rules.refill_target and self.draw_pile:
            card = self.draw_pile.pop()
            drawn.append(card)
            hand.append(card)
        if not drawn:
            return ()
        return (CardsDrawn(player=actor, count=len(drawn), cards=tuple(drawn)),)

    def _next_seat(self, actor: PlayerId) -> PlayerId:
        """Return the seat one step clockwise from ``actor``.

        No seat is ever skipped: the profile has no eliminations, and the game
        ends the moment its first player runs out of cards, so a live game never
        contains a finished seat to pass over.

        Args:
            actor: The seat to advance from.

        Returns:
            The next seat in clockwise order.

        Raises:
            StateInvariantError: If the actor is not a seat in this game.
        """
        if actor not in self.seat_order:
            raise StateInvariantError(f"Actor {actor} is not a seat in {self.seat_order}")
        return self.seat_order[(self.seat_order.index(actor) + 1) % len(self.seat_order)]

    def _restore(self, snapshot: GameState) -> None:
        """Copy every field of ``snapshot`` back onto this object in place.

        Rebinding a local would not be enough: callers hold this object. Nested
        collections are replaced with fresh deep copies so later mutation cannot
        corrupt the snapshot they came from.

        Args:
            snapshot: The state captured before mutation.
        """
        restored = deepcopy(snapshot)
        for item in fields(GameState):
            setattr(self, item.name, getattr(restored, item.name))


def _arrangements(player: PlayerState) -> tuple[Move, ...]:
    """Enumerate every arrangement a player may submit.

    Args:
        player: The submitting player's cards.

    Returns:
        All 20 three-card choices from the six hand and face-up cards, each with
        ascending IDs, in deterministic combination order.
    """
    pool = sorted(card.id for card in (*player.hand, *player.face_up))
    return tuple(Arrange(choice) for choice in combinations(pool, 3))


def _batches(cards: Sequence[Card], zone: Zone, constraint: PlayConstraint) -> tuple[Move, ...]:
    """Enumerate the legal rank/count batches from one active zone.

    Args:
        cards: The cards available in that zone.
        zone: The zone they come from; hand or face-up.
        constraint: Restriction the pile currently imposes.

    Returns:
        Every legal batch, ordered by rank then by count, or a forced pickup
        when nothing in the zone is playable. Voluntary pickup does not exist in
        this profile, so the two are never offered together.
    """
    available = Counter(card.rank for card in cards)
    moves: list[Move] = []
    for rank in sorted(available):
        if not can_play_rank(rank, constraint):
            continue
        moves.extend(Play(zone, rank, count) for count in range(1, available[rank] + 1))
    if not moves:
        return (PickUp(),)
    return tuple(moves)


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
        config: Rules profile supplying the deck composition; validated by the
            deck builder.

    Returns:
        The shuffled deck. The end of the list is the next card to be drawn.

    Raises:
        ValueError: If the profile is not the fixed ``shed-v1`` profile.
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
    replay rebuilds the opening position from the recorded deck order instead of
    re-running the shuffle. :meth:`GameState.create` is the seeded entry point.

    Args:
        deck: The shuffled deck; must be a permutation of the canonical deck.
        player_count: Number of seats, within the profile's supported range.
        dealer: Dealing seat.
        config: Rules profile.

    Returns:
        A SETUP state with arrangements pending for every seat, an empty discard
        pile, no burned cards, and an unrestricted constraint.

    Raises:
        ValueError: If the profile, player count, or dealer is invalid, or the
            deck is not the canonical 54 cards.
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
    consistency, the empty-pile constraint rule, and the phase-specific setup,
    play, and finished conditions. Refill consistency is enforced separately,
    when the actor's moves are generated.

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
