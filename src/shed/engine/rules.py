"""Legality functions and the ``Ruleset`` orchestration for ``shed-v1``.

There is exactly one legality implementation, :func:`legal_moves`.
``PlayerView.get_legal_moves``, ``GameState.get_legal_moves``, and
``Ruleset.get_legal_moves`` are conveniences that all delegate to it, so the
three entry points can never disagree.

Milestone status: initialization, observation, legal-move generation, and the
SETUP transition are implemented. Ordinary PLAY resolution -- transferring a
batch, resolving reveals, burns, pickups, refills, and termination -- is the
remaining engine work; :meth:`Ruleset.apply_move` raises ``NotImplementedError``
for PLAY rather than pretending a decision resolved.
"""

from collections import Counter
from copy import deepcopy
from dataclasses import fields
from itertools import combinations

from shed.engine.events import (
    ArrangementCommitted,
    Decision,
    GameStarted,
    HandDealt,
    ObservedEvent,
    Transition,
    UndoRecord,
)
from shed.engine.state import (
    GameState,
    Outcome,
    PlayerView,
    build_view,
    deal_initial_state,
    dealing_order,
    derive_active_zone,
    public_player_state,
    shuffled_deck,
    validate_decision_boundary,
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
    Phase,
    PickUp,
    Play,
    PlayConstraint,
    PlayerId,
    Rank,
    Reveal,
    RulesConfig,
    StateInvariantError,
    Unrestricted,
    Zone,
)

__all__ = [
    "OPENING_RANK_ORDER",
    "Ruleset",
    "can_play_rank",
    "legal_moves",
    "select_opening_player",
]

ALWAYS_PLAYABLE: frozenset[Rank] = frozenset({Rank.TWO, Rank.NINE, Rank.TEN, Rank.JOKER})
"""Ranks exempt from the current constraint, checked before any comparison."""

OPENING_RANK_ORDER: tuple[Rank, ...] = (
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
        ValueError: If the constraint is not a known constraint type.
    """
    if rank in ALWAYS_PLAYABLE:
        return True
    match constraint:
        case Unrestricted():
            return True
        case AtLeast(rank=minimum):
            return rank >= minimum
        case AtMost(rank=maximum):
            return rank <= maximum
    raise ValueError(f"Unknown play constraint {constraint!r}")


def _setup_moves(view: PlayerView) -> tuple[Move, ...]:
    """Enumerate every arrangement the actor may submit.

    Args:
        view: The actor's observation during SETUP.

    Returns:
        All 20 three-card choices from the actor's six hand and face-up cards,
        each with ascending IDs, in deterministic combination order.
    """
    pool = sorted(card.id for card in (*view.hand, *view.me.face_up))
    return tuple(Arrange(choice) for choice in combinations(pool, 3))


def _play_moves(view: PlayerView) -> tuple[Move, ...]:
    """Enumerate the actor's legal decisions during PLAY.

    Args:
        view: The actor's observation during PLAY.

    Returns:
        Reveal moves for a blind actor, otherwise every legal rank/count batch
        from the single active zone, or a forced pickup when nothing is
        playable. Every live actor has at least one move.

    Raises:
        StateInvariantError: If a refill is still pending for the actor, or the
            actor holds no cards at all. Both mean the previous decision was
            never fully resolved.
    """
    me = view.me
    zone = derive_active_zone(
        len(view.hand), len(me.face_up), len(me.face_down_slots), view.draw_count
    )
    if zone is None:
        raise StateInvariantError(
            f"Actor {view.viewer} holds no cards; resolve termination "
            "before requesting another decision"
        )
    if zone is Zone.FACE_DOWN:
        # Every remaining slot is legal; hidden ranks are never inspected here.
        return tuple(Reveal(slot) for slot in sorted(me.face_down_slots))

    available = Counter(card.rank for card in (view.hand if zone is Zone.HAND else me.face_up))
    moves: list[Move] = []
    for rank in sorted(available):
        if not can_play_rank(rank, view.constraint):
            continue
        moves.extend(Play(zone, rank, count) for count in range(1, available[rank] + 1))
    if not moves:
        return (PickUp(),)
    return tuple(moves)


def legal_moves(view: PlayerView) -> tuple[Move, ...]:
    """Generate every legal move for the viewer, from public information only.

    This is the engine's single legality implementation. It reads nothing but
    the view, so a viewer's options can never depend on hidden assignments they
    cannot observe.

    Args:
        view: The observation to generate moves for.

    Returns:
        A deterministically ordered tuple of legal moves, empty when the game is
        finished or it is not this viewer's decision.

    Raises:
        StateInvariantError: If the state is not at a valid decision boundary.
    """
    if view.phase is Phase.FINISHED or view.current_player != view.viewer:
        return ()
    if view.phase is Phase.SETUP:
        return _setup_moves(view)
    return _play_moves(view)


def select_opening_player(state: GameState) -> PlayerId:
    """Choose who opens, from the hands players hold after arranging.

    Ranks are searched in :data:`OPENING_RANK_ORDER`; the first rank present in
    anybody's hand decides. Ties among holders of that rank break clockwise
    after the dealer.

    Args:
        state: State whose arrangements have already been committed.

    Returns:
        The opening player.

    Raises:
        StateInvariantError: If no seat holds any card, which cannot happen
            after a valid deal.
    """
    order = dealing_order(state.seat_order, state.dealer)
    for rank in OPENING_RANK_ORDER:
        for seat in order:
            if any(card.rank is rank for card in state.players[seat].hand):
                return seat
    raise StateInvariantError("No player holds a card after arrangement")


class Ruleset:
    """The rules of ``shed-v1``: initialization, observation, legality, undo.

    One concrete class owns the whole contract. Small module-level helpers cover
    shared legality; separate classes per phase, effect, or card would add
    indirection without adding behaviour.
    """

    def __init__(self, config: RulesConfig = DEFAULT_RULES) -> None:
        """Create a ruleset for the fixed profile.

        Args:
            config: Rules profile; only the unmodified ``shed-v1`` profile is
                accepted.

        Raises:
            ValueError: If the profile is unknown or any field was changed.
        """
        config.validate()
        self._config = config

    @property
    def config(self) -> RulesConfig:
        """The fixed rules profile this ruleset enforces."""
        return self._config

    def create_initial_state(
        self,
        player_count: int,
        *,
        seed: int,
        dealer: PlayerId = DEFAULT_DEALER,
    ) -> GameState:
        """Shuffle and deal a fresh game.

        Args:
            player_count: Number of seats, within the profile's range.
            seed: Deck seed for the dedicated shuffle generator. It is trusted
                metadata: never expose it to an agent.
            dealer: Dealing seat; deal, arrangement, and tie-break order all
                start clockwise after it.

        Returns:
            A SETUP state awaiting the first arrangement.

        Raises:
            ValueError: If the player count or dealer is invalid.
        """
        deck = shuffled_deck(seed=seed, config=self._config)
        return deal_initial_state(
            deck, player_count=player_count, dealer=dealer, config=self._config
        )

    def initial_events(self, state: GameState) -> tuple[ObservedEvent, ...]:
        """Describe the deal as full events.

        The public opening event records the face-up cards visible before any
        arrangement, which is what lets agents remember where those cards
        started. Hand deals are private events: filtering replaces their
        identities for every player but the recipient.

        Args:
            state: A freshly dealt state.

        Returns:
            The opening event followed by one hand deal per seat, in dealing
            order. Identities are intact; filter before showing them to agents.
        """
        order = dealing_order(state.seat_order, state.dealer)
        events: list[ObservedEvent] = [
            GameStarted(
                dealer=state.dealer,
                seat_order=state.seat_order,
                players=tuple(
                    public_player_state(seat, state.players[seat]) for seat in state.seat_order
                ),
            )
        ]
        for seat in order:
            hand = tuple(sorted(state.players[seat].hand, key=lambda card: card.id))
            events.append(HandDealt(player=seat, count=len(hand), cards=hand))
        return tuple(events)

    def observe(
        self,
        state: GameState,
        player: PlayerId,
        *,
        history: tuple[ObservedEvent, ...] = (),
    ) -> PlayerView:
        """Build one player's immutable observation.

        Args:
            state: Authoritative state; it is not modified.
            player: The viewing seat.
            history: That player's already-filtered history. Passing another
                player's unfiltered events would leak hidden information.

        Returns:
            An independent snapshot of what this player may know.

        Raises:
            StateInvariantError: If the player has no seat, or the state is not
                at a valid decision boundary.
        """
        return build_view(state, player, history=history)

    def get_legal_moves(self, view: PlayerView) -> tuple[Move, ...]:
        """Return the legal moves for an observation.

        Args:
            view: The observation to generate moves for.

        Returns:
            The same tuple :func:`legal_moves` produces; this is a convenience,
            not a second implementation.
        """
        return legal_moves(view)

    def apply_move(self, state: GameState, move: Move) -> Transition:
        """Validate and apply one decision, mutating ``state`` in place.

        Validation completes before any mutation, so a rejected move leaves the
        state exactly as it was.

        Args:
            state: Authoritative state to advance.
            move: The submitted decision.

        Returns:
            The transition: the applied decision, an undo snapshot, and the full
            events in physical resolution order.

        Raises:
            IllegalMoveError: If the move is out of phase or is not among the
                actor's legal moves.
            NotImplementedError: For every PLAY decision. Ordinary play
                resolution is the remaining engine work; the path is refused
                explicitly rather than reported as a successful transition.
            StateInvariantError: If the state is not at a valid decision
                boundary.
        """
        if state.phase is Phase.FINISHED:
            raise IllegalMoveError("The game has finished; no move can be applied")
        actor = state.current_player
        if actor is None:
            raise StateInvariantError("No actor is scheduled to decide")

        legal = state.get_legal_moves()
        if move not in legal:
            raise IllegalMoveError(
                f"{move!r} is not legal for player {actor} in phase {state.phase.value}"
            )
        canonical = legal[legal.index(move)]

        if state.phase is not Phase.SETUP:
            raise NotImplementedError(
                "PLAY resolution is not implemented yet: milestone 2 covers setup, "
                "observations, and legal-move generation only"
            )

        undo = UndoRecord(before=deepcopy(state))
        try:
            events = self._apply_arrangement(state, actor, canonical)
            # The postcondition is part of the transition: a state that fails it
            # must be rolled back too, not left half-applied behind an error.
            validate_decision_boundary(state)
        except Exception:
            _restore(state, undo.before)
            raise
        return Transition(decision=Decision(player=actor, move=canonical), undo=undo, events=events)

    def _apply_arrangement(
        self, state: GameState, actor: PlayerId, move: Move
    ) -> tuple[ObservedEvent, ...]:
        """Store one arrangement, committing every arrangement once all arrive.

        A stored arrangement changes no visible card: hands and face-up
        collections stay exactly as dealt until the collective commit, so nobody
        can react to an opponent's choice while still making their own.

        Args:
            state: State in SETUP; mutated in place.
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
        setup = state.setup
        if setup is None:
            raise StateInvariantError("SETUP phase without setup state")

        setup.submissions[actor] = move
        setup.pending.remove(actor)
        if setup.pending:
            state.current_player = setup.pending[0]
            return ()

        events: list[ObservedEvent] = []
        for seat in dealing_order(state.seat_order, state.dealer):
            arrangement = setup.submissions[seat]
            player = state.players[seat]
            pool: dict[CardId, Card] = {card.id: card for card in (*player.hand, *player.face_up)}
            chosen = set(arrangement.face_up_cards)
            if not chosen <= set(pool):
                raise StateInvariantError(f"Player {seat} arranged cards they do not own")
            player.face_up = [pool[card_id] for card_id in sorted(chosen)]
            player.hand = [pool[card_id] for card_id in sorted(set(pool) - chosen)]
            events.append(ArrangementCommitted(player=seat, face_up=tuple(player.face_up)))

        state.setup = None
        state.phase = Phase.PLAY
        state.current_ply = 0
        state.current_player = select_opening_player(state)
        return tuple(events)

    def undo_move(self, state: GameState, transition: Transition) -> None:
        """Restore the state captured before ``transition`` was applied.

        Fields are restored on the existing ``GameState`` object so every holder
        of the reference observes the rollback. The snapshot is copied again on
        the way back, keeping the record reusable if the state is mutated after
        the undo. Undo is LIFO on the originating state; arbitrary out-of-order
        undo is not supported.

        Args:
            state: The state the transition was applied to.
            transition: The transition to roll back.
        """
        _restore(state, transition.undo.before)

    def get_outcome(self, state: GameState) -> Outcome | None:
        """Return the result of a finished game.

        Args:
            state: State to inspect.

        Returns:
            The outcome, or ``None`` while the game is still running.
        """
        return state.outcome

    def is_finished(self, state: GameState) -> bool:
        """Report whether the game has ended.

        Args:
            state: State to inspect.

        Returns:
            ``True`` once the state is in the FINISHED phase.
        """
        return state.phase is Phase.FINISHED


def _restore(state: GameState, snapshot: GameState) -> None:
    """Copy every field of ``snapshot`` back onto ``state`` in place.

    Rebinding a local variable would not be enough: callers hold the original
    object. Nested collections are replaced with fresh deep copies so later
    mutation of the state cannot corrupt the snapshot it came from.

    Args:
        state: The live state to rewind.
        snapshot: The state captured before mutation.
    """
    restored = deepcopy(snapshot)
    for item in fields(GameState):
        setattr(state, item.name, getattr(restored, item.name))
