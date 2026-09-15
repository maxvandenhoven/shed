from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum

from shed.engine import (
    DEFAULT_RULES,
    ORDINARY_RANKS,
    SUIT_ORDER,
    AtLeast,
    AtMost,
    Move,
    Phase,
    PickUp,
    Play,
    PlayConstraint,
    PlayerId,
    Rank,
    Reveal,
    RulesConfig,
    SlotId,
    StateInvariantError,
    Unrestricted,
    Zone,
    active_zone_for_counts,
    burn_reason,
    can_play_rank,
    constraint_after,
    legal_batches,
)

__all__ = [
    "ME",
    "OPPONENT",
    "SEATS",
    "Blocker",
    "CorrectState",
    "ObservationError",
    "ObservationEvent",
    "ObservedState",
    "PendingEntry",
    "PendingReason",
    "PickUpPile",
    "PlayCards",
    "RecordCards",
    "RevealFaceDown",
    "SeatObservation",
    "StatePatch",
    "SEAT_POSSESSIVE",
    "advice_blockers",
    "apply_event",
    "deck_copies",
    "describe_constraint",
    "join_game",
    "new_game",
    "observed_legal_moves",
    "seat_active_zone",
    "validate_observed",
]

ME: PlayerId = PlayerId(0)
"""The seat the phone belongs to; the only hand whose ranks are ever all known."""

OPPONENT: PlayerId = PlayerId(1)
"""The other seat at a two-player table."""

SEATS: tuple[PlayerId, PlayerId] = (ME, OPPONENT)
"""Both seats in turn order. The companion tracks two-player games only."""

SEAT_NAMES: Mapping[PlayerId, str] = {ME: "You", OPPONENT: "Opponent"}
"""How each seat is named in validation messages and in the action history."""

SEAT_POSSESSIVE: Mapping[PlayerId, str] = {ME: "your", OPPONENT: "your opponent's"}
"""How each seat is named where a message needs a possessive rather than a subject."""


class ObservationError(ValueError):
    """Raised when an observation cannot be reconciled with the tracked game.

    This is the companion's counterpart to
    :class:`~shed.engine.IllegalMoveError`: it means the entry was refused, not
    that the state is broken. The state the caller passed in is unchanged, so the
    message can be shown beside the operator's input for them to fix or to
    override with a correction.
    """


class PendingReason(Enum):
    """Why cards are in a hand before their ranks have been recorded."""

    DRAW = "draw"
    PICKUP = "pickup"


@dataclass(frozen=True, slots=True)
class PendingEntry:
    """Cards physically held whose ranks the operator still has to type in.

    The cards are already counted in :attr:`SeatObservation.hand_unknown` and
    already off the deck or off the pile, because the physical transfer has
    happened. Only their identities are outstanding, which is why a pending entry
    blocks advice but never blocks the count of who holds how many.

    Attributes:
        player: Whose hand the cards are in. Only :data:`ME` can ever be waiting:
            an opponent's draw is not merely unrecorded, it is unobservable, so it
            stays permanently unknown instead of pending.
        count: How many ranks are outstanding.
        reason: Whether they came off the deck or off the pile.
    """

    player: PlayerId
    count: int
    reason: PendingReason

    def __post_init__(self) -> None:
        """Check the entry describes at least one card for a real seat.

        Raises:
            ValueError: If the count is not positive or the seat is not a seat.
        """
        if self.count < 1:
            raise ValueError(f"PendingEntry.count must be positive, got {self.count}")
        if self.player not in SEATS:
            raise ValueError(f"PendingEntry.player {self.player} is not a seat")


@dataclass(frozen=True, slots=True)
class SeatObservation:
    """Everything observed about one seat's cards.

    Attributes:
        player: The seat described.
        hand_known: Ranks certainly in that hand, ascending. For :data:`ME` this
            is the hand itself; for the opponent it is only what a public
            transfer proved, the ranks of a pile they picked up, minus what they
            have since played.
        hand_unknown: Cards in that hand whose rank has not been observed.
        face_up: The public face-up set, ascending.
        face_down: Remaining face-down cards. They are a count and never a list:
            the slots are physically indistinguishable, so nothing distinguishes
            them in the model either.
    """

    player: PlayerId
    hand_known: tuple[Rank, ...]
    hand_unknown: int
    face_up: tuple[Rank, ...]
    face_down: int

    def __post_init__(self) -> None:
        """Sort the observed ranks so equal observations compare equal.

        Raises:
            ValueError: If the seat is not a seat or a count is negative.
        """
        if self.player not in SEATS:
            raise ValueError(f"SeatObservation.player {self.player} is not a seat")
        if self.hand_unknown < 0:
            raise ValueError(f"hand_unknown must not be negative, got {self.hand_unknown}")
        if self.face_down < 0:
            raise ValueError(f"face_down must not be negative, got {self.face_down}")
        object.__setattr__(self, "hand_known", tuple(sorted(self.hand_known)))
        object.__setattr__(self, "face_up", tuple(sorted(self.face_up)))

    @property
    def hand_count(self) -> int:
        """Cards in hand, known ranks and unknown ones together."""
        return len(self.hand_known) + self.hand_unknown

    @property
    def remaining_count(self) -> int:
        """Cards this seat still holds anywhere; zero means they have won."""
        return self.hand_count + len(self.face_up) + self.face_down

    @property
    def hand_fully_known(self) -> bool:
        """Whether every card in this hand has an observed rank."""
        return self.hand_unknown == 0


@dataclass(frozen=True, slots=True)
class ObservedState:
    """The whole tracked position of a physical game.

    Attributes:
        rules: The profile being played; only fixed ``standard`` is supported.
        seats: One observation per seat, indexed by :data:`SEATS` order.
        deck_count: Cards left in the draw pile, or ``None`` when the operator
            has not counted it. ``None`` is a real state and not a placeholder
            zero: without it neither a refill nor the active zone can be
            resolved, so the reducer refuses to guess and
            :func:`advice_blockers` asks for the count.
        pile: The discard pile bottom to top. ``None`` marks a card that is
            physically in the pile but whose rank was never seen, the ordinary
            case when joining a game in progress.
        burned: Cards removed from the game by burns, in the order they left,
            with ``None`` for the same reason.
        constraint: What the next ordinary rank must satisfy.
        to_act: Who has to decide, or ``None`` once the game is over.
        phase: :attr:`~shed.engine.Phase.PLAY` or
            :attr:`~shed.engine.Phase.FINISHED`. The companion never sees
            ``SETUP``: the hand/table swap happens physically before tracking
            starts, and the operator enters the position that swap produced.
        winner: Who emptied every zone first, once the game has finished.
        pending: Ranks still to be typed in, oldest first.
    """

    rules: RulesConfig
    seats: tuple[SeatObservation, SeatObservation]
    deck_count: int | None
    pile: tuple[Rank | None, ...]
    burned: tuple[Rank | None, ...]
    constraint: PlayConstraint
    to_act: PlayerId | None
    phase: Phase
    winner: PlayerId | None
    pending: tuple[PendingEntry, ...]

    def seat(self, player: PlayerId) -> SeatObservation:
        """Return one seat's observation.

        Args:
            player: The seat wanted.

        Returns:
            That seat's observation.

        Raises:
            ObservationError: If the player is not a seat at this table.
        """
        for seat in self.seats:
            if seat.player == player:
                return seat
        raise ObservationError(f"Player {player} is not a seat at this table")

    def other(self, player: PlayerId) -> PlayerId:
        """Return the opposing seat.

        Args:
            player: One seat.

        Returns:
            The other one.

        Raises:
            ObservationError: If the player is not a seat at this table.
        """
        if player == ME:
            return OPPONENT
        if player == OPPONENT:
            return ME
        raise ObservationError(f"Player {player} is not a seat at this table")

    def with_seat(self, seat: SeatObservation) -> ObservedState:
        """Return a copy with one seat's observation replaced.

        Args:
            seat: The replacement observation; it carries the seat it belongs to.

        Returns:
            A new state, leaving this one untouched.
        """
        seats = tuple(
            seat if existing.player == seat.player else existing for existing in self.seats
        )
        return replace(self, seats=(seats[0], seats[1]))

    @property
    def pile_size(self) -> int:
        """Cards in the discard pile, known ranks and unknown ones together."""
        return len(self.pile)

    @property
    def pile_top(self) -> Rank | None:
        """The topmost pile rank, or ``None`` for an empty pile or unseen card."""
        return self.pile[-1] if self.pile else None

    @property
    def is_finished(self) -> bool:
        """Whether the tracked game has ended."""
        return self.phase is Phase.FINISHED


def deck_copies(rules: RulesConfig = DEFAULT_RULES) -> Mapping[Rank, int]:
    """Return how many physical cards of each rank the deck holds.

    Args:
        rules: The profile supplying the joker count.

    Returns:
        Four of every ordinary rank and the profile's joker count, which is what
        bounds every rank tally the companion validates.
    """
    copies = {rank: len(SUIT_ORDER) for rank in ORDINARY_RANKS}
    copies[Rank.JOKER] = rules.joker_count
    return copies


def describe_constraint(constraint: PlayConstraint) -> str:
    """Describe a constraint in the words the phone interface uses.

    Args:
        constraint: The restriction in force.

    Returns:
        A short phrase naming the restriction.

    Raises:
        ValueError: If the constraint is not a known constraint type.
    """
    match constraint:
        case Unrestricted():
            return "no restriction"
        case AtLeast(rank=minimum):
            return f"at least {RANK_TEXT[minimum]}"
        case AtMost(rank=maximum):
            return f"at most {RANK_TEXT[maximum]} (a seven is in force)"
    raise ValueError(f"Unknown play constraint {constraint!r}")


RANK_TEXT: Mapping[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
    Rank.JOKER: "JK",
}
"""Short rank labels, matching ``shed.cli.RANK_TEXT``.

Duplicated rather than imported: :mod:`shed.cli` pulls in the match runner and
the replay codec, and the companion deliberately depends on the engine and the
agents alone. The spellings are kept identical so the phone, the console log, and
this module's messages all name a card the same way.
"""


def validate_observed(state: ObservedState) -> None:
    """Check that a tracked position could describe a real table.

    The checks are the ones a physical game makes available: nothing is negative,
    no rank appears more often than the deck holds, the 54 cards are all somewhere
    when the deck has been counted, an empty pile carries no restriction, and the
    phase agrees with who is holding cards. The first four catch the mistakes
    that actually happen at a table, a miscount, a rank typed twice, a pile
    entered without its unknown cards, which is why they run after every event
    rather than only at setup.

    Args:
        state: The position to check.

    Raises:
        ObservationError: If the position cannot describe a real table. The
            message names the specific disagreement so the interface can repeat
            it verbatim.
    """
    state.rules.validate()
    if tuple(seat.player for seat in state.seats) != SEATS:
        raise ObservationError("A tracked game needs exactly the two seats, you and your opponent")
    if state.deck_count is not None and state.deck_count < 0:
        raise ObservationError(f"The deck count cannot be negative, got {state.deck_count}")

    copies = deck_copies(state.rules)
    seen: Counter[Rank] = Counter()
    for seat in state.seats:
        seen.update(seat.hand_known)
        seen.update(seat.face_up)
    seen.update(rank for rank in state.pile if rank is not None)
    seen.update(rank for rank in state.burned if rank is not None)
    for rank, count in sorted(seen.items()):
        if count > copies[rank]:
            raise ObservationError(
                f"{count} cards of rank {RANK_TEXT[rank]} are recorded, "
                f"but the deck only holds {copies[rank]}"
            )

    if state.deck_count is not None:
        total = (
            state.deck_count
            + state.pile_size
            + len(state.burned)
            + sum(seat.remaining_count for seat in state.seats)
        )
        if total != state.rules.deck_size:
            raise ObservationError(
                f"The recorded cards add up to {total}, not the "
                f"{state.rules.deck_size} in the deck; recount the deck, the pile, "
                "or the burned cards"
            )

    if not state.pile and not isinstance(state.constraint, Unrestricted):
        raise ObservationError("An empty pile cannot carry a restriction")

    my_pending = sum(entry.count for entry in state.pending)
    if any(entry.player != ME for entry in state.pending):
        raise ObservationError("Only your own cards can be waiting to be recorded")
    if my_pending != state.seat(ME).hand_unknown:
        raise ObservationError(
            f"{state.seat(ME).hand_unknown} of your cards have no recorded rank but "
            f"{my_pending} are queued for entry"
        )

    match state.phase:
        case Phase.PLAY:
            if state.to_act is None:
                raise ObservationError("A live game needs somebody to act")
            if state.winner is not None:
                raise ObservationError("A live game cannot have a winner")
            if not state.seat(state.to_act).remaining_count:
                raise ObservationError(
                    f"{SEAT_NAMES[state.to_act]} holds no cards but is still to act"
                )
        case Phase.FINISHED:
            if state.to_act is not None:
                raise ObservationError("A finished game cannot also be somebody's turn")
            if state.winner is None:
                raise ObservationError("A finished game needs a winner")
            if state.seat(state.winner).remaining_count:
                raise ObservationError("The winner cannot still hold cards")
        case Phase.SETUP:
            raise ObservationError(
                "The companion tracks play only; enter the position the hand/table swap produced"
            )


def seat_active_zone(state: ObservedState, player: PlayerId) -> Zone | None:
    """Return the zone a seat must play from, using the engine's own ordering.

    Args:
        state: The tracked position.
        player: The seat asked about.

    Returns:
        The zone, or ``None`` when that seat holds nothing anywhere.

    Raises:
        ObservationError: If the deck has not been counted, which leaves the
            hand-empty case undecidable, or if the position owes a refill the
            reducer should already have resolved.
    """
    if state.deck_count is None:
        raise ObservationError(
            "The remaining deck count is unknown, so it is not clear whether "
            "an empty hand refills or the table comes into play"
        )
    seat = state.seat(player)
    try:
        return active_zone_for_counts(
            seat.hand_count, len(seat.face_up), seat.face_down, state.deck_count
        )
    except StateInvariantError as error:  # An unrefilled hand; recorded state is off.
        raise ObservationError(
            f"{SEAT_NAMES[player]} has an empty hand while {state.deck_count} cards "
            "are still in the deck; record the refill or correct the deck count"
        ) from error


def _zone_tally(seat: SeatObservation, zone: Zone) -> tuple[Counter[Rank], int]:
    """Return the observed ranks in one zone and how many ranks were not observed.

    Args:
        seat: The seat holding the zone.
        zone: The zone to tally; hand or face-up.

    Returns:
        A tally of the ranks known to be there, and the number of cards in the
        zone whose rank is unknown. A face-up set is public, so its unknown count
        is always zero.

    Raises:
        ObservationError: If asked to tally the face-down zone, which holds no
            observed ranks at all.
    """
    if zone is Zone.HAND:
        return Counter(seat.hand_known), seat.hand_unknown
    if zone is Zone.FACE_UP:
        return Counter(seat.face_up), 0
    raise ObservationError("Face-down cards have no observed ranks to tally")


def observed_legal_moves(state: ObservedState, player: PlayerId) -> tuple[Move, ...]:
    """Generate the moves a seat may make, from observed information only.

    The generation is the engine's: :func:`~shed.engine.legal_batches` for a hand
    or face-up batch, one :class:`~shed.engine.Reveal` per remaining face-down
    card, and nothing once the game is over. Only the *input* is weaker, and it is
    weaker in one direction that matters: an opponent's hand is partly unknown, so
    the batches generated for that seat are the ones we can prove they hold, never
    a guess at the rest.

    Args:
        state: The tracked position.
        player: The seat whose options are wanted.

    Returns:
        The moves that seat may make in the position as observed. Slot
        identifiers on reveals are positional bookkeeping: face-down cards are
        indistinguishable, so every reveal is the same action.

    Raises:
        ObservationError: If the position cannot answer the question, the game
            is over, the deck has not been counted, or the seat holds nothing.
    """
    if state.is_finished:
        return ()
    zone = seat_active_zone(state, player)
    if zone is None:
        raise ObservationError(f"{SEAT_NAMES[player]} holds no cards anywhere")
    seat = state.seat(player)
    if zone is Zone.FACE_DOWN:
        return tuple(Reveal(SlotId(index)) for index in range(seat.face_down))
    tally, _ = _zone_tally(seat, zone)
    return legal_batches(tally, zone, state.constraint)


@dataclass(frozen=True, slots=True)
class Blocker:
    """One reason the companion will not offer advice yet.

    Attributes:
        code: Stable machine-readable reason, for the interface to branch on.
        message: What the operator has to record, in their own terms.
    """

    code: str
    message: str


def advice_blockers(state: ObservedState) -> tuple[Blocker, ...]:
    """List what stops the companion recommending a move, in reading order.

    The companion refuses to advise on a guess. Every blocker here names a
    concrete observation to record instead, which is what the recommendation panel
    shows in place of a move.

    Args:
        state: The tracked position.

    Returns:
        Every current blocker, or an empty tuple when a recommendation can be
        built.
    """
    blockers: list[Blocker] = []
    if state.is_finished:
        winner = state.winner
        name = SEAT_NAMES[winner] if winner is not None else "Nobody"
        blockers.append(Blocker("finished", f"The game is over. {name} went out."))
        return tuple(blockers)
    if state.deck_count is None:
        blockers.append(
            Blocker(
                "deck_unknown",
                "Count the cards left in the deck and record it under Correct state.",
            )
        )
    for entry in state.pending:
        source = "drew" if entry.reason is PendingReason.DRAW else "picked up"
        blockers.append(
            Blocker(
                f"pending_{entry.reason.value}",
                f"Enter the {entry.count} card{'' if entry.count == 1 else 's'} you {source}.",
            )
        )
    if state.to_act != ME:
        blockers.append(Blocker("not_my_turn", "It is your opponent's turn. Record what they do."))
    if not state.seat(ME).hand_fully_known and not state.pending:
        blockers.append(
            Blocker(
                "hand_unknown",
                f"{state.seat(ME).hand_unknown} of your own cards have no recorded "
                "rank; fix your hand under Correct state.",
            )
        )
    return tuple(blockers)


@dataclass(frozen=True, slots=True)
class PlayCards:
    """Somebody played a batch of one rank from whichever zone is theirs to use.

    The source zone is not recorded because it is derived: the profile's
    active-zone ordering fixes it from the counts, so recording it would be a
    second version of the same fact.

    Attributes:
        player: Who played.
        rank: The rank every card in the batch shares.
        count: How many cards went onto the pile in this one action.
    """

    player: PlayerId
    rank: Rank
    count: int


@dataclass(frozen=True, slots=True)
class PickUpPile:
    """Somebody took the whole pile into their hand.

    Attributes:
        player: Who picked up.
    """

    player: PlayerId


@dataclass(frozen=True, slots=True)
class RevealFaceDown:
    """Somebody turned over a face-down card, and it turned out to be this rank.

    The rank is part of the event rather than something resolved afterwards: the
    card is face up on the table the moment it is turned, and its legality can
    only be judged against the constraint that was in force *before* it was
    revealed, so the rank has to arrive with the action.

    Attributes:
        player: Who revealed.
        rank: The rank that came up.
    """

    player: PlayerId
    rank: Rank


@dataclass(frozen=True, slots=True)
class RecordCards:
    """The ranks of cards already counted into my hand but not yet identified.

    Resolves the oldest :class:`PendingEntry`: the cards were drawn or picked up
    when the move was recorded, so this event changes no count, only what is
    known.

    Attributes:
        ranks: The ranks, in any order; exactly as many as the pending entry owes.
    """

    ranks: tuple[Rank, ...]


@dataclass(frozen=True, slots=True)
class StatePatch:
    """The fields a correction may set. Every field left ``None`` is untouched.

    Attributes:
        my_hand: My hand, replacing it outright. Because my own hand is knowable
            in full, setting it also clears every outstanding entry for it: there
            is nothing left to identify once the ranks are given.
        my_face_up: My face-up set.
        my_face_down: How many face-down cards I still have.
        opponent_hand_known: Ranks proven to be in the opponent's hand.
        opponent_hand_unknown: How many of their cards have no observed rank.
        opponent_face_up: Their face-up set.
        opponent_face_down: How many face-down cards they still have.
        deck_count: Cards left in the deck.
        deck_unknown: Set the deck count back to uncounted. ``deck_count`` wins if
            both are given.
        pile: The pile bottom to top, ``None`` for a card whose rank is not known.
        burned_count: How many cards have been burned out of the game. Burned
            identities are not asked for; the count is what the card accounting
            needs.
        constraint: The restriction in force.
        to_act: Whose turn it is. Setting it also reopens a game recorded as
            finished, which is the way back from a win entered by mistake.
    """

    my_hand: tuple[Rank, ...] | None = None
    my_face_up: tuple[Rank, ...] | None = None
    my_face_down: int | None = None
    opponent_hand_known: tuple[Rank, ...] | None = None
    opponent_hand_unknown: int | None = None
    opponent_face_up: tuple[Rank, ...] | None = None
    opponent_face_down: int | None = None
    deck_count: int | None = None
    deck_unknown: bool = False
    pile: tuple[Rank | None, ...] | None = None
    burned_count: int | None = None
    constraint: PlayConstraint | None = None
    to_act: PlayerId | None = None

    @property
    def is_empty(self) -> bool:
        """Whether this patch would change nothing at all."""
        return self == StatePatch()


@dataclass(frozen=True, slots=True)
class CorrectState:
    """An explicit correction to the tracked position, recorded as an event.

    A correction is never a silent rewrite: it joins the log like any other
    observation, so it replays, it appears in the history, and Undo removes it.
    That is deliberate, the state the companion shows is always the fold of a
    log the operator can read back.

    Attributes:
        patch: The fields to set.
        note: Why, in the operator's words; carried into the history line.
    """

    patch: StatePatch
    note: str = ""


type ObservationEvent = PlayCards | PickUpPile | RevealFaceDown | RecordCards | CorrectState


def apply_event(state: ObservedState, event: ObservationEvent) -> ObservedState:
    """Fold one observation into the tracked position.

    Args:
        state: The position before the observation.
        event: What was observed.

    Returns:
        A new position. The state passed in is frozen and is never modified, so a
        rejected event leaves the caller exactly where it was.

    Raises:
        ObservationError: If the observation contradicts the tracked position, or
            if the position is missing something the observation needs.
    """
    match event:
        case PlayCards(player=player, rank=rank, count=count):
            result = _apply_play(state, player, rank, count)
        case PickUpPile(player=player):
            result = _apply_pickup(state, player)
        case RevealFaceDown(player=player, rank=rank):
            result = _apply_reveal(state, player, rank)
        case RecordCards(ranks=ranks):
            result = _apply_record(state, ranks)
        case CorrectState(patch=patch):
            result = _apply_correction(state, patch)
    validate_observed(result)
    return result


def _require_turn(state: ObservedState, player: PlayerId) -> None:
    """Refuse a table action that the position cannot accept yet.

    Args:
        state: The position before the action.
        player: Who the action is attributed to.

    Raises:
        ObservationError: If the game is over, ranks are still owed, or it is the
            other seat's turn.
    """
    if state.is_finished:
        raise ObservationError(
            "The game is recorded as finished; undo the last entry or correct whose turn it is"
        )
    if state.pending:
        entry = state.pending[0]
        source = "drew" if entry.reason is PendingReason.DRAW else "picked up"
        raise ObservationError(
            f"Record the {entry.count} card{'' if entry.count == 1 else 's'} you "
            f"{source} before anything else happens"
        )
    if state.to_act != player:
        whose = "nobody's" if state.to_act is None else SEAT_POSSESSIVE[state.to_act]
        raise ObservationError(f"It is {whose} turn to act, not {SEAT_POSSESSIVE[player]}")


def _spend_from_hand(seat: SeatObservation, rank: Rank, count: int) -> SeatObservation:
    """Remove a batch from one hand, using observed copies before unknown ones.

    Spending the known copies first is the conservative reading of what was seen:
    if we knew the opponent held two sevens and they play one, one seven is still
    proven to be there. It never invents the reverse, a rank we never observed is
    taken out of the unknown count, not conjured into the known ranks and back out
    again.

    Args:
        seat: The seat playing.
        rank: The rank played.
        count: How many cards.

    Returns:
        The seat with the batch removed.

    Raises:
        ObservationError: If the hand cannot supply that many cards.
    """
    known = list(seat.hand_known)
    taken = 0
    while taken < count and rank in known:
        known.remove(rank)
        taken += 1
    outstanding = count - taken
    if outstanding > seat.hand_unknown:
        raise ObservationError(
            f"{SEAT_NAMES[seat.player]} cannot play {count} x {RANK_TEXT[rank]} from a "
            f"hand of {seat.hand_count} card{'' if seat.hand_count == 1 else 's'}"
        )
    return replace(seat, hand_known=tuple(known), hand_unknown=seat.hand_unknown - outstanding)


def _spend_from_face_up(seat: SeatObservation, rank: Rank, count: int) -> SeatObservation:
    """Remove a batch from one face-up set.

    Args:
        seat: The seat playing.
        rank: The rank played.
        count: How many cards.

    Returns:
        The seat with the batch removed.

    Raises:
        ObservationError: If the face-up set does not hold that many of the rank.
            A face-up set is public, so this is a straight contradiction rather
            than a limit of what was observed.
    """
    face_up = list(seat.face_up)
    if face_up.count(rank) < count:
        raise ObservationError(
            f"{SEAT_NAMES[seat.player]} has {face_up.count(rank)} x "
            f"{RANK_TEXT[rank]} face up, not {count}"
        )
    for _ in range(count):
        face_up.remove(rank)
    return replace(seat, face_up=tuple(face_up))


def _take_pile(state: ObservedState, player: PlayerId, extra: Rank | None) -> ObservedState:
    """Move the pile, and any card that failed a reveal, into one hand.

    Known pile ranks join the taker's known ranks, everybody at the table watched
    them go down, and unknown ones raise the taker's unknown count. For my own
    hand those unknown cards become a pending entry, because I can see them and
    will type them in; for the opponent's they simply stay unknown.

    Args:
        state: The position before the transfer.
        player: Who takes the cards.
        extra: The rank of a failed reveal joining the same transfer, or ``None``.

    Returns:
        The position with the pile transferred and the restriction cleared.
    """
    seat = state.seat(player)
    known = [rank for rank in state.pile if rank is not None]
    unknown = sum(1 for rank in state.pile if rank is None)
    if extra is not None:
        known.append(extra)
    seat = replace(
        seat,
        hand_known=(*seat.hand_known, *known),
        hand_unknown=seat.hand_unknown + unknown,
    )
    moved = state.with_seat(seat)
    pending = moved.pending
    if unknown and player == ME:
        pending = (*pending, PendingEntry(ME, unknown, PendingReason.PICKUP))
    return replace(moved, pile=(), constraint=Unrestricted(), pending=pending)


def _resolve_pile(
    state: ObservedState, player: PlayerId, rank: Rank, count: int
) -> tuple[ObservedState, PlayerId]:
    """Burn the pile or apply the rank's effect, and say who acts next.

    Both rules come straight from the engine: :func:`~shed.engine.burn_reason`
    decides whether the pile goes, and :func:`~shed.engine.constraint_after`
    decides what it leaves behind when it stays.

    Args:
        state: The position with the cards already on the pile.
        player: Who played them.
        rank: The rank played or successfully revealed.
        count: Cards in that single action; a reveal is one.

    Returns:
        The position after the pile resolved, and the seat to act if the game
        continues, the same seat after a burn, the other one otherwise.
    """
    if burn_reason(rank, count) is None:
        return replace(state, constraint=constraint_after(rank, state.constraint)), state.other(
            player
        )
    burned = (*state.burned, *state.pile)
    return replace(state, pile=(), burned=burned, constraint=Unrestricted()), player


def _refill(state: ObservedState, player: PlayerId) -> ObservedState:
    """Draw one hand back up to the profile's target while the deck lasts.

    The cards leave the deck and enter the hand immediately, because physically
    they already have. What is outstanding is only their identity: mine become a
    pending entry to type in, and the opponent's become permanently unknown, since
    nobody at the table saw them.

    Args:
        state: The position after the action resolved.
        player: Who refills.

    Returns:
        The position after the draw.

    Raises:
        ObservationError: If the deck has not been counted, so the draw size
            cannot be determined.
    """
    if state.deck_count is None:
        raise ObservationError(
            "The remaining deck count is unknown, so the size of the refill "
            "cannot be determined; record the deck count first"
        )
    seat = state.seat(player)
    drawn = min(max(0, state.rules.refill_target - seat.hand_count), state.deck_count)
    if not drawn:
        return state
    refilled = state.with_seat(replace(seat, hand_unknown=seat.hand_unknown + drawn))
    pending = refilled.pending
    if player == ME:
        pending = (*pending, PendingEntry(ME, drawn, PendingReason.DRAW))
    return replace(refilled, deck_count=state.deck_count - drawn, pending=pending)


def _finish_turn(state: ObservedState, player: PlayerId, next_actor: PlayerId) -> ObservedState:
    """Count the win and schedule whoever acts next.

    The order is the engine's: the win is checked after replenishment, so shedding
    a last hand card while the deck can still refill wins nobody the game, and a
    burn that empties a seat wins instead of granting the retained turn.

    Args:
        state: The position after the action and the refill.
        player: Who just acted.
        next_actor: Who would act next if the game continues.

    Returns:
        The finished position, or the live one with the next seat scheduled.
    """
    if state.seat(player).remaining_count:
        return replace(state, to_act=next_actor)
    return replace(state, phase=Phase.FINISHED, winner=player, to_act=None)


def _apply_play(state: ObservedState, player: PlayerId, rank: Rank, count: int) -> ObservedState:
    """Resolve a batch played from a hand or a face-up set.

    Args:
        state: The position before the play.
        player: Who played.
        rank: The rank played.
        count: How many cards.

    Returns:
        The position after the whole chain: transfer, burn or effect, refill, win
        check, and the next seat.

    Raises:
        ObservationError: If the batch is empty, the rank cannot go on this pile,
            the seat is playing from the wrong zone, or the zone cannot supply the
            cards.
    """
    _require_turn(state, player)
    if count < 1:
        raise ObservationError("A play has to name at least one card")
    zone = seat_active_zone(state, player)
    if zone is None:
        raise ObservationError(f"{SEAT_NAMES[player]} holds no cards anywhere")
    if zone is Zone.FACE_DOWN:
        raise ObservationError(
            f"{SEAT_NAMES[player]} is down to face-down cards; record the reveal and "
            "the rank that came up"
        )
    if not can_play_rank(rank, state.constraint):
        raise ObservationError(
            f"{RANK_TEXT[rank]} cannot be played while the pile demands "
            f"{describe_constraint(state.constraint)}"
        )
    seat = state.seat(player)
    if zone is Zone.HAND:
        seat = _spend_from_hand(seat, rank, count)
    else:
        seat = _spend_from_face_up(seat, rank, count)
    played = replace(state.with_seat(seat), pile=(*state.pile, *(rank,) * count))
    resolved, next_actor = _resolve_pile(played, player, rank, count)
    return _finish_turn(_refill(resolved, player), player, next_actor)


def _apply_pickup(state: ObservedState, player: PlayerId) -> ObservedState:
    """Resolve somebody taking the pile.

    Args:
        state: The position before the pickup.
        player: Who picked up.

    Returns:
        The position after the transfer, the refill, and the next seat.

    Raises:
        ObservationError: If the seat is in the face-down phase, holds nothing, or
            is known to hold a card it could have played. ``standard`` has no
            voluntary pickup, and this is checked against the ranks actually
            observed: a hand with unknown cards is never assumed to have been
            playable, only one whose observed ranks prove it.
    """
    _require_turn(state, player)
    zone = seat_active_zone(state, player)
    if zone is None:
        raise ObservationError(f"{SEAT_NAMES[player]} holds no cards anywhere")
    if zone is Zone.FACE_DOWN:
        raise ObservationError(
            f"{SEAT_NAMES[player]} is down to face-down cards; a reveal that fails "
            "picks the pile up by itself"
        )
    tally, _ = _zone_tally(state.seat(player), zone)
    provable = legal_batches(tally, zone, state.constraint)
    if provable != (PickUp(),):
        playable = ", ".join(
            sorted({RANK_TEXT[move.rank] for move in provable if isinstance(move, Play)})
        )
        raise ObservationError(
            f"{SEAT_NAMES[player]} can play {playable} on this pile, and there is "
            "no voluntary pickup. Fix the recorded cards if they are wrong."
        )
    taken = _take_pile(state, player, extra=None)
    return _finish_turn(_refill(taken, player), player, state.other(player))


def _apply_reveal(state: ObservedState, player: PlayerId, rank: Rank) -> ObservedState:
    """Resolve a face-down card being turned over.

    Legality is judged against the constraint in force *before* the reveal, as the
    profile requires, so a card that cannot go down takes the pile with it.

    Args:
        state: The position before the reveal.
        player: Who revealed.
        rank: The rank that came up.

    Returns:
        The position after the reveal resolved either way.

    Raises:
        ObservationError: If that seat is not playing from its face-down cards, or
            has none left.
    """
    _require_turn(state, player)
    zone = seat_active_zone(state, player)
    if zone is not Zone.FACE_DOWN:
        raise ObservationError(
            f"{SEAT_NAMES[player]} still has cards in hand or face up; a face-down "
            "card only comes into play once both are gone"
        )
    seat = state.seat(player)
    if not seat.face_down:
        raise ObservationError(f"{SEAT_NAMES[player]} has no face-down cards left")
    playable = can_play_rank(rank, state.constraint)
    turned = state.with_seat(replace(seat, face_down=seat.face_down - 1))
    if playable:
        onto_pile = replace(turned, pile=(*turned.pile, rank))
        resolved, next_actor = _resolve_pile(onto_pile, player, rank, 1)
    else:
        resolved = _take_pile(turned, player, extra=rank)
        next_actor = state.other(player)
    return _finish_turn(_refill(resolved, player), player, next_actor)


def _apply_record(state: ObservedState, ranks: tuple[Rank, ...]) -> ObservedState:
    """Attach ranks to cards already counted into my hand.

    Args:
        state: The position with an outstanding entry.
        ranks: The ranks of those cards.

    Returns:
        The position with those cards' ranks known and the entry cleared.

    Raises:
        ObservationError: If nothing is waiting, or the wrong number of ranks was
            given. The count is fixed by what was physically transferred, so a
            mismatch is a typing slip rather than new information.
    """
    if not state.pending:
        raise ObservationError("No cards are waiting to be recorded")
    entry = state.pending[0]
    if len(ranks) != entry.count:
        raise ObservationError(
            f"Record exactly {entry.count} rank{'' if entry.count == 1 else 's'}, got {len(ranks)}"
        )
    seat = state.seat(entry.player)
    updated = replace(
        seat,
        hand_known=(*seat.hand_known, *ranks),
        hand_unknown=seat.hand_unknown - entry.count,
    )
    return replace(state.with_seat(updated), pending=state.pending[1:])


def _apply_correction(state: ObservedState, patch: StatePatch) -> ObservedState:
    """Apply an explicit correction to the tracked position.

    Args:
        state: The position before the correction.
        patch: The fields to set.

    Returns:
        The corrected position. Termination is re-derived afterwards, so a
        correction that empties a seat finishes the game and one that puts cards
        back reopens it.

    Raises:
        ObservationError: If the patch changes nothing, or the result could not
            describe a real table. :func:`apply_event` runs the same validation,
            so an impossible correction is refused rather than stored.
    """
    if patch.is_empty:
        raise ObservationError("A correction has to change something")
    mine = state.seat(ME)
    theirs = state.seat(OPPONENT)
    pending = state.pending
    if patch.my_hand is not None:
        mine = replace(mine, hand_known=patch.my_hand, hand_unknown=0)
        pending = ()
    if patch.my_face_up is not None:
        mine = replace(mine, face_up=patch.my_face_up)
    if patch.my_face_down is not None:
        mine = replace(mine, face_down=patch.my_face_down)
    if patch.opponent_hand_known is not None:
        theirs = replace(theirs, hand_known=patch.opponent_hand_known)
    if patch.opponent_hand_unknown is not None:
        theirs = replace(theirs, hand_unknown=patch.opponent_hand_unknown)
    if patch.opponent_face_up is not None:
        theirs = replace(theirs, face_up=patch.opponent_face_up)
    if patch.opponent_face_down is not None:
        theirs = replace(theirs, face_down=patch.opponent_face_down)

    corrected = replace(state.with_seat(mine).with_seat(theirs), pending=pending)
    if patch.deck_count is not None:
        corrected = replace(corrected, deck_count=patch.deck_count)
    elif patch.deck_unknown:
        corrected = replace(corrected, deck_count=None)
    if patch.pile is not None:
        corrected = replace(corrected, pile=patch.pile)
    if patch.burned_count is not None:
        corrected = replace(corrected, burned=(None,) * patch.burned_count)
    if patch.constraint is not None:
        corrected = replace(corrected, constraint=patch.constraint)
    if patch.to_act is not None:
        corrected = replace(corrected, to_act=patch.to_act, phase=Phase.PLAY, winner=None)
    return _retire_finished_seats(corrected)


def _retire_finished_seats(state: ObservedState) -> ObservedState:
    """Re-derive whether a corrected position is over, and who won.

    Args:
        state: A position straight out of a correction.

    Returns:
        The same position, finished if a seat now holds nothing. The seat to act is
        preferred as the winner when both are empty, which can only happen in a
        position that a correction built by hand.
    """
    if state.is_finished:
        return state
    empty = [seat.player for seat in state.seats if not seat.remaining_count]
    if not empty:
        return state
    winner = state.to_act if state.to_act in empty else empty[0]
    return replace(state, phase=Phase.FINISHED, winner=winner, to_act=None)


def new_game(
    *,
    my_hand: tuple[Rank, ...],
    my_face_up: tuple[Rank, ...],
    opponent_face_up: tuple[Rank, ...],
    starting_player: PlayerId,
    rules: RulesConfig = DEFAULT_RULES,
) -> ObservedState:
    """Build the opening position of a game that is starting now.

    The companion joins after the physical hand/table swap, so the caller passes
    the arrangement that swap produced rather than a pre-swap deal. Everything else
    at the start of a game is fixed by the profile: three face-down cards each,
    three unknown cards in the opponent's hand, an empty pile, nothing burned, and
    a deck of whatever the deal left.

    Args:
        my_hand: My three hand ranks.
        my_face_up: My three face-up ranks.
        opponent_face_up: Their three face-up ranks.
        starting_player: Who actually played first at the table. The profile's
            opener rule reads hidden hands, so it cannot be applied from here,
            and it does not need to be, because the table already decided.
        rules: The profile; only fixed ``standard`` is supported.

    Returns:
        The opening position.

    Raises:
        ObservationError: If the counts do not match the profile's deal, the seat
            named is not a seat, or the ranks could not come from one deck.
    """
    if starting_player not in SEATS:
        raise ObservationError(f"{starting_player} is not a seat at a two-player table")
    expected_hand = rules.initial_hand_size
    expected_face_up = rules.initial_face_up_count
    if len(my_hand) != expected_hand:
        raise ObservationError(f"Enter your {expected_hand} hand cards, got {len(my_hand)}")
    for label, face_up in (("your", my_face_up), ("your opponent's", opponent_face_up)):
        if len(face_up) != expected_face_up:
            raise ObservationError(
                f"Enter {label} {expected_face_up} face-up cards, got {len(face_up)}"
            )
    face_down = rules.initial_face_down_count
    dealt = 2 * (expected_hand + expected_face_up + face_down)
    state = ObservedState(
        rules=rules,
        seats=(
            SeatObservation(ME, my_hand, 0, my_face_up, face_down),
            SeatObservation(OPPONENT, (), expected_hand, opponent_face_up, face_down),
        ),
        deck_count=rules.deck_size - dealt,
        pile=(),
        burned=(),
        constraint=Unrestricted(),
        to_act=starting_player,
        phase=Phase.PLAY,
        winner=None,
        pending=(),
    )
    validate_observed(state)
    return state


def join_game(
    *,
    my_hand: tuple[Rank, ...],
    my_face_up: tuple[Rank, ...],
    my_face_down: int,
    opponent_hand_count: int,
    opponent_hand_known: tuple[Rank, ...],
    opponent_face_up: tuple[Rank, ...],
    opponent_face_down: int,
    deck_count: int | None,
    pile: tuple[Rank | None, ...],
    constraint: PlayConstraint,
    to_act: PlayerId,
    rules: RulesConfig = DEFAULT_RULES,
) -> ObservedState:
    """Build a position for a game already under way.

    Nothing is inferred that the operator did not see. The burned collection is the
    one exception, and it is arithmetic rather than a guess: burned cards are
    exactly the ones the 54 cannot otherwise account for, so they are recorded as
    that many cards of unknown rank. When the deck has not been counted, that
    subtraction is unavailable, nothing is burned into the model, and
    :func:`advice_blockers` asks for the count instead.

    Args:
        my_hand: My hand ranks; I can always see my own hand.
        my_face_up: My remaining face-up ranks.
        my_face_down: My remaining face-down cards.
        opponent_hand_count: How many cards they are holding.
        opponent_hand_known: Ranks known to be among them, which is usually none.
        opponent_face_up: Their remaining face-up ranks.
        opponent_face_down: Their remaining face-down cards.
        deck_count: Cards left in the deck, or ``None`` if not counted.
        pile: The pile bottom to top, ``None`` for any card whose rank is unknown.
        constraint: The restriction in force, which the operator reads off the
            table rather than deriving from a history the companion never saw.
        to_act: Who is to act.
        rules: The profile; only fixed ``standard`` is supported.

    Returns:
        The joined position.

    Raises:
        ObservationError: If the counts cannot describe a real table, a hand
            smaller than its known ranks, a negative count, a rank recorded more
            often than the deck holds, or cards that do not add up to 54.
    """
    if to_act not in SEATS:
        raise ObservationError(f"{to_act} is not a seat at a two-player table")
    if opponent_hand_count < len(opponent_hand_known):
        raise ObservationError(
            f"You know {len(opponent_hand_known)} of your opponent's cards but say "
            f"they hold {opponent_hand_count}"
        )
    seats = (
        SeatObservation(ME, my_hand, 0, my_face_up, my_face_down),
        SeatObservation(
            OPPONENT,
            opponent_hand_known,
            opponent_hand_count - len(opponent_hand_known),
            opponent_face_up,
            opponent_face_down,
        ),
    )
    burned_count = 0
    if deck_count is not None:
        accounted = deck_count + len(pile) + sum(seat.remaining_count for seat in seats)
        burned_count = rules.deck_size - accounted
        if burned_count < 0:
            raise ObservationError(
                f"The cards entered add up to {accounted}, more than the "
                f"{rules.deck_size} in the deck; recount before joining"
            )
    state = ObservedState(
        rules=rules,
        seats=seats,
        deck_count=deck_count,
        pile=pile,
        burned=(None,) * burned_count,
        constraint=constraint,
        to_act=to_act,
        phase=Phase.PLAY,
        winner=None,
        pending=(),
    )
    validate_observed(state)
    return state
