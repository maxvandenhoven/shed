"""The recoverable session document: a versioned start, a log, and what they fold to.

A session is the whole game, and it is *only* two things: the position the
operator entered when they started tracking, and every observation since, in
order. The position on screen is always the fold of that log, never a separately
maintained copy of it, which is what makes the recovery features fall out of the
design rather than being bolted on. Undo pops the last observation. A correction
appends one. The action history is the log printed. Export is the document written
out, and import is one read back.

The browser owns the document and saves it after every accepted observation; this
module is what reads it. So the server keeps no session state at all: restarting
Python, or losing it mid-game, costs nothing the browser cannot resend. That is
also why a document that no longer folds cleanly does not raise here.
:func:`derive` stops at the observation that failed, reports it, and hands back the
position it reached, so a session damaged by a bad import or a stale schema can
still be read, undone, and exported rather than being dead on arrival.

Every payload the phone renders comes from :func:`render`, which is a pure function
of the document.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from shed.companion.advice import Recommendation, describe_move, recommend
from shed.companion.codec import (
    RANK_BY_CODE,
    SEAT_CODE,
    CompanionDataError,
    decode_event,
    decode_state,
    encode_constraint,
    encode_event,
    encode_move,
    encode_state,
)
from shed.companion.observed import (
    ME,
    RANK_TEXT,
    SEAT_NAMES,
    Blocker,
    CorrectState,
    ObservationError,
    ObservationEvent,
    ObservedState,
    PendingReason,
    PickUpPile,
    PlayCards,
    RecordCards,
    RevealFaceDown,
    advice_blockers,
    apply_event,
    describe_constraint,
    observed_legal_moves,
    seat_active_zone,
    validate_observed,
)
from shed.engine import (
    Move,
    PlayerId,
    Rank,
    Reveal,
    Zone,
    can_play_rank,
)

__all__ = [
    "COMPANION_SCHEMA_VERSION",
    "Derivation",
    "Session",
    "decode_session",
    "derive",
    "describe_event",
    "encode_session",
    "render",
    "revision",
]

COMPANION_SCHEMA_VERSION = 1
"""Version of the session document this release reads and writes.

The document is what survives a refresh, a restart, and an exported backup, so it
is versioned separately from the replay schema in :mod:`shed.replay`: the two
describe different things and will not change together. A document from a future
version is refused with its number in the message rather than being read
optimistically.
"""


@dataclass(frozen=True, slots=True)
class Session:
    """One tracked game as a document.

    Attributes:
        schema_version: The document version; see
            :data:`COMPANION_SCHEMA_VERSION`.
        initial: The position the operator entered when tracking started -- a fresh
            deal after the physical swap, or a game already in progress.
        events: Every observation since, oldest first.
    """

    schema_version: int
    initial: ObservedState
    events: tuple[ObservationEvent, ...]

    def appended(self, event: ObservationEvent) -> Session:
        """Return this session with one more observation on the end.

        Args:
            event: The observation to append.

        Returns:
            A new session; this one is unchanged.
        """
        return Session(self.schema_version, self.initial, (*self.events, event))

    def undone(self) -> Session:
        """Return this session with its last observation removed.

        Returns:
            A new session. Undo is the log's own shape rather than an inverse
            operation, so it works identically on a play, a pickup, a recorded
            draw, and a correction.

        Raises:
            ObservationError: If there is nothing to undo, which the interface
                turns into a disabled button rather than an error.
        """
        if not self.events:
            raise ObservationError("There is nothing to undo")
        return Session(self.schema_version, self.initial, self.events[:-1])


def encode_session(session: Session) -> dict[str, Any]:
    """Encode a session for storage, for the wire, and for export.

    Args:
        session: The session.

    Returns:
        A JSON-ready document.
    """
    return {
        "schema_version": session.schema_version,
        "initial": encode_state(session.initial),
        "events": [encode_event(event) for event in session.events],
    }


def decode_session(value: object, where: str = "session") -> Session:
    """Decode a session document, refusing anything this release cannot read.

    Args:
        value: The document.
        where: Field path, for the message.

    Returns:
        The session. Whether its log still folds cleanly is :func:`derive`'s
        question, not this one.

    Raises:
        CompanionDataError: If the document is malformed or carries a schema
            version this release does not implement.
    """
    if not isinstance(value, dict):
        raise CompanionDataError(f"{where} must be an object")
    version = value.get("schema_version")
    if version != COMPANION_SCHEMA_VERSION:
        raise CompanionDataError(
            f"{where}.schema_version: this release reads version "
            f"{COMPANION_SCHEMA_VERSION}, got {version!r}"
        )
    events = value.get("events", [])
    if not isinstance(events, list):
        raise CompanionDataError(f"{where}.events must be an array")
    return Session(
        schema_version=COMPANION_SCHEMA_VERSION,
        initial=decode_state(value.get("initial"), f"{where}.initial"),
        events=tuple(
            decode_event(item, f"{where}.events[{index}]") for index, item in enumerate(events)
        ),
    )


def revision(session: Session) -> str:
    """Identify the exact content of a session in one short string.

    A recommendation is only about the position it was asked for. The interface
    tags every request with this value and drops any answer that comes back against
    a different one, so a slow reply cannot land on a table that has moved on.

    Args:
        session: The session.

    Returns:
        The event count and a digest of the canonical document, such as
        ``"12-9f3c1a7b5e2d4086"``. The count makes it readable; the digest makes it
        change whenever anything at all does, including a correction that leaves the
        length alone.
    """
    canonical = json.dumps(encode_session(session), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{len(session.events)}-{digest}"


def advice_seed(session: Session) -> int:
    """Derive the agent's tie-breaking seed from the session content.

    Seeding from the revision rather than from a clock makes a recommendation a
    function of the position: asking twice gives the same answer, and a screenshot
    of a suggestion can be reproduced from the exported document.

    Args:
        session: The session.

    Returns:
        A non-negative seed.
    """
    return int(revision(session).split("-", 1)[1], 16)


@dataclass(frozen=True, slots=True)
class Derivation:
    """The result of folding a session's log.

    Attributes:
        state: The position reached. When ``error`` is set this is the position
            *before* the observation that failed, so the interface still has
            something coherent to show and something to undo.
        applied: How many observations were folded in.
        history: One line per folded observation, oldest first.
        error: Why folding stopped, or ``None`` when the whole log applied.
    """

    state: ObservedState
    applied: int
    history: tuple[str, ...]
    error: str | None


def derive(session: Session) -> Derivation:
    """Fold a session's log into the position it describes.

    Args:
        session: The session.

    Returns:
        The derivation. A log that stops applying is reported rather than raised:
        the operator needs the damaged session on screen to undo or export it.

    Raises:
        ObservationError: If the *initial* position itself could not describe a real
            table. There is nothing to show or undo in that case, so it is the one
            failure that does raise.
    """
    validate_observed(session.initial)
    state = session.initial
    history: list[str] = []
    for index, event in enumerate(session.events):
        try:
            advanced = apply_event(state, event)
        except ObservationError as error:
            return Derivation(
                state=state,
                applied=index,
                history=tuple(history),
                error=f"Observation {index + 1} no longer applies: {error}",
            )
        history.append(describe_event(event, state))
        state = advanced
    return Derivation(state=state, applied=len(session.events), history=tuple(history), error=None)


def _rank_list(ranks: tuple[Rank, ...]) -> str:
    """Spell a group of ranks the way the interface does.

    Args:
        ranks: The ranks.

    Returns:
        The labels separated by spaces, ascending, or ``"nothing"`` when empty.
    """
    return " ".join(RANK_TEXT[rank] for rank in sorted(ranks)) if ranks else "nothing"


def describe_event(event: ObservationEvent, before: ObservedState) -> str:
    """Describe one observation as a history line.

    The line is written against the position the observation was folded into, so it
    can say what the entry actually did -- how big the pile that was picked up was,
    whether a revealed card went down or came back.

    Args:
        event: The observation.
        before: The position it was applied to.

    Returns:
        One sentence for the action history.

    Raises:
        ValueError: If the observation is not a known type.
    """
    match event:
        case PlayCards(player=player, rank=rank, count=count):
            zone = ""
            try:
                if seat_active_zone(before, player) is Zone.FACE_UP:
                    zone = " from the table"
            except ObservationError:  # An unresolvable zone still deserves a line.
                zone = ""
            return f"{SEAT_NAMES[player]} played {count} x {RANK_TEXT[rank]}{zone}"
        case PickUpPile(player=player):
            size = before.pile_size
            return (
                f"{SEAT_NAMES[player]} picked up the pile ({size} card{'' if size == 1 else 's'})"
            )
        case RevealFaceDown(player=player, rank=rank):
            if can_play_rank(rank, before.constraint):
                return f"{SEAT_NAMES[player]} turned over {RANK_TEXT[rank]} and it went down"
            taker = "you" if player == ME else "they"
            return (
                f"{SEAT_NAMES[player]} turned over {RANK_TEXT[rank]}; it could not go "
                f"down, so {taker} picked the pile up"
            )
        case RecordCards(ranks=ranks):
            reason = before.pending[0].reason if before.pending else PendingReason.DRAW
            source = "drew" if reason is PendingReason.DRAW else "picked up"
            return f"You recorded what you {source}: {_rank_list(ranks)}"
        case CorrectState(note=note):
            suffix = f" -- {note}" if note else ""
            return f"Correction recorded{suffix}"
    raise ValueError(f"Unknown observation {event!r}")


def _rank_groups(ranks: tuple[Rank, ...]) -> list[dict[str, Any]]:
    """Group ranks into the quantity buttons the interface draws.

    Args:
        ranks: The ranks held in one zone.

    Returns:
        One entry per distinct rank, ascending, with its quantity.
    """
    tally = Counter(ranks)
    return [
        {"rank": RANK_TEXT[rank], "count": tally[rank], "label": RANK_TEXT[rank]}
        for rank in sorted(tally)
    ]


def _zone_code(zone: Zone | None) -> str | None:
    """Encode an active zone for the interface.

    Args:
        zone: The zone, or ``None`` for a seat that has finished.

    Returns:
        The wire code, or ``None``.
    """
    return None if zone is None else zone.value


def _seat_payload(state: ObservedState, player: PlayerId) -> dict[str, Any]:
    """Render one seat for the table-status panel.

    Args:
        state: The tracked position.
        player: The seat.

    Returns:
        Counts, the ranks actually observed, and which zone that seat plays from.
        ``hand_unknown`` is reported beside the known ranks rather than folded into
        them, so the screen shows uncertainty instead of hiding it.
    """
    seat = state.seat(player)
    try:
        zone = _zone_code(seat_active_zone(state, player))
    except ObservationError:
        zone = None
    return {
        "player": SEAT_CODE[player],
        "name": SEAT_NAMES[player],
        "hand_count": seat.hand_count,
        "hand_known": _rank_groups(seat.hand_known),
        "hand_unknown": seat.hand_unknown,
        "face_up": _rank_groups(seat.face_up),
        "face_up_count": len(seat.face_up),
        "face_down": seat.face_down,
        "remaining": seat.remaining_count,
        "active_zone": zone,
    }


def _options_payload(state: ObservedState, recommended: Move | None) -> list[dict[str, Any]]:
    """Render my own legal actions as tap targets.

    Args:
        state: The tracked position.
        recommended: The move the agent suggested, so one option can be marked.

    Returns:
        One entry per distinct action. Every legal reveal collapses into a single
        entry: face-down cards are indistinguishable, so offering one button per
        slot would invent a choice the table does not have.
    """
    if state.to_act != ME or state.pending:
        return []
    try:
        moves = observed_legal_moves(state, ME)
    except ObservationError:
        return []
    options: list[dict[str, Any]] = []
    seen_reveal = False
    for move in moves:
        if isinstance(move, Reveal):
            if seen_reveal:
                continue
            seen_reveal = True
        payload = encode_move(move)
        payload["label"] = describe_move(move)
        payload["recommended"] = recommended is not None and move == recommended
        options.append(payload)
    return options


def _recommendation_payload(recommendation: Recommendation) -> dict[str, Any]:
    """Render a recommendation for the panel that shows it.

    Args:
        recommendation: The suggestion.

    Returns:
        The move to submit back when the operator taps "I played this", plus the
        reasoning, the rule effect, and the standing caveat.
    """
    return {
        "move": encode_move(recommendation.move),
        "headline": recommendation.headline,
        "reasoning": recommendation.reasoning,
        "effect": recommendation.effect,
        "caveat": recommendation.caveat,
        "considered": recommendation.considered,
        "notes": list(recommendation.notes),
    }


def _blocker_payload(blockers: tuple[Blocker, ...]) -> list[dict[str, str]]:
    """Render the reasons advice is unavailable.

    Args:
        blockers: The blockers.

    Returns:
        One entry each, with the stable code the interface branches on.
    """
    return [{"code": blocker.code, "message": blocker.message} for blocker in blockers]


def _pending_payload(state: ObservedState) -> list[dict[str, Any]]:
    """Render the ranks still to be typed in.

    Args:
        state: The tracked position.

    Returns:
        One entry per outstanding group, oldest first, each with the prompt to show.
    """
    payload: list[dict[str, Any]] = []
    for entry in state.pending:
        source = "drew" if entry.reason is PendingReason.DRAW else "took from the pile"
        payload.append(
            {
                "count": entry.count,
                "reason": entry.reason.value,
                "prompt": (
                    f"Enter the {entry.count} card{'' if entry.count == 1 else 's'} you {source}"
                ),
            }
        )
    return payload


def render(session: Session) -> dict[str, Any]:
    """Render everything one screen needs from a session document.

    This is the server's whole job: a pure function from the document the browser
    holds to the payload the browser draws. Nothing is cached, nothing is stored,
    and calling it twice on the same document gives the same answer, agent
    tie-breaking included.

    Args:
        session: The session, already decoded.

    Returns:
        The payload: the revision to tag a reply with, the derived position, my
        options, the recommendation or the blockers standing in for it, and the
        action history.

    Raises:
        ObservationError: If the session's initial position could not describe a
            real table; see :func:`derive`.
    """
    derivation = derive(session)
    state = derivation.state
    blockers = advice_blockers(state)
    recommendation: Recommendation | None = None
    if not blockers:
        try:
            recommendation = recommend(state, seed=advice_seed(session))
        except ObservationError as error:  # Refuse to guess rather than show a move.
            blockers = (*blockers, Blocker("unavailable", str(error)))

    playable = tuple(
        rank for rank in sorted(RANK_BY_CODE.values()) if can_play_rank(rank, state.constraint)
    )
    opponent = state.seat(state.other(ME))
    try:
        opponent_zone = seat_active_zone(state, opponent.player)
    except ObservationError:  # Unresolvable without a deck count; bound by the hand.
        opponent_zone = Zone.HAND
    in_face_up = opponent_zone is Zone.FACE_UP
    opponent_most = len(opponent.face_up) if in_face_up else opponent.hand_count
    suggested = None if recommendation is None else recommendation.move
    return {
        "schema_version": session.schema_version,
        "revision": revision(session),
        "event_count": len(session.events),
        "applied": derivation.applied,
        "replay_error": derivation.error,
        "can_undo": bool(session.events),
        "state": {
            "phase": state.phase.value,
            "finished": state.is_finished,
            "winner": None if state.winner is None else SEAT_CODE[state.winner],
            "to_act": None if state.to_act is None else SEAT_CODE[state.to_act],
            "to_act_name": None if state.to_act is None else SEAT_NAMES[state.to_act],
            "my_turn": state.to_act == ME,
            "constraint": {
                **encode_constraint(state.constraint),
                "text": describe_constraint(state.constraint),
            },
            "playable_ranks": [RANK_TEXT[rank] for rank in playable],
            "pile": {
                "size": state.pile_size,
                "unknown": sum(1 for rank in state.pile if rank is None),
                "top": None if state.pile_top is None else RANK_TEXT[state.pile_top],
                "ranks": [None if rank is None else RANK_TEXT[rank] for rank in state.pile[-6:]],
            },
            "deck_count": state.deck_count,
            "burned_count": len(state.burned),
            "seats": [_seat_payload(state, seat.player) for seat in state.seats],
            "pending": _pending_payload(state),
            "opponent_max_count": opponent_most,
        },
        "options": _options_payload(state, suggested),
        "recommendation": (
            None if recommendation is None else _recommendation_payload(recommendation)
        ),
        "blockers": _blocker_payload(blockers),
        "history": [
            {"index": index + 1, "text": text} for index, text in enumerate(derivation.history)
        ],
    }
