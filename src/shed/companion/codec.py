"""The JSON vocabulary the phone and the server share, and its decoder.

Everything crossing the wire is decoded here, and decoding is the only place
untyped data is checked. The engine deliberately assumes its annotations and
validates domain invariants alone, so a browser sending ``"count": "two"`` or
``"rank": "Z"`` has to be refused before any engine or reducer type is
constructed. Every failure raises :class:`CompanionDataError` with a message
naming the field, because those messages are what the interface shows.

The wire format is a *document*, not a diff: a schema version, the versioned
initial state, and the ordered observation log. The browser owns that document and
stores it; the server holds no session state at all and simply folds the log it is
given. That is what makes a Python restart invisible to a game in progress, and it
is why the same decoder validates both a request body and a file the operator
imported from a backup.

Rank spellings are the ones :mod:`shed.companion.observed` uses on screen, so an
exported file reads the way the phone does.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from shed.agents import AGENT_KINDS, AgentSpec
from shed.companion.advice import DEFAULT_CHOICE, AgentChoice, profile_for
from shed.companion.observed import (
    ME,
    OPPONENT,
    RANK_TEXT,
    SEATS,
    CorrectState,
    ObservationEvent,
    ObservedState,
    PendingEntry,
    PendingReason,
    PickUpPile,
    PlayCards,
    RecordCards,
    RevealFaceDown,
    SeatObservation,
    StatePatch,
)
from shed.engine import (
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
    Unrestricted,
    Zone,
)

__all__ = [
    "CompanionDataError",
    "RANK_BY_CODE",
    "SEAT_BY_CODE",
    "decode_agent",
    "decode_constraint",
    "decode_event",
    "decode_rank",
    "decode_state",
    "encode_agent",
    "encode_constraint",
    "encode_event",
    "encode_move",
    "encode_state",
]

RANK_BY_CODE: Mapping[str, Rank] = {code: rank for rank, code in RANK_TEXT.items()}
"""Rank per wire code; the inverse of the interface's own rank spellings."""

SEAT_BY_CODE: Mapping[str, PlayerId] = {"me": ME, "opponent": OPPONENT}
"""Seat per wire code. Seats are named, not numbered, so a log reads plainly."""

SEAT_CODE: Mapping[PlayerId, str] = {player: code for code, player in SEAT_BY_CODE.items()}
"""Wire code per seat."""


class CompanionDataError(ValueError):
    """Raised when external data is not the shape the companion accepts.

    Decoding failures are told apart from
    :class:`~shed.companion.observed.ObservationError` on purpose: this one means
    the payload was malformed -- a bad import, an old schema, a client bug -- while
    that one means a well-formed observation contradicted the game.
    """


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    """Require a JSON object.

    Args:
        value: The decoded value.
        where: Field path, for the message.

    Returns:
        The value as a mapping.

    Raises:
        CompanionDataError: If it is not a JSON object.
    """
    if not isinstance(value, Mapping):
        raise CompanionDataError(f"{where} must be an object")
    return value


def _sequence(value: object, where: str) -> Sequence[Any]:
    """Require a JSON array.

    Args:
        value: The decoded value.
        where: Field path, for the message.

    Returns:
        The value as a sequence.

    Raises:
        CompanionDataError: If it is not a JSON array. A string is refused too,
            even though it is a sequence, because a string is never the array a
            caller meant.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise CompanionDataError(f"{where} must be an array")
    return value


def _integer(value: object, where: str, *, minimum: int | None = 0) -> int:
    """Require an integer, refusing the booleans and floats JSON allows.

    Args:
        value: The decoded value.
        where: Field path, for the message.
        minimum: Smallest acceptable value, or ``None`` for no bound.

    Returns:
        The integer.

    Raises:
        CompanionDataError: If it is not an integer or is below the bound. ``True``
            is an ``int`` in Python and ``3.0`` is not one, so both are named
            explicitly rather than coerced.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompanionDataError(f"{where} must be a whole number")
    if minimum is not None and value < minimum:
        raise CompanionDataError(f"{where} must be at least {minimum}, got {value}")
    return value


def _text(value: object, where: str) -> str:
    """Require a string.

    Args:
        value: The decoded value.
        where: Field path, for the message.

    Returns:
        The string.

    Raises:
        CompanionDataError: If it is not a string.
    """
    if not isinstance(value, str):
        raise CompanionDataError(f"{where} must be a string")
    return value


def decode_rank(value: object, where: str) -> Rank:
    """Decode one rank code.

    Args:
        value: The code, such as ``"10"`` or ``"JK"``.
        where: Field path, for the message.

    Returns:
        The rank.

    Raises:
        CompanionDataError: If the code names no rank. The message lists the codes,
            because a typed rank is the most common thing to get wrong.
    """
    code = _text(value, where).strip().upper()
    if code not in RANK_BY_CODE:
        allowed = " ".join(RANK_TEXT[rank] for rank in sorted(RANK_BY_CODE.values()))
        raise CompanionDataError(f"{where}: {code!r} is not a rank; use one of {allowed}")
    return RANK_BY_CODE[code]


def decode_ranks(value: object, where: str) -> tuple[Rank, ...]:
    """Decode an array of rank codes.

    Args:
        value: The array.
        where: Field path, for the message.

    Returns:
        The ranks in the order given.

    Raises:
        CompanionDataError: If the value is not an array of rank codes.
    """
    return tuple(
        decode_rank(item, f"{where}[{index}]") for index, item in enumerate(_sequence(value, where))
    )


def decode_optional_ranks(value: object, where: str) -> tuple[Rank | None, ...]:
    """Decode a pile-style array, where ``null`` marks an unobserved card.

    Args:
        value: The array; ``null`` entries stay unknown.
        where: Field path, for the message.

    Returns:
        The entries in order, ``None`` where the rank was never seen.

    Raises:
        CompanionDataError: If an entry is neither ``null`` nor a rank code.
    """
    return tuple(
        None if item is None else decode_rank(item, f"{where}[{index}]")
        for index, item in enumerate(_sequence(value, where))
    )


def decode_seat(value: object, where: str) -> PlayerId:
    """Decode a seat code.

    Args:
        value: ``"me"`` or ``"opponent"``.
        where: Field path, for the message.

    Returns:
        The seat.

    Raises:
        CompanionDataError: If the code names no seat.
    """
    code = _text(value, where)
    if code not in SEAT_BY_CODE:
        raise CompanionDataError(f"{where}: {code!r} is not a seat; use 'me' or 'opponent'")
    return SEAT_BY_CODE[code]


def encode_constraint(constraint: PlayConstraint) -> dict[str, Any]:
    """Encode the restriction in force.

    Args:
        constraint: The restriction.

    Returns:
        A tagged object: ``{"kind": "unrestricted"}``, or a kind with a ``rank``.

    Raises:
        ValueError: If the constraint is not a known constraint type.
    """
    match constraint:
        case Unrestricted():
            return {"kind": "unrestricted"}
        case AtLeast(rank=rank):
            return {"kind": "at_least", "rank": RANK_TEXT[rank]}
        case AtMost(rank=rank):
            return {"kind": "at_most", "rank": RANK_TEXT[rank]}
    raise ValueError(f"Unknown play constraint {constraint!r}")


def decode_constraint(value: object, where: str) -> PlayConstraint:
    """Decode the restriction in force.

    Args:
        value: The tagged object.
        where: Field path, for the message.

    Returns:
        The constraint. Its own constructor rejects a joker as a bound, so an
        impossible restriction cannot be built from a payload.

    Raises:
        CompanionDataError: If the kind is unknown, a bound is missing, or the
            bound is one the profile forbids.
    """
    payload = _mapping(value, where)
    kind = _text(payload.get("kind"), f"{where}.kind")
    if kind == "unrestricted":
        return Unrestricted()
    if kind not in ("at_least", "at_most"):
        raise CompanionDataError(
            f"{where}.kind: {kind!r} is not a restriction; use 'unrestricted', "
            "'at_least', or 'at_most'"
        )
    rank = decode_rank(payload.get("rank"), f"{where}.rank")
    try:
        return AtLeast(rank) if kind == "at_least" else AtMost(rank)
    except ValueError as error:
        raise CompanionDataError(f"{where}: {error}") from error


def encode_state(state: ObservedState) -> dict[str, Any]:
    """Encode a tracked position.

    Unobserved cards encode as ``null`` in the pile and burned arrays and as
    ``hand_unknown`` counts, so a decoded document is exactly as ignorant as the
    one that was saved: a round trip can never sharpen an observation.

    Args:
        state: The position.

    Returns:
        A JSON-ready object.
    """
    return {
        "rules": state.rules.id,
        "seats": [
            {
                "player": SEAT_CODE[seat.player],
                "hand_known": [RANK_TEXT[rank] for rank in seat.hand_known],
                "hand_unknown": seat.hand_unknown,
                "face_up": [RANK_TEXT[rank] for rank in seat.face_up],
                "face_down": seat.face_down,
            }
            for seat in state.seats
        ],
        "deck_count": state.deck_count,
        "pile": [None if rank is None else RANK_TEXT[rank] for rank in state.pile],
        "burned": [None if rank is None else RANK_TEXT[rank] for rank in state.burned],
        "constraint": encode_constraint(state.constraint),
        "to_act": None if state.to_act is None else SEAT_CODE[state.to_act],
        "phase": state.phase.value,
        "winner": None if state.winner is None else SEAT_CODE[state.winner],
        "pending": [
            {
                "player": SEAT_CODE[entry.player],
                "count": entry.count,
                "reason": entry.reason.value,
            }
            for entry in state.pending
        ],
    }


def _decode_seat_observation(value: object, where: str) -> SeatObservation:
    """Decode one seat's observation.

    Args:
        value: The seat object.
        where: Field path, for the message.

    Returns:
        The observation.

    Raises:
        CompanionDataError: If a field is missing or the wrong shape.
    """
    payload = _mapping(value, where)
    try:
        return SeatObservation(
            player=decode_seat(payload.get("player"), f"{where}.player"),
            hand_known=decode_ranks(payload.get("hand_known", []), f"{where}.hand_known"),
            hand_unknown=_integer(payload.get("hand_unknown", 0), f"{where}.hand_unknown"),
            face_up=decode_ranks(payload.get("face_up", []), f"{where}.face_up"),
            face_down=_integer(payload.get("face_down", 0), f"{where}.face_down"),
        )
    except ValueError as error:
        if isinstance(error, CompanionDataError):
            raise
        raise CompanionDataError(f"{where}: {error}") from error


def _decode_pending(value: object, where: str) -> tuple[PendingEntry, ...]:
    """Decode the queue of ranks still to be typed in.

    Args:
        value: The array of entries.
        where: Field path, for the message.

    Returns:
        The queue, oldest first.

    Raises:
        CompanionDataError: If an entry is malformed or names an unknown reason.
    """
    entries: list[PendingEntry] = []
    for index, item in enumerate(_sequence(value, where)):
        path = f"{where}[{index}]"
        payload = _mapping(item, path)
        reason = _text(payload.get("reason"), f"{path}.reason")
        if reason not in {member.value for member in PendingReason}:
            raise CompanionDataError(f"{path}.reason: {reason!r} is not a known reason")
        try:
            entries.append(
                PendingEntry(
                    player=decode_seat(payload.get("player"), f"{path}.player"),
                    count=_integer(payload.get("count"), f"{path}.count", minimum=1),
                    reason=PendingReason(reason),
                )
            )
        except ValueError as error:
            if isinstance(error, CompanionDataError):
                raise
            raise CompanionDataError(f"{path}: {error}") from error
    return tuple(entries)


def decode_state(value: object, where: str = "state") -> ObservedState:
    """Decode a tracked position.

    The result is *not* validated here. Decoding answers "is this the shape of a
    position"; whether the position could describe a real table is
    :func:`~shed.companion.observed.validate_observed`, which the session applies
    when it starts folding the log, so one error path reports both.

    Args:
        value: The state object.
        where: Field path, for the message.

    Returns:
        The position.

    Raises:
        CompanionDataError: If a field is missing or the wrong shape, or the rules
            profile is not one this release implements.
    """
    payload = _mapping(value, where)
    profile = _text(payload.get("rules", RulesConfig().id), f"{where}.rules")
    rules = RulesConfig()
    if profile != rules.id:
        raise CompanionDataError(f"{where}.rules: this release plays {rules.id!r}, not {profile!r}")
    seats = _sequence(payload.get("seats"), f"{where}.seats")
    if len(seats) != len(SEATS):
        raise CompanionDataError(f"{where}.seats must hold exactly {len(SEATS)} seats")
    observations = tuple(
        _decode_seat_observation(item, f"{where}.seats[{index}]")
        for index, item in enumerate(seats)
    )
    deck = payload.get("deck_count")
    phase_code = _text(payload.get("phase", Phase.PLAY.value), f"{where}.phase")
    if phase_code not in {member.value for member in Phase}:
        raise CompanionDataError(f"{where}.phase: {phase_code!r} is not a phase")
    to_act = payload.get("to_act")
    winner = payload.get("winner")
    return ObservedState(
        rules=rules,
        seats=(observations[0], observations[1]),
        deck_count=None if deck is None else _integer(deck, f"{where}.deck_count"),
        pile=decode_optional_ranks(payload.get("pile", []), f"{where}.pile"),
        burned=decode_optional_ranks(payload.get("burned", []), f"{where}.burned"),
        constraint=decode_constraint(
            payload.get("constraint", {"kind": "unrestricted"}), f"{where}.constraint"
        ),
        to_act=None if to_act is None else decode_seat(to_act, f"{where}.to_act"),
        phase=Phase(phase_code),
        winner=None if winner is None else decode_seat(winner, f"{where}.winner"),
        pending=_decode_pending(payload.get("pending", []), f"{where}.pending"),
    )


def encode_move(move: Move) -> dict[str, Any]:
    """Encode one engine move for the interface to show or to submit back.

    Args:
        move: The move.

    Returns:
        A tagged object. A reveal carries no slot: face-down cards are
        indistinguishable, so naming one would suggest a choice that does not
        exist.

    Raises:
        ValueError: If the move is not one the companion can express.
    """
    match move:
        case Play(source=source, rank=rank, count=count):
            return {
                "kind": "play",
                "zone": "hand" if source is Zone.HAND else "face_up",
                "rank": RANK_TEXT[rank],
                "count": count,
            }
        case Reveal():
            return {"kind": "reveal"}
        case PickUp():
            return {"kind": "pickup"}
    raise ValueError(f"The companion cannot encode {move!r}")


def _decode_patch(value: object, where: str) -> StatePatch:
    """Decode a correction's field set.

    Args:
        value: The patch object; absent fields are left alone.
        where: Field path, for the message.

    Returns:
        The patch.

    Raises:
        CompanionDataError: If a present field is the wrong shape.
    """
    payload = _mapping(value, where)
    unknown = sorted(set(payload) - _PATCH_FIELDS)
    if unknown:
        raise CompanionDataError(f"{where}: unknown correction field(s) {', '.join(unknown)}")

    def ranks(name: str) -> tuple[Rank, ...] | None:
        """Decode one optional rank array from the patch.

        Args:
            name: Field name.

        Returns:
            The ranks, or ``None`` when the field is absent.
        """
        return None if name not in payload else decode_ranks(payload[name], f"{where}.{name}")

    def count(name: str) -> int | None:
        """Decode one optional count from the patch.

        Args:
            name: Field name.

        Returns:
            The count, or ``None`` when the field is absent.
        """
        return None if name not in payload else _integer(payload[name], f"{where}.{name}")

    constraint = payload.get("constraint")
    to_act = payload.get("to_act")
    return StatePatch(
        my_hand=ranks("my_hand"),
        my_face_up=ranks("my_face_up"),
        my_face_down=count("my_face_down"),
        opponent_hand_known=ranks("opponent_hand_known"),
        opponent_hand_unknown=count("opponent_hand_unknown"),
        opponent_face_up=ranks("opponent_face_up"),
        opponent_face_down=count("opponent_face_down"),
        deck_count=count("deck_count"),
        deck_unknown=bool(payload.get("deck_unknown", False)),
        pile=(
            None
            if "pile" not in payload
            else decode_optional_ranks(payload["pile"], f"{where}.pile")
        ),
        burned_count=count("burned_count"),
        constraint=(
            None if constraint is None else decode_constraint(constraint, f"{where}.constraint")
        ),
        to_act=None if to_act is None else decode_seat(to_act, f"{where}.to_act"),
    )


_PATCH_FIELDS: frozenset[str] = frozenset(
    {
        "my_hand",
        "my_face_up",
        "my_face_down",
        "opponent_hand_known",
        "opponent_hand_unknown",
        "opponent_face_up",
        "opponent_face_down",
        "deck_count",
        "deck_unknown",
        "pile",
        "burned_count",
        "constraint",
        "to_act",
    }
)
"""Correction fields the decoder accepts.

An unknown field is an error rather than a silent no-op: a typo in a correction
would otherwise look as though it had been applied.
"""


def encode_event(event: ObservationEvent) -> dict[str, Any]:
    """Encode one observation for the log.

    Args:
        event: The observation.

    Returns:
        A tagged object.

    Raises:
        ValueError: If the event is not a known observation type.
    """
    match event:
        case PlayCards(player=player, rank=rank, count=count):
            return {
                "kind": "play",
                "player": SEAT_CODE[player],
                "rank": RANK_TEXT[rank],
                "count": count,
            }
        case PickUpPile(player=player):
            return {"kind": "pickup", "player": SEAT_CODE[player]}
        case RevealFaceDown(player=player, rank=rank):
            return {"kind": "reveal", "player": SEAT_CODE[player], "rank": RANK_TEXT[rank]}
        case RecordCards(ranks=ranks):
            return {"kind": "record", "ranks": [RANK_TEXT[rank] for rank in ranks]}
        case CorrectState(patch=patch, note=note):
            return {"kind": "correct", "patch": _encode_patch(patch), "note": note}
    raise ValueError(f"Unknown observation {event!r}")


def _encode_patch(patch: StatePatch) -> dict[str, Any]:
    """Encode only the fields a correction actually set.

    Args:
        patch: The patch.

    Returns:
        A JSON-ready object holding just the changed fields, so a log line shows
        what the correction touched and nothing else.
    """
    encoded: dict[str, Any] = {}
    for name in ("my_hand", "my_face_up", "opponent_hand_known", "opponent_face_up"):
        value = getattr(patch, name)
        if value is not None:
            encoded[name] = [RANK_TEXT[rank] for rank in value]
    for name in (
        "my_face_down",
        "opponent_hand_unknown",
        "opponent_face_down",
        "deck_count",
        "burned_count",
    ):
        value = getattr(patch, name)
        if value is not None:
            encoded[name] = value
    if patch.deck_unknown:
        encoded["deck_unknown"] = True
    if patch.pile is not None:
        encoded["pile"] = [None if rank is None else RANK_TEXT[rank] for rank in patch.pile]
    if patch.constraint is not None:
        encoded["constraint"] = encode_constraint(patch.constraint)
    if patch.to_act is not None:
        encoded["to_act"] = SEAT_CODE[patch.to_act]
    return encoded


def decode_event(value: object, where: str = "event") -> ObservationEvent:
    """Decode one observation.

    Args:
        value: The tagged object.
        where: Field path, for the message.

    Returns:
        The observation.

    Raises:
        CompanionDataError: If the kind is unknown or a field is the wrong shape.
    """
    payload = _mapping(value, where)
    kind = _text(payload.get("kind"), f"{where}.kind")
    match kind:
        case "play":
            return PlayCards(
                player=decode_seat(payload.get("player"), f"{where}.player"),
                rank=decode_rank(payload.get("rank"), f"{where}.rank"),
                count=_integer(payload.get("count"), f"{where}.count", minimum=1),
            )
        case "pickup":
            return PickUpPile(player=decode_seat(payload.get("player"), f"{where}.player"))
        case "reveal":
            return RevealFaceDown(
                player=decode_seat(payload.get("player"), f"{where}.player"),
                rank=decode_rank(payload.get("rank"), f"{where}.rank"),
            )
        case "record":
            return RecordCards(ranks=decode_ranks(payload.get("ranks"), f"{where}.ranks"))
        case "correct":
            return CorrectState(
                patch=_decode_patch(payload.get("patch", {}), f"{where}.patch"),
                note=_text(payload.get("note", ""), f"{where}.note"),
            )
    raise CompanionDataError(
        f"{where}.kind: {kind!r} is not an observation; use play, pickup, reveal, "
        "record, or correct"
    )


def encode_agent(choice: AgentChoice) -> dict[str, Any]:
    """Encode which agent advises a game.

    The label and the summary are encoded alongside the choice so a saved document
    still says, in words, which strategy it was played with -- useful when it is
    read back by a release whose agent list has moved on.

    Args:
        choice: The chosen strategy and its tie-break salt.

    Returns:
        A JSON-ready object.
    """
    profile = choice.profile
    return {
        "kind": choice.spec.kind,
        "name": choice.spec.name,
        "seed": choice.seed,
        "label": profile.label,
        "summary": profile.summary,
        "caveat": profile.caveat,
    }


def decode_agent(value: object, where: str = "agent") -> AgentChoice:
    """Decode which agent advises a game.

    The descriptive fields :func:`encode_agent` writes are deliberately ignored on
    the way back: they are a record of what an earlier release said, and this one
    re-derives them from the package so a renamed or re-described strategy is not
    frozen into old documents.

    Args:
        value: The agent object, or ``None`` for the default choice.
        where: Field path, for the message.

    Returns:
        The choice.

    Raises:
        CompanionDataError: If a field is the wrong shape, or the kind is not one
            this release ships. The message lists the kinds that exist, because an
            imported document is the likeliest source of one that does not.
    """
    if value is None:
        return DEFAULT_CHOICE
    payload = _mapping(value, where)
    kind = _text(payload.get("kind"), f"{where}.kind")
    if kind not in AGENT_KINDS:
        raise CompanionDataError(
            f"{where}.kind: {kind!r} is not an agent this release ships; "
            f"use one of {', '.join(AGENT_KINDS)}"
        )
    seed = payload.get("seed")
    if seed is not None:
        seed = _integer(seed, f"{where}.seed", minimum=0)
    name = payload.get("name", kind)
    try:
        spec = AgentSpec(kind=kind, name=_text(name, f"{where}.name"))
    except ValueError as error:
        raise CompanionDataError(f"{where}: {error}") from error
    profile_for(kind)  # Refuses a kind the companion cannot even name.
    return AgentChoice(spec=spec, seed=seed)
