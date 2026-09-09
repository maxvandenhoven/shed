"""Versioned JSON replay: writing a finished match down, and playing it back.

This module is the project's whole serialization boundary. Every JSON encoder
and decoder lives here, so the engine, the agents, and the match runner keep
working with typed domain objects and never learn that a file format exists.
Nothing imports this module from inside those layers; it imports them.

Two directions, deliberately asymmetric:

* **Encoding** takes trusted typed records -- a :class:`~shed.match.MatchResult`
  the runner just produced -- and writes explicitly tagged JSON. Every union is
  tagged with a ``"type"`` field, every enum is stored by value, and no Python
  pickle is involved: a replay is a readable artifact, not a serialized object
  graph.
* **Decoding** takes an untrusted document that merely claims to be a replay. It
  validates shapes, tags, primitive types, required fields, and the supported
  schema and profile *here*, then constructs the domain objects explicitly --
  :class:`~shed.engine.Rank`, :class:`~shed.engine.Suit`, the identifier types,
  :class:`~shed.engine.RulesConfig`, the moves. Those constructors check domain
  invariants and never coerce, which is exactly why the coercion question has to
  be settled before they are called. In particular ``True`` is not an integer
  here, even though Python says it is: a JSON boolean in a count, a rank, or a
  seat is rejected rather than silently played as a one.

Legality is not this module's business. A decoded move is only well-formed; the
engine decides whether it was playable, because :func:`verify_replay` replays it
through :meth:`~shed.engine.GameState.apply_move` exactly as the runner did.

Replay reconstructs the opening position from the recorded deck order with
:func:`~shed.engine.deal_initial_state`, the same pure helper the runner deals
with. Recording the order alongside the seed is what makes an old replay
independent of any future change to the shuffle. Agents are never built: a
replay reproduces the *recorded decisions*, not the search that chose them,
which is why it is deterministic even though the timed match that produced it
was not.

A complete replay contains hidden information -- face-down identities, the deck
order, and every private draw. It is a trusted post-match artifact. Never hand
one to an agent, and never confuse it with the filtered history an observation
carries.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import shed
from shed.agents import AgentSpec
from shed.engine import (
    Arrange,
    ArrangementCommitted,
    AtLeast,
    AtMost,
    BurnReason,
    Card,
    CardId,
    CardRevealed,
    CardsDrawn,
    CardsPlayed,
    GameEnded,
    GameStarted,
    GameState,
    HandDealt,
    IllegalMoveError,
    Move,
    ObservedEvent,
    Outcome,
    Phase,
    PickUp,
    PileBurned,
    PilePickedUp,
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
    deal_initial_state,
    shuffled_deck,
)
from shed.match import (
    AppliedDecision,
    CloseReason,
    FinalPosition,
    MatchConfig,
    MatchMetadata,
    MatchResult,
    MatchStatus,
    PlayerPosition,
    TurnRecord,
    WorkerFailed,
)

__all__ = [
    "SUPPORTED_PROFILES",
    "SUPPORTED_SCHEMAS",
    "Replay",
    "ReplayCheck",
    "ReplayFormatError",
    "ReplayHeader",
    "decode_card",
    "decode_constraint",
    "decode_event",
    "decode_move",
    "decode_replay",
    "detect_source_revision",
    "encode_card",
    "encode_constraint",
    "encode_event",
    "encode_move",
    "match_document",
    "read_replay",
    "verify_replay",
    "write_match",
]

SUPPORTED_SCHEMAS: tuple[int, ...] = (shed.REPLAY_SCHEMA_VERSION,)
"""Replay schema versions this release can read. It writes the last one."""

SUPPORTED_PROFILES: tuple[str, ...] = (shed.RULES_PROFILE_ID,)
"""Rules profiles this release can replay; a replay of any other is refused."""

type JsonObject = dict[str, object]
"""A decoded JSON object, before anything about its contents is known."""


class ReplayFormatError(ValueError):
    """Raised when a document is not a replay this release can read.

    It covers every rejection at this boundary: a malformed shape, a missing or
    wrongly typed field, an unknown tag, a value a domain constructor refuses,
    and an unsupported schema or rules profile. Illegal *play* is not a format
    error -- :func:`verify_replay` reports that as a failed check instead.
    """


# --------------------------------------------------------------------------- #
# Encoding: trusted typed records to tagged JSON.
# --------------------------------------------------------------------------- #


def encode_card(card: Card) -> JsonObject:
    """Encode one physical card.

    Args:
        card: The card to encode.

    Returns:
        Its identifier, its rank as the enum's integer value, and its suit as
        the enum's string value, or ``None`` for a joker.
    """
    return {
        "id": int(card.id),
        "rank": int(card.rank),
        "suit": None if card.suit is None else card.suit.value,
    }


def _encode_cards(cards: Sequence[Card]) -> list[object]:
    """Encode a run of cards, preserving its order.

    Args:
        cards: Cards in an order that matters -- pile order, transfer order, or
            the canonical deck order.

    Returns:
        The encoded cards in the same order.
    """
    return [encode_card(card) for card in cards]


def encode_constraint(constraint: PlayConstraint) -> JsonObject:
    """Encode a play constraint with an explicit tag.

    Args:
        constraint: The restriction standing on the pile.

    Returns:
        A tagged object: ``unrestricted``, or ``at_least``/``at_most`` with the
        bounding rank.

    Raises:
        ReplayFormatError: If the union grew a member this encoder does not
            know, which would silently drop a rule from the artifact.
    """
    match constraint:
        case Unrestricted():
            return {"type": "unrestricted"}
        case AtLeast(rank=rank):
            return {"type": "at_least", "rank": int(rank)}
        case AtMost(rank=rank):
            return {"type": "at_most", "rank": int(rank)}
        case _:  # pragma: no cover - unreachable while the union is closed.
            raise ReplayFormatError(f"Cannot encode play constraint {constraint!r}")


def encode_move(move: Move) -> JsonObject:
    """Encode a move with an explicit tag.

    Args:
        move: The decision to encode.

    Returns:
        A tagged object, such as ``{"type": "play", "source": "hand",
        "rank": 7, "count": 2}``.

    Raises:
        ReplayFormatError: If the move union grew a member this encoder does not
            know.
    """
    match move:
        case Arrange(face_up_cards=ids):
            return {"type": "arrange", "face_up_cards": [int(card_id) for card_id in ids]}
        case Play(source=source, rank=rank, count=count):
            return {"type": "play", "source": source.value, "rank": int(rank), "count": count}
        case Reveal(slot=slot):
            return {"type": "reveal", "slot": int(slot)}
        case PickUp():
            return {"type": "pick_up"}
        case _:  # pragma: no cover - unreachable while the union is closed.
            raise ReplayFormatError(f"Cannot encode move {move!r}")


def _encode_public(public: PublicPlayerState) -> JsonObject:
    """Encode one player's public state.

    Args:
        public: The public entry to encode.

    Returns:
        The seat, its public hand size, its face-up cards, and its remaining
        face-down slots. Hidden identities are not part of this shape at all.
    """
    return {
        "player": int(public.player),
        "hand_count": public.hand_count,
        "face_up": _encode_cards(public.face_up),
        "face_down_slots": [int(slot) for slot in public.face_down_slots],
    }


def encode_event(event: ObservedEvent) -> JsonObject:
    """Encode one observed event with an explicit tag.

    Full internal events are encoded, identities intact: a replay is a trusted
    artifact, and the recorded stream is what verification compares against. The
    private events keep their ``cards`` field nullable, so a filtered copy --
    the shape an agent would have seen -- encodes just as faithfully.

    Args:
        event: The event to encode.

    Returns:
        A tagged object naming the event type and its fields.

    Raises:
        ReplayFormatError: If the event union grew a member this encoder does
            not know.
    """
    match event:
        case GameStarted(dealer=dealer, seat_order=seats, players=players):
            return {
                "type": "game_started",
                "dealer": int(dealer),
                "seat_order": [int(seat) for seat in seats],
                "players": [_encode_public(public) for public in players],
            }
        case HandDealt(player=player, count=count, cards=cards):
            return {
                "type": "hand_dealt",
                "player": int(player),
                "count": count,
                "cards": None if cards is None else _encode_cards(cards),
            }
        case ArrangementCommitted(player=player, face_up=face_up):
            return {
                "type": "arrangement_committed",
                "player": int(player),
                "face_up": _encode_cards(face_up),
            }
        case CardsPlayed(player=player, source=source, cards=cards):
            return {
                "type": "cards_played",
                "player": int(player),
                "source": source.value,
                "cards": _encode_cards(cards),
            }
        case CardRevealed(player=player, slot=slot, card=card, playable=playable):
            return {
                "type": "card_revealed",
                "player": int(player),
                "slot": int(slot),
                "card": encode_card(card),
                "playable": playable,
            }
        case CardsDrawn(player=player, count=count, cards=cards):
            return {
                "type": "cards_drawn",
                "player": int(player),
                "count": count,
                "cards": None if cards is None else _encode_cards(cards),
            }
        case PilePickedUp(player=player, cards=cards):
            return {"type": "pile_picked_up", "player": int(player), "cards": _encode_cards(cards)}
        case PileBurned(player=player, cards=cards, reason=reason):
            return {
                "type": "pile_burned",
                "player": int(player),
                "cards": _encode_cards(cards),
                "reason": reason.value,
            }
        case GameEnded(outcome=outcome):
            return {"type": "game_ended", "outcome": _encode_outcome(outcome)}
        case _:  # pragma: no cover - unreachable while the union is closed.
            raise ReplayFormatError(f"Cannot encode event {event!r}")


def _encode_outcome(outcome: Outcome) -> JsonObject:
    """Encode a game outcome.

    Args:
        outcome: The result to encode.

    Returns:
        The winning seat.
    """
    return {"winner": int(outcome.winner)}


def _encode_failure(failure: WorkerFailed) -> JsonObject:
    """Encode a worker failure record.

    Args:
        failure: What the worker reported, or what the runner recorded for it.

    Returns:
        The exception's type name and its truncated message.
    """
    return {"exception": failure.exception, "message": failure.message}


def _encode_turn(turn: TurnRecord) -> JsonObject:
    """Encode one decision's selection diagnostics.

    Timing measurements are recorded because they describe the match that
    happened; verification never requires them to reproduce.

    Args:
        turn: The record to encode.

    Returns:
        The decision's identity, its selected move, why it closed, its counters,
        its failure if there was one, its budget and measured times, and the
        agent seed it was decided with.
    """
    return {
        "decision_id": turn.decision_id,
        "player": int(turn.player),
        "phase": turn.phase.value,
        "move": encode_move(turn.move),
        "reason": turn.reason.value,
        "used_fallback": turn.used_fallback,
        "accepted": turn.accepted,
        "rejected": turn.rejected,
        "failure": None if turn.failure is None else _encode_failure(turn.failure),
        "budget_seconds": turn.budget_seconds,
        "selection_seconds": turn.selection_seconds,
        "cleanup_seconds": turn.cleanup_seconds,
        "agent_seed": turn.agent_seed,
    }


def _encode_decision(decision: AppliedDecision) -> JsonObject:
    """Encode one applied decision and everything it resolved into.

    Args:
        decision: An applied decision from the match result.

    Returns:
        Its turn record and its full events in resolution order.
    """
    return {
        "turn": _encode_turn(decision.turn),
        "events": [encode_event(event) for event in decision.events],
    }


def _encode_position(position: FinalPosition) -> JsonObject:
    """Encode the digest of the position a match stopped in.

    Cards appear as identifiers here rather than as objects: the document
    already carries the whole deck, and what a position comparison asks is where
    each physical card ended up.

    Args:
        position: The digest to encode.

    Returns:
        The phase, actor, ply counter, constraint, the three shared piles, and
        one entry per seat.
    """
    return {
        "phase": position.phase.value,
        "current_player": None if position.current_player is None else int(position.current_player),
        "current_ply": position.current_ply,
        "constraint": encode_constraint(position.constraint),
        "draw_pile": [int(card_id) for card_id in position.draw_pile],
        "discard_pile": [int(card_id) for card_id in position.discard_pile],
        "burned_cards": [int(card_id) for card_id in position.burned_cards],
        "players": [
            {
                "player": int(player.player),
                "hand": [int(card_id) for card_id in player.hand],
                "face_up": [int(card_id) for card_id in player.face_up],
                "face_down": [[int(slot), int(card_id)] for slot, card_id in player.face_down],
            }
            for player in position.players
        ],
    }


def _encode_rules(rules: RulesConfig) -> JsonObject:
    """Encode the rules profile.

    The whole profile is written, not just its identifier, so a reader can see
    what the recorded game claimed to be playing rather than trusting a label.

    Args:
        rules: The profile the match ran under.

    Returns:
        Every field of the profile.
    """
    return {
        "id": rules.id,
        "min_players": rules.min_players,
        "max_players": rules.max_players,
        "joker_count": rules.joker_count,
        "initial_hand_size": rules.initial_hand_size,
        "initial_face_up_count": rules.initial_face_up_count,
        "initial_face_down_count": rules.initial_face_down_count,
        "refill_target": rules.refill_target,
    }


def _encode_config(config: MatchConfig) -> JsonObject:
    """Encode the timing, limit, and failure policy the match ran under.

    Args:
        config: The match configuration.

    Returns:
        Every field of the configuration.
    """
    return {
        "seconds_per_turn": config.seconds_per_turn,
        "max_play_decisions": config.max_play_decisions,
        "fallback_seed": config.fallback_seed,
        "agent_seed": config.agent_seed,
        "strict_failures": config.strict_failures,
    }


def _encode_spec(spec: AgentSpec) -> JsonObject:
    """Encode one participant specification.

    Args:
        spec: The participant. Live agents are never serialized; this is the
            whole description a decision was built from.

    Returns:
        The agent kind and its evaluation label.
    """
    return {"kind": spec.kind, "name": spec.name}


def detect_source_revision(start: Path | None = None) -> str | None:
    """Read the git commit the working tree is on, if that is knowable.

    Recorded as provenance only. This reads ``.git`` directly rather than
    running a subprocess, which keeps process management out of the library. It
    deliberately handles only the ordinary cases -- a ``.git`` directory with a
    detached ``HEAD`` or a loose branch ref -- and reports nothing for a packed
    ref, a worktree's ``.git`` file, or a tree that is not a repository at all.

    Args:
        start: Directory to search upwards from. Defaults to the process's
            current working directory.

    Returns:
        The 40-character commit hash, or ``None`` when it cannot be read
        cheaply.
    """
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        head = directory / ".git" / "HEAD"
        if not head.is_file():
            continue
        pointer = head.read_text(encoding="utf-8").strip()
        if not pointer.startswith("ref:"):
            return pointer or None
        reference = directory / ".git" / pointer.removeprefix("ref:").strip()
        if reference.is_file():
            return reference.read_text(encoding="utf-8").strip() or None
        return None
    return None


def match_document(result: MatchResult, *, source_revision: str | None = None) -> JsonObject:
    """Build the complete replay document for one finished match.

    The deck order is recomputed here from the recorded deal seed with the same
    shuffle the runner dealt with, and stored explicitly. That is what lets a
    future release change its shuffle without invalidating old replays: replay
    reads the order, never the seed.

    Args:
        result: What the runner produced. It is trusted, typed, and complete.
        source_revision: Commit the match was played at, when it is known. Pass
            :func:`detect_source_revision` for the ordinary case.

    Returns:
        A JSON-ready document: the header, how the match was set up, and what
        happened, including the applied-decision stream and any selection that
        was chosen but never applied.
    """
    metadata = result.metadata
    deck = shuffled_deck(seed=metadata.deal_seed, config=metadata.rules)
    return {
        "schema": shed.REPLAY_SCHEMA_VERSION,
        "package_version": shed.__version__,
        "python_version": platform.python_version(),
        "source_revision": source_revision,
        "rules": _encode_rules(metadata.rules),
        "setup": {
            "player_count": metadata.player_count,
            "dealer": int(metadata.dealer),
            "deal_seed": metadata.deal_seed,
            "deck": _encode_cards(deck),
            "agents": [_encode_spec(spec) for spec in metadata.agents],
            "config": _encode_config(metadata.config),
        },
        "result": {
            "status": result.status.value,
            "outcome": None if result.outcome is None else _encode_outcome(result.outcome),
            "failure": result.failure,
            "play_decisions": result.play_decisions,
            "initial_events": [encode_event(event) for event in result.initial_events],
            "decisions": [_encode_decision(decision) for decision in result.decisions],
            "unapplied_turns": [_encode_turn(turn) for turn in result.unapplied_turns],
            "final_position": _encode_position(result.final_position),
        },
    }


def write_match(
    result: MatchResult,
    path: Path,
    *,
    source_revision: str | None = None,
) -> Path:
    """Write one match's replay to a UTF-8 JSON file.

    Missing parent directories are created, so a caller can name
    ``results/match.json`` in a fresh checkout.

    Args:
        result: The match to record.
        path: Destination file, overwritten if it exists.
        source_revision: Commit the match was played at, when it is known.

    Returns:
        The path written, for the caller to report.

    Raises:
        ValueError: If a measurement is not finite. JSON has no ``NaN``, and
            writing one would produce a file no strict reader accepts.
    """
    document = match_document(result, source_revision=source_revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(f"{text}\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Decoding: untrusted JSON to typed domain objects.
# --------------------------------------------------------------------------- #


def _at(path: str, key: str | int) -> str:
    """Extend a field path, for error messages that say where the fault is.

    Args:
        path: Path of the containing value, such as ``result.decisions[2]``.
        key: A member name or a list index.

    Returns:
        The child's path.
    """
    return f"{path}[{key}]" if isinstance(key, int) else f"{path}.{key}"


def _mapping(value: object, path: str) -> JsonObject:
    """Require a JSON object.

    Args:
        value: The raw decoded value.
        path: Where it came from.

    Returns:
        The value as a mapping of string keys.

    Raises:
        ReplayFormatError: If it is not a JSON object.
    """
    if not isinstance(value, dict):
        raise ReplayFormatError(f"{path} must be an object, got {type(value).__name__}")
    return value


def _sequence(value: object, path: str) -> list[object]:
    """Require a JSON array.

    Args:
        value: The raw decoded value.
        path: Where it came from.

    Returns:
        The value as a list.

    Raises:
        ReplayFormatError: If it is not a JSON array.
    """
    if not isinstance(value, list):
        raise ReplayFormatError(f"{path} must be an array, got {type(value).__name__}")
    return value


def _text(value: object, path: str) -> str:
    """Require a JSON string.

    Args:
        value: The raw decoded value.
        path: Where it came from.

    Returns:
        The string.

    Raises:
        ReplayFormatError: If it is not a string.
    """
    if not isinstance(value, str):
        raise ReplayFormatError(f"{path} must be a string, got {type(value).__name__}")
    return value


def _integer(value: object, path: str) -> int:
    """Require a JSON integer, refusing a boolean.

    Python makes ``bool`` a subclass of ``int``, so ``True`` would otherwise
    pass every later check and be played as a one. JSON keeps the two apart, and
    so does this boundary.

    Args:
        value: The raw decoded value.
        path: Where it came from.

    Returns:
        The integer.

    Raises:
        ReplayFormatError: If it is a boolean, a float, or anything else.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayFormatError(f"{path} must be an integer, got {type(value).__name__}")
    return value


def _number(value: object, path: str) -> float:
    """Require a finite JSON number, refusing a boolean.

    Args:
        value: The raw decoded value.
        path: Where it came from.

    Returns:
        The value as a float; a JSON integer is a valid measurement.

    Raises:
        ReplayFormatError: If it is a boolean or not a number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReplayFormatError(f"{path} must be a number, got {type(value).__name__}")
    return float(value)


def _flag(value: object, path: str) -> bool:
    """Require a JSON boolean.

    Args:
        value: The raw decoded value.
        path: Where it came from.

    Returns:
        The boolean.

    Raises:
        ReplayFormatError: If it is not a boolean. An integer is not accepted
            here either: the two are distinct in JSON.
    """
    if not isinstance(value, bool):
        raise ReplayFormatError(f"{path} must be a boolean, got {type(value).__name__}")
    return value


def _field[T](raw: JsonObject, key: str, path: str, decode: Callable[[object, str], T]) -> T:
    """Decode one required member of an object.

    Args:
        raw: The containing object.
        key: Member name.
        path: Path of the containing object.
        decode: How to decode the member's value.

    Returns:
        The decoded member.

    Raises:
        ReplayFormatError: If the member is missing or does not decode.
    """
    if key not in raw:
        raise ReplayFormatError(f"{_at(path, key)} is missing")
    return decode(raw[key], _at(path, key))


def _list_of[T](decode: Callable[[object, str], T]) -> Callable[[object, str], tuple[T, ...]]:
    """Build a decoder for an array whose items all decode the same way.

    Args:
        decode: Decoder for one item.

    Returns:
        A decoder producing a tuple, so decoded records stay immutable.
    """

    def decode_items(value: object, path: str) -> tuple[T, ...]:
        """Decode every item of the array at ``path``.

        Args:
            value: The raw array.
            path: Where it came from.

        Returns:
            The decoded items, in order.
        """
        return tuple(
            decode(item, _at(path, index)) for index, item in enumerate(_sequence(value, path))
        )

    return decode_items


def _nullable[T](decode: Callable[[object, str], T]) -> Callable[[object, str], T | None]:
    """Build a decoder that also accepts ``null``.

    Args:
        decode: Decoder for a present value.

    Returns:
        A decoder returning ``None`` for JSON ``null``.
    """

    def decode_optional(value: object, path: str) -> T | None:
        """Decode ``value`` unless it is ``null``.

        Args:
            value: The raw value.
            path: Where it came from.

        Returns:
            The decoded value, or ``None``.
        """
        return None if value is None else decode(value, path)

    return decode_optional


def _build[T](path: str, factory: Callable[[], T]) -> T:
    """Construct a domain object, reporting its own validation at this path.

    The domain constructors raise :class:`ValueError` for a value that is
    well-typed but impossible -- a negative slot, a duplicate arrangement, an
    unknown agent kind, a modified profile. Those are format errors from a
    reader's point of view, so they are re-raised as one with the field path
    attached.

    Args:
        path: Path of the value being constructed.
        factory: Calls the constructor with already-decoded arguments.

    Returns:
        The constructed object.

    Raises:
        ReplayFormatError: If the constructor rejects its arguments.
    """
    try:
        return factory()
    except ValueError as error:
        raise ReplayFormatError(f"{path} is invalid: {error}") from error


def _enum[E: Enum](enum_type: type[E], value: object, path: str) -> E:
    """Look one enum member up by value.

    Args:
        enum_type: The enum to look in.
        value: An already type-checked primitive.
        path: Where it came from.

    Returns:
        The matching member.

    Raises:
        ReplayFormatError: If no member has that value.
    """
    try:
        return enum_type(value)
    except ValueError:
        allowed = ", ".join(repr(member.value) for member in enum_type)
        raise ReplayFormatError(f"{path} must be one of {allowed}, got {value!r}") from None


def _rank(value: object, path: str) -> Rank:
    """Decode a rank from its integer value.

    Args:
        value: The raw value; a boolean is not an integer here.
        path: Where it came from.

    Returns:
        The rank.

    Raises:
        ReplayFormatError: If it is not an integer naming a rank.
    """
    return _enum(Rank, _integer(value, path), path)


def _suit(value: object, path: str) -> Suit:
    """Decode a suit from its string value.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The suit.

    Raises:
        ReplayFormatError: If it is not a string naming a suit.
    """
    return _enum(Suit, _text(value, path), path)


def _zone(value: object, path: str) -> Zone:
    """Decode a zone from its string value.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The zone.

    Raises:
        ReplayFormatError: If it is not a string naming a zone.
    """
    return _enum(Zone, _text(value, path), path)


def _phase(value: object, path: str) -> Phase:
    """Decode a phase from its string value.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The phase.

    Raises:
        ReplayFormatError: If it is not a string naming a phase.
    """
    return _enum(Phase, _text(value, path), path)


def _player(value: object, path: str) -> PlayerId:
    """Decode a seat identifier.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The seat, explicitly constructed as a :data:`~shed.engine.PlayerId`.

    Raises:
        ReplayFormatError: If it is not an integer.
    """
    return PlayerId(_integer(value, path))


def _card_id(value: object, path: str) -> CardId:
    """Decode a card identifier.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The identifier, explicitly constructed as a
        :data:`~shed.engine.CardId`.

    Raises:
        ReplayFormatError: If it is not an integer.
    """
    return CardId(_integer(value, path))


def _slot_id(value: object, path: str) -> SlotId:
    """Decode a face-down slot identifier.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The identifier, explicitly constructed as a
        :data:`~shed.engine.SlotId`.

    Raises:
        ReplayFormatError: If it is not an integer.
    """
    return SlotId(_integer(value, path))


def decode_card(value: object, path: str = "card") -> Card:
    """Decode one physical card.

    Args:
        value: The raw value.
        path: Path used in error messages.

    Returns:
        The card, with its rank and suit constructed explicitly.

    Raises:
        ReplayFormatError: If the shape is wrong or the card is impossible, such
            as a joker carrying a suit.
    """
    raw = _mapping(value, path)
    return _build(
        path,
        lambda: Card(
            id=_field(raw, "id", path, _card_id),
            rank=_field(raw, "rank", path, _rank),
            suit=_field(raw, "suit", path, _nullable(_suit)),
        ),
    )


def _tag(raw: JsonObject, path: str) -> str:
    """Read the ``type`` discriminator of a tagged object.

    Args:
        raw: The tagged object.
        path: Where it came from.

    Returns:
        The tag.

    Raises:
        ReplayFormatError: If the tag is missing or is not a string.
    """
    return _field(raw, "type", path, _text)


def decode_constraint(value: object, path: str = "constraint") -> PlayConstraint:
    """Decode a tagged play constraint.

    Args:
        value: The raw value.
        path: Path used in error messages.

    Returns:
        The constraint.

    Raises:
        ReplayFormatError: If the tag is unknown, a field is missing or wrongly
            typed, or the bound is one the constraint refuses, such as a joker.
    """
    raw = _mapping(value, path)
    tag = _tag(raw, path)
    match tag:
        case "unrestricted":
            return Unrestricted()
        case "at_least":
            return _build(path, lambda: AtLeast(rank=_field(raw, "rank", path, _rank)))
        case "at_most":
            return _build(path, lambda: AtMost(rank=_field(raw, "rank", path, _rank)))
        case _:
            raise ReplayFormatError(f"{path}.type is not a known constraint: {tag!r}")


def decode_move(value: object, path: str = "move") -> Move:
    """Decode a tagged move.

    The move is only checked for being well formed. Whether it was legal in the
    position it was recorded for is the engine's decision, made again during
    verification.

    Args:
        value: The raw value.
        path: Path used in error messages.

    Returns:
        The move, built from explicitly constructed ranks, zones, and
        identifiers.

    Raises:
        ReplayFormatError: If the tag is unknown, a field is missing or wrongly
            typed -- a boolean count included -- or a move constructor refuses
            the values.
    """
    raw = _mapping(value, path)
    tag = _tag(raw, path)
    match tag:
        case "arrange":
            ids = _field(raw, "face_up_cards", path, _list_of(_card_id))
            if len(ids) != 3:
                raise ReplayFormatError(
                    f"{path}.face_up_cards must name exactly three cards, got {len(ids)}"
                )
            return _build(path, lambda: Arrange(face_up_cards=(ids[0], ids[1], ids[2])))
        case "play":
            return _build(
                path,
                lambda: Play(
                    source=_field(raw, "source", path, _zone),
                    rank=_field(raw, "rank", path, _rank),
                    count=_field(raw, "count", path, _integer),
                ),
            )
        case "reveal":
            return _build(path, lambda: Reveal(slot=_field(raw, "slot", path, _slot_id)))
        case "pick_up":
            return PickUp()
        case _:
            raise ReplayFormatError(f"{path}.type is not a known move: {tag!r}")


def _decode_public(value: object, path: str) -> PublicPlayerState:
    """Decode one player's public state.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The public entry.

    Raises:
        ReplayFormatError: If a field is missing or wrongly typed.
    """
    raw = _mapping(value, path)
    return PublicPlayerState(
        player=_field(raw, "player", path, _player),
        hand_count=_field(raw, "hand_count", path, _integer),
        face_up=_field(raw, "face_up", path, _list_of(decode_card)),
        face_down_slots=_field(raw, "face_down_slots", path, _list_of(_slot_id)),
    )


def _decode_outcome(value: object, path: str) -> Outcome:
    """Decode a game outcome.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The outcome.

    Raises:
        ReplayFormatError: If the winner is missing or is not an integer.
    """
    raw = _mapping(value, path)
    return Outcome(winner=_field(raw, "winner", path, _player))


def decode_event(value: object, path: str = "event") -> ObservedEvent:
    """Decode a tagged observed event.

    Args:
        value: The raw value.
        path: Path used in error messages.

    Returns:
        The event, with cards, zones, and reasons constructed explicitly. A
        private event whose ``cards`` are ``null`` decodes to the filtered
        shape, keeping its public count.

    Raises:
        ReplayFormatError: If the tag is unknown or a field is missing or
            wrongly typed.
    """
    raw = _mapping(value, path)
    tag = _tag(raw, path)
    cards = _list_of(decode_card)
    match tag:
        case "game_started":
            return GameStarted(
                dealer=_field(raw, "dealer", path, _player),
                seat_order=_field(raw, "seat_order", path, _list_of(_player)),
                players=_field(raw, "players", path, _list_of(_decode_public)),
            )
        case "hand_dealt":
            return HandDealt(
                player=_field(raw, "player", path, _player),
                count=_field(raw, "count", path, _integer),
                cards=_field(raw, "cards", path, _nullable(cards)),
            )
        case "arrangement_committed":
            return ArrangementCommitted(
                player=_field(raw, "player", path, _player),
                face_up=_field(raw, "face_up", path, cards),
            )
        case "cards_played":
            return CardsPlayed(
                player=_field(raw, "player", path, _player),
                source=_field(raw, "source", path, _zone),
                cards=_field(raw, "cards", path, cards),
            )
        case "card_revealed":
            return CardRevealed(
                player=_field(raw, "player", path, _player),
                slot=_field(raw, "slot", path, _slot_id),
                card=_field(raw, "card", path, decode_card),
                playable=_field(raw, "playable", path, _flag),
            )
        case "cards_drawn":
            return CardsDrawn(
                player=_field(raw, "player", path, _player),
                count=_field(raw, "count", path, _integer),
                cards=_field(raw, "cards", path, _nullable(cards)),
            )
        case "pile_picked_up":
            return PilePickedUp(
                player=_field(raw, "player", path, _player),
                cards=_field(raw, "cards", path, cards),
            )
        case "pile_burned":
            return PileBurned(
                player=_field(raw, "player", path, _player),
                cards=_field(raw, "cards", path, cards),
                reason=_enum(BurnReason, _field(raw, "reason", path, _text), _at(path, "reason")),
            )
        case "game_ended":
            return GameEnded(outcome=_field(raw, "outcome", path, _decode_outcome))
        case _:
            raise ReplayFormatError(f"{path}.type is not a known event: {tag!r}")


def _decode_failure(value: object, path: str) -> WorkerFailed:
    """Decode a worker failure record.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The failure.

    Raises:
        ReplayFormatError: If a field is missing or is not a string.
    """
    raw = _mapping(value, path)
    return WorkerFailed(
        exception=_field(raw, "exception", path, _text),
        message=_field(raw, "message", path, _text),
    )


def _decode_turn(value: object, path: str) -> TurnRecord:
    """Decode one decision's selection diagnostics.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The turn record.

    Raises:
        ReplayFormatError: If a field is missing or wrongly typed.
    """
    raw = _mapping(value, path)
    return TurnRecord(
        decision_id=_field(raw, "decision_id", path, _integer),
        player=_field(raw, "player", path, _player),
        phase=_field(raw, "phase", path, _phase),
        move=_field(raw, "move", path, decode_move),
        reason=_enum(CloseReason, _field(raw, "reason", path, _text), _at(path, "reason")),
        used_fallback=_field(raw, "used_fallback", path, _flag),
        accepted=_field(raw, "accepted", path, _integer),
        rejected=_field(raw, "rejected", path, _integer),
        failure=_field(raw, "failure", path, _nullable(_decode_failure)),
        budget_seconds=_field(raw, "budget_seconds", path, _number),
        selection_seconds=_field(raw, "selection_seconds", path, _number),
        cleanup_seconds=_field(raw, "cleanup_seconds", path, _number),
        agent_seed=_field(raw, "agent_seed", path, _integer),
    )


def _decode_decision(value: object, path: str) -> AppliedDecision:
    """Decode one applied decision.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The decision and the events it resolved into.

    Raises:
        ReplayFormatError: If a field is missing or wrongly typed.
    """
    raw = _mapping(value, path)
    return AppliedDecision(
        turn=_field(raw, "turn", path, _decode_turn),
        events=_field(raw, "events", path, _list_of(decode_event)),
    )


def _decode_player_position(value: object, path: str) -> PlayerPosition:
    """Decode one seat's entry in a final position.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The seat's cards, by identifier.

    Raises:
        ReplayFormatError: If a field is missing or wrongly typed, or a
            face-down entry is not a slot/card pair.
    """
    raw = _mapping(value, path)
    return PlayerPosition(
        player=_field(raw, "player", path, _player),
        hand=_field(raw, "hand", path, _list_of(_card_id)),
        face_up=_field(raw, "face_up", path, _list_of(_card_id)),
        face_down=_field(raw, "face_down", path, _list_of(_decode_slot_entry)),
    )


def _decode_slot_entry(value: object, path: str) -> tuple[SlotId, CardId]:
    """Decode one face-down slot/card pair.

    Args:
        value: The raw value, a two-element array.
        path: Where it came from.

    Returns:
        The slot and the card still hidden in it.

    Raises:
        ReplayFormatError: If it is not a pair of integers.
    """
    pair = _sequence(value, path)
    if len(pair) != 2:
        raise ReplayFormatError(f"{path} must be a [slot, card] pair, got {len(pair)} items")
    return _slot_id(pair[0], _at(path, 0)), _card_id(pair[1], _at(path, 1))


def _decode_position(value: object, path: str) -> FinalPosition:
    """Decode the digest of the position a match stopped in.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The digest.

    Raises:
        ReplayFormatError: If a field is missing or wrongly typed.
    """
    raw = _mapping(value, path)
    return FinalPosition(
        phase=_field(raw, "phase", path, _phase),
        current_player=_field(raw, "current_player", path, _nullable(_player)),
        current_ply=_field(raw, "current_ply", path, _integer),
        constraint=_field(raw, "constraint", path, decode_constraint),
        draw_pile=_field(raw, "draw_pile", path, _list_of(_card_id)),
        discard_pile=_field(raw, "discard_pile", path, _list_of(_card_id)),
        burned_cards=_field(raw, "burned_cards", path, _list_of(_card_id)),
        players=_field(raw, "players", path, _list_of(_decode_player_position)),
    )


def _decode_rules(value: object, path: str) -> RulesConfig:
    """Decode the rules profile and check this release implements it.

    The profile identifier is checked before the object is built, so an unknown
    profile is refused by name rather than through a field comparison. The
    profile's own validation then refuses a document that claims ``shed-v1``
    while carrying different numbers.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The validated profile.

    Raises:
        ReplayFormatError: If a field is missing or wrongly typed, the profile
            is not supported, or its values are not the fixed profile's.
    """
    raw = _mapping(value, path)
    identifier = _field(raw, "id", path, _text)
    if identifier not in SUPPORTED_PROFILES:
        supported = ", ".join(repr(name) for name in SUPPORTED_PROFILES)
        raise ReplayFormatError(
            f"{_at(path, 'id')} is rules profile {identifier!r}; this release replays {supported}"
        )
    config = _build(
        path,
        lambda: RulesConfig(
            id=identifier,
            min_players=_field(raw, "min_players", path, _integer),
            max_players=_field(raw, "max_players", path, _integer),
            joker_count=_field(raw, "joker_count", path, _integer),
            initial_hand_size=_field(raw, "initial_hand_size", path, _integer),
            initial_face_up_count=_field(raw, "initial_face_up_count", path, _integer),
            initial_face_down_count=_field(raw, "initial_face_down_count", path, _integer),
            refill_target=_field(raw, "refill_target", path, _integer),
        ),
    )
    _build(path, config.validate)
    return config


def _decode_config(value: object, path: str) -> MatchConfig:
    """Decode the timing, limit, and failure policy.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The configuration, whose own validation refuses an unusable budget.

    Raises:
        ReplayFormatError: If a field is missing or wrongly typed, or the
            configuration is one no match could have run under.
    """
    raw = _mapping(value, path)
    return _build(
        path,
        lambda: MatchConfig(
            seconds_per_turn=_field(raw, "seconds_per_turn", path, _number),
            max_play_decisions=_field(raw, "max_play_decisions", path, _integer),
            fallback_seed=_field(raw, "fallback_seed", path, _integer),
            agent_seed=_field(raw, "agent_seed", path, _integer),
            strict_failures=_field(raw, "strict_failures", path, _flag),
        ),
    )


def _decode_spec(value: object, path: str) -> AgentSpec:
    """Decode one participant specification.

    Args:
        value: The raw value.
        path: Where it came from.

    Returns:
        The specification, whose own validation refuses an unknown agent kind.

    Raises:
        ReplayFormatError: If a field is missing or wrongly typed, or the kind
            is not one this release builds.
    """
    raw = _mapping(value, path)
    return _build(
        path,
        lambda: AgentSpec(
            kind=_field(raw, "kind", path, _text),
            name=_field(raw, "name", path, _text),
        ),
    )


@dataclass(frozen=True, slots=True)
class ReplayHeader:
    """Provenance of a replay document.

    Attributes:
        schema: Replay schema version; one of :data:`SUPPORTED_SCHEMAS`.
        package_version: Version of ``shed`` that wrote the file.
        python_version: Interpreter the match ran on.
        source_revision: Commit the match was played at, when it was knowable.
    """

    schema: int
    package_version: str
    python_version: str
    source_revision: str | None


@dataclass(frozen=True, slots=True)
class Replay:
    """A decoded replay: everything needed to reproduce one match.

    The fields mirror what the runner produced, as the same typed records. The
    deck sits beside the metadata rather than inside it because the runner deals
    from a seed while a replay deals from the recorded order.

    Attributes:
        header: Provenance of the document.
        metadata: How the match was set up.
        deck: The shuffled deck the match was dealt from, in deal order.
        status: How the match ended.
        outcome: The winner, when the rules finished the game.
        failure: Recorded detail for a non-finished status.
        play_decisions: Applied PLAY decisions the match counted.
        initial_events: The full events describing the deal.
        decisions: The applied-decision stream, in order.
        unapplied_turns: Selections that were never applied, kept out of the
            applied stream so a replay cannot play a move the match did not.
        final_position: Digest of the position the match stopped in.
    """

    header: ReplayHeader
    metadata: MatchMetadata
    deck: tuple[Card, ...]
    status: MatchStatus
    outcome: Outcome | None
    failure: str | None
    play_decisions: int
    initial_events: tuple[ObservedEvent, ...]
    decisions: tuple[AppliedDecision, ...]
    unapplied_turns: tuple[TurnRecord, ...]
    final_position: FinalPosition

    @property
    def turns(self) -> tuple[TurnRecord, ...]:
        """Return every selection the match made, applied or not, in order.

        Returns:
            The applied decisions' records merged with the unapplied ones,
            ordered by decision identifier.
        """
        records = [decision.turn for decision in self.decisions]
        records.extend(self.unapplied_turns)
        return tuple(sorted(records, key=lambda turn: turn.decision_id))


def _decode_header(raw: JsonObject, path: str) -> ReplayHeader:
    """Decode the document header and check the schema is supported.

    Args:
        raw: The whole document.
        path: Path of the document, for error messages.

    Returns:
        The header.

    Raises:
        ReplayFormatError: If the schema is missing, is not an integer, or is a
            version this release does not read.
    """
    schema = _field(raw, "schema", path, _integer)
    if schema not in SUPPORTED_SCHEMAS:
        supported = ", ".join(str(version) for version in SUPPORTED_SCHEMAS)
        raise ReplayFormatError(
            f"{_at(path, 'schema')} is replay schema {schema}; this release reads {supported}"
        )
    return ReplayHeader(
        schema=schema,
        package_version=_field(raw, "package_version", path, _text),
        python_version=_field(raw, "python_version", path, _text),
        source_revision=_field(raw, "source_revision", path, _nullable(_text)),
    )


def decode_replay(document: object, path: str = "replay") -> Replay:
    """Decode a whole replay document.

    Everything external is settled here: the schema, the rules profile, the
    shape of every record, the primitive type of every field, and the
    construction of every domain object. What comes back is as typed as anything
    the runner built, so :func:`verify_replay` and the scripts never touch raw
    JSON.

    Args:
        document: The parsed JSON document.
        path: Path used in error messages.

    Returns:
        The decoded replay.

    Raises:
        ReplayFormatError: If the document is not a replay this release reads.
    """
    raw = _mapping(document, path)
    header = _decode_header(raw, path)
    rules = _field(raw, "rules", path, _decode_rules)

    setup_path = _at(path, "setup")
    setup = _field(raw, "setup", path, _mapping)
    metadata = MatchMetadata(
        rules=rules,
        player_count=_field(setup, "player_count", setup_path, _integer),
        dealer=_field(setup, "dealer", setup_path, _player),
        deal_seed=_field(setup, "deal_seed", setup_path, _integer),
        agents=_field(setup, "agents", setup_path, _list_of(_decode_spec)),
        config=_field(setup, "config", setup_path, _decode_config),
    )

    result_path = _at(path, "result")
    result = _field(raw, "result", path, _mapping)
    status_path = _at(result_path, "status")
    return Replay(
        header=header,
        metadata=metadata,
        deck=_field(setup, "deck", setup_path, _list_of(decode_card)),
        status=_enum(MatchStatus, _field(result, "status", result_path, _text), status_path),
        outcome=_field(result, "outcome", result_path, _nullable(_decode_outcome)),
        failure=_field(result, "failure", result_path, _nullable(_text)),
        play_decisions=_field(result, "play_decisions", result_path, _integer),
        initial_events=_field(result, "initial_events", result_path, _list_of(decode_event)),
        decisions=_field(result, "decisions", result_path, _list_of(_decode_decision)),
        unapplied_turns=_field(result, "unapplied_turns", result_path, _list_of(_decode_turn)),
        final_position=_field(result, "final_position", result_path, _decode_position),
    )


def _reject_constant(literal: str) -> object:
    """Refuse the non-standard JSON literals Python's parser accepts by default.

    Args:
        literal: The literal found, one of ``NaN``, ``Infinity``, ``-Infinity``.

    Raises:
        ReplayFormatError: Always. A replay is ordinary JSON, and a measurement
            that is not a number is not a measurement.
    """
    raise ReplayFormatError(f"{literal} is not valid JSON in a replay")


def read_replay(path: Path) -> Replay:
    """Read and decode a replay file.

    Args:
        path: The UTF-8 JSON file to read.

    Returns:
        The decoded replay.

    Raises:
        ReplayFormatError: If the file is not valid JSON or is not a replay this
            release reads.
        OSError: If the file cannot be read at all.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ReplayFormatError(f"{path} is not valid JSON: {error}") from error
    except UnicodeDecodeError as error:
        raise ReplayFormatError(f"{path} is not valid UTF-8: {error}") from error
    return decode_replay(document, path=str(path))


# --------------------------------------------------------------------------- #
# Verification: replaying the recorded decisions through the engine.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ReplayCheck:
    """What replaying a recorded match found.

    Attributes:
        problems: One line per disagreement, in the order they were found. A
            verified replay has none.
        applied: Recorded decisions that the engine accepted before the check
            stopped.
        outcome: The outcome the replayed game reached, which a verified replay
            shares with the recording.
    """

    problems: tuple[str, ...]
    applied: int
    outcome: Outcome | None

    @property
    def ok(self) -> bool:
        """Whether the replay reproduced the recording exactly."""
        return not self.problems


def verify_replay(replay: Replay) -> ReplayCheck:
    """Replay a recorded match through the engine and compare what happens.

    The opening position is dealt from the recorded deck order with
    :func:`~shed.engine.deal_initial_state`, then every applied decision is
    handed to :meth:`~shed.engine.GameState.apply_move` in order. No agent is
    built and no worker is started: the recorded moves are the input, so the
    result is deterministic even though the timed match that produced them was
    not, and the original budgets are irrelevant.

    Selections the match never applied are not replayed. A truncated or aborted
    match therefore verifies against the position it really stopped in.

    Args:
        replay: A decoded replay.

    Returns:
        The check. Every disagreement is reported rather than raised, including
        a recorded move the engine now refuses: an illegal recording is a failed
        verification, not a crash.
    """
    problems: list[str] = []
    try:
        state = deal_initial_state(
            list(replay.deck),
            player_count=replay.metadata.player_count,
            dealer=replay.metadata.dealer,
            config=replay.metadata.rules,
        )
    except ValueError as error:
        return ReplayCheck(
            problems=(f"the recorded deal cannot be rebuilt: {error}",), applied=0, outcome=None
        )

    if state.initial_events() != replay.initial_events:
        problems.append("the recorded deal events differ from the dealt position")

    applied = 0
    for decision in replay.decisions:
        turn = decision.turn
        label = f"decision {turn.decision_id}"
        if state.is_finished:
            problems.append(f"{label} was recorded after the game had already finished")
            break
        if turn.player != state.current_player:
            problems.append(
                f"{label} was recorded for player {turn.player}, "
                f"but player {state.current_player} is to decide"
            )
            break
        if turn.phase is not state.phase:
            problems.append(
                f"{label} was recorded in phase {turn.phase.value}, "
                f"but the position is in phase {state.phase.value}"
            )
            break
        try:
            transition = state.apply_move(turn.move)
        except (IllegalMoveError, StateInvariantError) as error:
            problems.append(f"{label} is not playable: {type(error).__name__}: {error}")
            break
        applied += 1
        if transition.events != decision.events:
            problems.append(f"{label} resolved into different events than were recorded")
            break

    problems.extend(_compare_conclusion(replay, state, applied))
    return ReplayCheck(problems=tuple(problems), applied=applied, outcome=state.outcome)


def _compare_conclusion(replay: Replay, state: GameState, applied: int) -> list[str]:
    """Compare how the replayed game ended with what was recorded.

    Args:
        replay: The recording.
        state: The state the recorded decisions led to.
        applied: How many decisions were applied.

    Returns:
        One line per disagreement about completeness, status, outcome, ply
        count, or final position; empty when they agree.
    """
    problems: list[str] = []
    if applied != len(replay.decisions):
        return problems  # The stream stopped early; later comparisons are noise.

    finished = replay.status is MatchStatus.FINISHED
    if finished is not state.is_finished:
        problems.append(
            f"the recorded status is {replay.status.value}, "
            f"but the replayed game {'finished' if state.is_finished else 'did not finish'}"
        )
    if state.outcome != replay.outcome:
        problems.append(
            f"the replayed outcome {state.outcome} is not the recorded {replay.outcome}"
        )
    if state.current_ply != replay.play_decisions:
        problems.append(
            f"the replayed game resolved {state.current_ply} play decisions, "
            f"but {replay.play_decisions} were recorded"
        )
    if FinalPosition.from_state(state) != replay.final_position:
        problems.append("the replayed final position differs from the recorded one")
    return problems
