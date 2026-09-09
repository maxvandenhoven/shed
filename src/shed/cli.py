"""Shared pieces of the command-line entry points.

The scripts under ``scripts/`` are argument parsers and nothing else. Everything
they would otherwise duplicate lives here: turning ``--agents random greedy``
into a seated lineup, validating numeric arguments, narrating a match as public
actions, and summarizing how one ended. Both the play command and the replay
command use these, and the gauntlet command will.

Narration has two settings, and the default is the careful one. Under
:attr:`Visibility.PUBLIC` a private event is reported by its public count alone
-- the same thing an opponent at the table knows -- and everything else printed
is public by construction: cards already on the table, the pile, a revealed
card. Under :attr:`Visibility.OMNISCIENT` the identities are printed too, and
:func:`describe_position` will dump the authoritative position on top.

Both are views of trusted data. The events a runner records and the events a
replay file holds always carry hidden identities; ``PUBLIC`` is redaction
applied on the way to the console, not a limit on what is available. The
information boundary that matters is elsewhere and is unaffected by any of this:
an agent sees a :class:`~shed.engine.PlayerView`, which never contains another
seat's cards whatever the operator asked to print.
"""

from __future__ import annotations

import argparse
import signal
from collections.abc import Iterable, Sequence
from enum import Enum

from shed.agents import AGENT_KINDS, AgentSpec
from shed.engine import (
    DEFAULT_RULES,
    ArrangementCommitted,
    AtLeast,
    AtMost,
    Card,
    CardRevealed,
    CardsDrawn,
    CardsPlayed,
    GameEnded,
    GameStarted,
    HandDealt,
    ObservedEvent,
    Outcome,
    PileBurned,
    PilePickedUp,
    PlayConstraint,
    PlayerId,
    Rank,
    Unrestricted,
)
from shed.match import FinalPosition, MatchMetadata, MatchResult, MatchStatus, TurnRecord
from shed.replay import Replay

__all__ = [
    "Visibility",
    "build_lineup",
    "card_text",
    "describe_constraint",
    "describe_event",
    "describe_position",
    "match_summary",
    "narrate",
    "positive_count",
    "positive_seconds",
    "replay_summary",
    "restore_default_sigpipe",
]

RANK_TEXT: dict[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
    Rank.JOKER: "JK",
}
"""Single-character rank labels, so a batch of cards reads as one short group."""


class Visibility(Enum):
    """How much of a recorded game the console is allowed to show.

    Attributes:
        PUBLIC: What an onlooker at the table knows. Private events -- the deal
            and every replenishment draw -- are reported by count only. This is
            the default everywhere.
        OMNISCIENT: Everything the record holds, identities included. An opt-in
            operator view of a match that has already been played; it changes
            nothing about what an agent is given while playing.
    """

    PUBLIC = "public"
    OMNISCIENT = "omniscient"


def card_text(card: Card) -> str:
    """Render one card compactly.

    Deliberately ASCII -- ``Th`` rather than a suit symbol -- because this goes
    to whatever console the user has, and a replay summary is not worth an
    encoding failure.

    Args:
        card: The card to render.

    Returns:
        Rank and suit initial, such as ``7d`` or ``Ts``, or ``JK`` for a joker.
    """
    label = RANK_TEXT[card.rank]
    return label if card.suit is None else f"{label}{card.suit.value[0]}"


def _cards_text(cards: Iterable[Card]) -> str:
    """Render a run of cards.

    Args:
        cards: The cards, in the order they should be read.

    Returns:
        The rendered cards separated by spaces.
    """
    return " ".join(card_text(card) for card in cards)


def _count_text(count: int, noun: str) -> str:
    """Render a count with its noun, pluralized in the ordinary English way.

    Args:
        count: How many.
        noun: Singular form of the noun.

    Returns:
        The count and the noun, such as ``1 card`` or ``3 cards``.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _private_text(count: int, cards: tuple[Card, ...] | None, visibility: Visibility) -> str:
    """Render the cards of a private event, or only how many there were.

    This is the single place a hidden identity can reach the console, which is
    what keeps the two visibilities from becoming two narrators.

    Args:
        count: How many cards the event moved; public either way.
        cards: The identities, or ``None`` in an event already filtered for
            somebody who may not know them.
        visibility: How much the console may show.

    Returns:
        The count alone, or the count and the cards.
    """
    counted = _count_text(count, "card")
    if visibility is Visibility.PUBLIC or cards is None:
        return counted
    return f"{counted}: {_cards_text(cards)}"


def describe_event(event: ObservedEvent, visibility: Visibility = Visibility.PUBLIC) -> str:
    """Describe one event as a line of commentary.

    Only the two private events read ``visibility`` at all: everything else here
    was public when it happened. An event that was already filtered keeps its
    identities hidden even under :attr:`Visibility.OMNISCIENT`, because they are
    genuinely not in it.

    Args:
        event: A full internal event.
        visibility: How much of it to show. The default is what an onlooker at
            the table would have seen.

    Returns:
        One line describing what happened.
    """
    match event:
        case GameStarted(dealer=dealer, seat_order=seats, players=players):
            shown = ", ".join(
                f"player {public.player} shows {_cards_text(public.face_up)}" for public in players
            )
            return f"player {dealer} deals to {len(seats)} seats; {shown}"
        case HandDealt(player=player, count=count, cards=cards):
            return f"player {player} is dealt {_private_text(count, cards, visibility)}"
        case ArrangementCommitted(player=player, face_up=face_up):
            return f"player {player} settles on {_cards_text(face_up)} face up"
        case CardsPlayed(player=player, source=source, cards=cards):
            return f"player {player} plays {_cards_text(cards)} from {source.value}"
        case CardRevealed(player=player, slot=slot, card=card, playable=playable):
            verdict = "playable" if playable else "not playable"
            return f"player {player} reveals {card_text(card)} in slot {slot}: {verdict}"
        case CardsDrawn(player=player, count=count, cards=cards):
            return f"player {player} draws {_private_text(count, cards, visibility)}"
        case PilePickedUp(player=player, cards=cards):
            return (
                f"player {player} picks up {_count_text(len(cards), 'card')}: {_cards_text(cards)}"
            )
        case PileBurned(player=player, cards=cards, reason=reason):
            return f"player {player} burns {_count_text(len(cards), 'card')} ({reason.value})"
        case GameEnded(outcome=outcome):
            return f"player {outcome.winner} wins"


def narrate(
    events: Iterable[ObservedEvent], visibility: Visibility = Visibility.PUBLIC
) -> list[str]:
    """Describe a run of events as commentary.

    Args:
        events: Full internal events in resolution order.
        visibility: How much of them to show.

    Returns:
        One line per event, in the same order.
    """
    return [describe_event(event, visibility) for event in events]


def describe_constraint(constraint: PlayConstraint) -> str:
    """Describe what the next ordinary rank has to satisfy.

    Args:
        constraint: The restriction standing on the pile.

    Returns:
        A short phrase, such as ``at least 9`` or ``unrestricted``.
    """
    match constraint:
        case Unrestricted():
            return "unrestricted"
        case AtLeast(rank=rank):
            return f"at least {RANK_TEXT[rank]}"
        case AtMost(rank=rank):
            return f"at most {RANK_TEXT[rank]}"


def _by_id(deck: Iterable[Card]) -> dict[int, Card]:
    """Index a deck so a position's identifiers can be rendered as cards.

    Args:
        deck: The recorded deck, in any order.

    Returns:
        Every card, keyed by identifier.
    """
    return {int(card.id): card for card in deck}


def _zone_text(identifiers: Iterable[int], cards: dict[int, Card]) -> str:
    """Render one zone of a position.

    Args:
        identifiers: Card identifiers in their stored order.
        cards: Index built by :func:`_by_id`.

    Returns:
        The rendered cards, or ``-`` for an empty zone.
    """
    return " ".join(card_text(cards[identifier]) for identifier in identifiers) or "-"


def describe_position(position: FinalPosition, deck: Iterable[Card]) -> list[str]:
    """Describe an authoritative position in full, hidden cards included.

    This is the omniscient counterpart to :func:`describe_event`: where the
    narration says what happened, this says where everything ended up, face-down
    slots and the undrawn deck included. Print it only when the operator asked
    for it.

    Args:
        position: The digest to describe.
        deck: The recorded deck, which resolves the digest's identifiers.

    Returns:
        One header line, three lines for the shared piles, and one line per
        seat.

    Raises:
        KeyError: If the position names a card the deck does not contain, which
            means the two came from different matches.
    """
    cards = _by_id(deck)
    actor = "nobody" if position.current_player is None else f"player {position.current_player}"
    lines = [
        f"position: {position.phase.value} | ply {position.current_ply} | to act: {actor} | "
        f"constraint: {describe_constraint(position.constraint)}",
        f"  draw pile ({len(position.draw_pile)}, next draw last): "
        f"{_zone_text(position.draw_pile, cards)}",
        f"  discard ({len(position.discard_pile)}): {_zone_text(position.discard_pile, cards)}",
        f"  burned ({len(position.burned_cards)}): {_zone_text(position.burned_cards, cards)}",
    ]
    for player in position.players:
        face_down = (
            " ".join(f"{slot}={card_text(cards[card_id])}" for slot, card_id in player.face_down)
            or "-"
        )
        lines.append(
            f"  player {player.player}: hand {_zone_text(player.hand, cards)} | "
            f"face up {_zone_text(player.face_up, cards)} | face down {face_down}"
        )
    return lines


def build_lineup(kinds: Sequence[str]) -> dict[PlayerId, AgentSpec]:
    """Seat one participant per requested agent kind.

    Labels are derived from the seat, so a lineup may repeat a kind -- two
    greedy agents become ``greedy-0`` and ``greedy-1`` -- and every participant
    still has a distinct name in results.

    Args:
        kinds: Agent kinds in seat order, seat ``0`` first.

    Returns:
        The lineup, keyed by seat.

    Raises:
        ValueError: If the table size is outside the profile's range or a kind
            is not one the factory builds.
    """
    if not DEFAULT_RULES.min_players <= len(kinds) <= DEFAULT_RULES.max_players:
        raise ValueError(
            f"{DEFAULT_RULES.id} supports {DEFAULT_RULES.min_players}-"
            f"{DEFAULT_RULES.max_players} players, got {len(kinds)}"
        )
    return {
        PlayerId(seat): AgentSpec(kind=kind, name=f"{kind}-{seat}")
        for seat, kind in enumerate(kinds)
    }


def restore_default_sigpipe() -> None:
    """Let a closed pipe end the process quietly, as a Unix tool should.

    Python turns ``SIGPIPE`` into a :class:`BrokenPipeError`, so a long action
    log piped into ``head`` ends in a traceback instead of simply stopping.
    Restoring the default disposition makes these commands behave like every
    other program in a pipeline. Call it from a script's main guard only: it
    changes process-wide signal state and has no business running on import.

    On a platform without ``SIGPIPE`` -- Windows -- this does nothing.
    """
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def positive_seconds(text: str) -> float:
    """Parse a strictly positive, finite duration for ``argparse``.

    Args:
        text: The raw argument.

    Returns:
        The duration in seconds.

    Raises:
        argparse.ArgumentTypeError: If it is not a number, or is not positive
            and finite. Infinity and NaN are rejected here so the failure names
            the option rather than surfacing from the match configuration.
    """
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not value > 0.0 or value == float("inf"):
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive, finite number of seconds")
    return value


def positive_count(text: str) -> int:
    """Parse a strictly positive integer for ``argparse``.

    Args:
        text: The raw argument.

    Returns:
        The count.

    Raises:
        argparse.ArgumentTypeError: If it is not an integer, or is not positive.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer")
    return value


def agent_kinds_help() -> str:
    """Describe the agent kinds a lineup may name.

    Returns:
        A comma-separated list for an option's help text.
    """
    return ", ".join(AGENT_KINDS)


def _participant(agents: Sequence[AgentSpec], seat: PlayerId) -> str:
    """Name the participant in one seat.

    Args:
        agents: Participants in seat order.
        seat: The seat to name.

    Returns:
        The participant's label, or a placeholder when the recorded lineup does
        not cover that seat.
    """
    return agents[seat].name if 0 <= seat < len(agents) else "unknown"


def _status_line(
    status: MatchStatus,
    outcome: Outcome | None,
    failure: str | None,
    agents: Sequence[AgentSpec],
) -> str:
    """Summarize how a match ended.

    Args:
        status: The runner's terminal status.
        outcome: The winner, when the rules finished the game.
        failure: Recorded detail for a non-finished status.
        agents: Participants in seat order, for naming a winner.

    Returns:
        One line naming the status and either the winner or the failure detail.
        A truncated or aborted match is never reported as having a winner.
    """
    if outcome is not None:
        winner = f"player {outcome.winner} ({_participant(agents, outcome.winner)}) wins"
        return f"status: {status.value} - {winner}"
    return f"status: {status.value} - {failure or 'no winner'}"


def _diagnostics_line(turns: Sequence[TurnRecord], play_decisions: int) -> str:
    """Summarize the selection diagnostics of a whole match.

    Args:
        turns: Every selection the match made, applied or not.
        play_decisions: Applied PLAY decisions.

    Returns:
        One line with the decision counts and how often the agents misbehaved.
    """
    fallbacks = sum(turn.used_fallback for turn in turns)
    rejected = sum(turn.rejected for turn in turns)
    failures = sum(turn.failure is not None for turn in turns)
    return (
        f"decisions: {len(turns)} ({play_decisions} play) | fallbacks: {fallbacks} | "
        f"rejected submissions: {rejected} | worker failures: {failures}"
    )


def _summary(
    metadata: MatchMetadata,
    status: MatchStatus,
    outcome: Outcome | None,
    failure: str | None,
    turns: Sequence[TurnRecord],
    play_decisions: int,
) -> list[str]:
    """Build the shared console summary of one match.

    Args:
        metadata: How the match was set up.
        status: The runner's terminal status.
        outcome: The winner, when the rules finished the game.
        failure: Recorded detail for a non-finished status.
        turns: Every selection the match made, applied or not.
        play_decisions: Applied PLAY decisions.

    Returns:
        The summary lines, without a trailing newline. The deal seed is included
        because these are trusted post-match outputs, not agent-facing ones.
    """
    lines = [
        f"{metadata.rules.id} | {metadata.player_count} players | "
        f"deal seed {metadata.deal_seed} | dealer {metadata.dealer} | "
        f"{metadata.config.seconds_per_turn:g}s per decision"
    ]
    lines.extend(
        f"  seat {seat}: {spec.kind} ({spec.name})" for seat, spec in enumerate(metadata.agents)
    )
    lines.append(_diagnostics_line(turns, play_decisions))
    lines.append(_status_line(status, outcome, failure, metadata.agents))
    return lines


def match_summary(result: MatchResult) -> list[str]:
    """Summarize a match the runner just played.

    Args:
        result: The match result.

    Returns:
        The summary lines.
    """
    return _summary(
        result.metadata,
        result.status,
        result.outcome,
        result.failure,
        result.turns,
        result.play_decisions,
    )


def replay_summary(replay: Replay) -> list[str]:
    """Summarize a match read back from a replay file.

    Args:
        replay: The decoded replay.

    Returns:
        The summary lines, preceded by the document's provenance.
    """
    header = replay.header
    revision = header.source_revision or "unknown revision"
    lines = [
        f"replay schema {header.schema} | shed {header.package_version} | "
        f"python {header.python_version} | {revision}"
    ]
    lines.extend(
        _summary(
            replay.metadata,
            replay.status,
            replay.outcome,
            replay.failure,
            replay.turns,
            replay.play_decisions,
        )
    )
    return lines
