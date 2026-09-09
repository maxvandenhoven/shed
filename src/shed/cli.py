"""Shared pieces of the command-line entry points.

The scripts under ``scripts/`` are argument parsers and nothing else. Everything
they would otherwise duplicate lives here: turning ``--agents random greedy``
into a seated lineup, validating numeric arguments, narrating a match as public
actions, and summarizing how one ended. Both the play command and the replay
command use these, and the gauntlet command will.

Narration is deliberately conservative about hidden information. A replay file
is a trusted artifact that holds every face-down identity and every private
draw, so a summary written from one could dump the whole game. It does not:
:func:`describe_event` reports private events by their public count alone, the
same thing an opponent at the table knows. Everything else it prints is public
by construction -- cards already on the table, the pile, a revealed card.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence

from shed.agents import AGENT_KINDS, AgentSpec
from shed.engine import (
    DEFAULT_RULES,
    ArrangementCommitted,
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
    PlayerId,
    Rank,
)
from shed.match import MatchMetadata, MatchResult, MatchStatus, TurnRecord
from shed.replay import Replay

__all__ = [
    "build_lineup",
    "card_text",
    "describe_event",
    "match_summary",
    "narrate",
    "positive_count",
    "positive_seconds",
    "replay_summary",
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


def describe_event(event: ObservedEvent) -> str:
    """Describe one event as a line of public commentary.

    Private events -- the deal and every replenishment draw -- are reported by
    count only. Their identities are in the replay, and this function is what
    keeps them out of a console summary.

    Args:
        event: A full internal event.

    Returns:
        One line describing what an onlooker at the table would have seen.
    """
    match event:
        case GameStarted(dealer=dealer, seat_order=seats, players=players):
            shown = ", ".join(
                f"player {public.player} shows {_cards_text(public.face_up)}" for public in players
            )
            return f"player {dealer} deals to {len(seats)} seats; {shown}"
        case HandDealt(player=player, count=count):
            return f"player {player} is dealt {_count_text(count, 'card')}"
        case ArrangementCommitted(player=player, face_up=face_up):
            return f"player {player} settles on {_cards_text(face_up)} face up"
        case CardsPlayed(player=player, source=source, cards=cards):
            return f"player {player} plays {_cards_text(cards)} from {source.value}"
        case CardRevealed(player=player, slot=slot, card=card, playable=playable):
            verdict = "playable" if playable else "not playable"
            return f"player {player} reveals {card_text(card)} in slot {slot}: {verdict}"
        case CardsDrawn(player=player, count=count):
            return f"player {player} draws {_count_text(count, 'card')}"
        case PilePickedUp(player=player, cards=cards):
            return (
                f"player {player} picks up {_count_text(len(cards), 'card')}: {_cards_text(cards)}"
            )
        case PileBurned(player=player, cards=cards, reason=reason):
            return f"player {player} burns {_count_text(len(cards), 'card')} ({reason.value})"
        case GameEnded(outcome=outcome):
            return f"player {outcome.winner} wins"


def narrate(events: Iterable[ObservedEvent]) -> list[str]:
    """Describe a run of events as public commentary.

    Args:
        events: Full internal events in resolution order.

    Returns:
        One line per event, in the same order.
    """
    return [describe_event(event) for event in events]


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
