"""The command line: shared helpers, and the two documented commands.

The helpers are tested directly, because that is where the reusable logic lives.
The commands are then run the way the README documents them -- as scripts, from
the repository root -- so the acceptance criterion is checked as a user would
check it rather than by importing a ``main`` function.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from shed.agents import AgentSpec
from shed.cli import (
    ConsoleStyle,
    Visibility,
    build_lineup,
    card_text,
    describe_constraint,
    describe_event,
    describe_position,
    match_summary,
    narrate,
    positive_count,
    positive_seconds,
    replay_summary,
)
from shed.engine import (
    AtLeast,
    AtMost,
    Card,
    CardId,
    CardsDrawn,
    HandDealt,
    Phase,
    PlayConstraint,
    PlayerId,
    Rank,
    Unrestricted,
)
from shed.match import FinalPosition, MatchStatus, PlayerPosition
from shed.replay import decode_replay, match_deck, match_document, verify_replay
from tests.test_replay import DECK, roundtrip, sample_events, sync_match


def key(card: Card) -> int:
    """Sort key putting cards in canonical identifier order.

    Args:
        card: The card to place.

    Returns:
        Its identifier.
    """
    return int(card.id)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""The directory the documented commands are run from."""

QUICK_MATCH = ("--seconds-per-turn", "0.5", "--max-play-decisions", "8")
"""Options that keep a real timed match in the test suite short."""

OMNISCIENT = ConsoleStyle(visibility=Visibility.OMNISCIENT)
"""Everything the record holds, with cards spelled by rank alone."""

SUITED = ConsoleStyle(show_suits=True)
"""Public commentary, with every card carrying its suit."""


def run_script(name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run one of the project's scripts from the repository root.

    Args:
        name: Script file name under ``scripts/``.
        *arguments: Command-line arguments to pass it.

    Returns:
        The finished process, with its output captured as text.
    """
    # A fixed argument vector, no shell, and the project's own scripts.
    return subprocess.run(
        [sys.executable, f"scripts/{name}", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class TestLineups:
    """Turning a list of agent kinds into seated participants."""

    def test_repeated_kinds_get_distinct_labels(self) -> None:
        """Two greedy agents are still two distinguishable participants."""
        assert build_lineup(["greedy", "greedy", "random"]) == {
            PlayerId(0): AgentSpec(kind="greedy", name="greedy-0"),
            PlayerId(1): AgentSpec(kind="greedy", name="greedy-1"),
            PlayerId(2): AgentSpec(kind="random", name="random-2"),
        }

    @pytest.mark.parametrize(
        "kinds",
        [["random"], ["random"] * 6, []],
        ids=["one-seat", "six-seats", "no-seats"],
    )
    def test_unsupported_table_sizes_are_refused(self, kinds: list[str]) -> None:
        """The profile's range is checked before a runner is ever built."""
        with pytest.raises(ValueError, match="supports 2-5 players"):
            build_lineup(kinds)

    def test_an_unknown_kind_is_refused(self) -> None:
        """A mistyped lineup fails where it is written, not inside a worker."""
        with pytest.raises(ValueError, match="Unknown agent kind"):
            build_lineup(["random", "psychic"])


class TestArgumentTypes:
    """The numeric argument validators the parsers share."""

    @pytest.mark.parametrize(("text", "expected"), [("2", 2.0), ("0.25", 0.25)])
    def test_a_positive_duration_is_accepted(self, text: str, expected: float) -> None:
        """A usable budget parses to a float."""
        assert positive_seconds(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["0", "-1", "inf", "nan", "abc", ""],
        ids=["zero", "negative", "inf", "nan", "text", "empty"],
    )
    def test_an_unusable_duration_is_refused(self, text: str) -> None:
        """NaN and infinity are refused here, so the option is named in the error."""
        with pytest.raises(Exception, match="is not a"):
            positive_seconds(text)

    def test_a_positive_count_is_accepted(self) -> None:
        """A usable limit parses to an integer."""
        assert positive_count("10") == 10

    @pytest.mark.parametrize(
        "text", ["0", "-3", "1.5", "many"], ids=["zero", "negative", "float", "text"]
    )
    def test_an_unusable_count_is_refused(self, text: str) -> None:
        """A limit must be a positive whole number of decisions."""
        with pytest.raises(Exception, match="is not a"):
            positive_count(text)


class TestNarration:
    """Public commentary, written from a file that holds hidden information."""

    def test_every_event_type_gets_a_line(self) -> None:
        """Narration is total: no event is silently dropped."""
        events = sample_events()
        assert len(narrate(events)) == len(events)
        assert all(line for line in narrate(events))

    @pytest.mark.parametrize(
        "event",
        [
            HandDealt(player=PlayerId(0), count=3, cards=DECK[:3]),
            CardsDrawn(player=PlayerId(1), count=2, cards=DECK[3:5]),
        ],
        ids=["deal", "draw"],
    )
    def test_private_identities_never_reach_the_console(
        self, event: HandDealt | CardsDrawn
    ) -> None:
        """A dealt or drawn card is reported by count, exactly as opponents see it.

        The public line has to equal the line for the same event with its
        identities already stripped. Hunting for a rendered card as a substring
        would not do: a bare rank can be the count's own digit.
        """
        redacted = replace(event, cards=None)
        assert describe_event(event) == describe_event(redacted)
        assert describe_event(event, OMNISCIENT) != describe_event(redacted, OMNISCIENT)

    @pytest.mark.parametrize(
        "event",
        [
            HandDealt(player=PlayerId(0), count=3, cards=DECK[:3]),
            CardsDrawn(player=PlayerId(1), count=2, cards=DECK[3:5]),
        ],
        ids=["deal", "draw"],
    )
    def test_omniscient_narration_names_them(self, event: HandDealt | CardsDrawn) -> None:
        """The opt-in view prints exactly the identities the default withholds."""
        line = describe_event(event, OMNISCIENT)
        cards = event.cards
        assert cards is not None
        assert all(card_text(card) in line for card in cards)

    def test_a_filtered_event_stays_hidden_even_when_omniscient(self) -> None:
        """Identities a record does not hold cannot be printed from it."""
        filtered = HandDealt(player=PlayerId(0), count=3, cards=None)
        assert describe_event(filtered, OMNISCIENT) == "player 0 is dealt 3 cards"

    def test_public_narration_is_the_default(self) -> None:
        """Nothing leaks by omission: the careful setting is the one you get."""
        event = HandDealt(player=PlayerId(0), count=3, cards=DECK[:3])
        assert describe_event(event) == describe_event(event, ConsoleStyle())
        assert narrate([event]) == narrate([event], ConsoleStyle())

    def test_public_cards_are_named(self) -> None:
        """Cards everybody can see are shown, because that is the point of a summary.

        Asserted with suits on, where ``2c`` cannot be confused with a count.
        """
        played = next(line for line in narrate(sample_events(), SUITED) if "plays" in line)
        assert card_text(DECK[0], SUITED) in played


class TestCardSpelling:
    """How a card is written, which is independent of what may be shown."""

    @pytest.mark.parametrize(
        ("card", "bare", "suited"),
        [
            (DECK[0], "2", "2c"),
            (DECK[8], "10", "10c"),
            (DECK[50], "K", "Ks"),
            (DECK[53], "JK", "JK"),
        ],
        ids=["two", "ten", "king", "joker"],
    )
    def test_suits_are_appended_only_when_asked_for(
        self, card: Card, bare: str, suited: str
    ) -> None:
        """Rank alone by default; a joker has no suit to add either way."""
        assert card_text(card) == bare
        assert card_text(card, SUITED) == suited

    def test_the_spelling_is_independent_of_the_visibility(self) -> None:
        """The two settings compose: either can be on without the other."""
        event = HandDealt(player=PlayerId(0), count=2, cards=(DECK[8], DECK[50]))
        both = ConsoleStyle(visibility=Visibility.OMNISCIENT, show_suits=True)

        assert describe_event(event, OMNISCIENT) == "player 0 is dealt 2 cards: 10 K"
        assert describe_event(event, both) == "player 0 is dealt 2 cards: 10c Ks"
        assert describe_event(event, SUITED) == "player 0 is dealt 2 cards"

    def test_a_position_follows_the_same_spelling(self) -> None:
        """One style reaches every renderer, the position dump included."""
        result = sync_match(max_play_decisions=6)
        deck = match_deck(result.metadata)
        bare = describe_position(result.final_position, deck, OMNISCIENT)
        suited = describe_position(result.final_position, deck, ConsoleStyle(show_suits=True))

        assert "face down 0=" in bare[4]
        assert len(suited[4]) > len(bare[4])


class TestPositions:
    """The omniscient position dump, which resolves identifiers to cards."""

    @pytest.mark.parametrize(
        ("constraint", "expected"),
        [
            (Unrestricted(), "unrestricted"),
            (AtLeast(Rank.NINE), "at least 9"),
            (AtMost(Rank.SEVEN), "at most 7"),
            (AtLeast(Rank.TEN), "at least 10"),
        ],
        ids=["unrestricted", "at-least", "at-most", "ten"],
    )
    def test_constraints_read_as_english(self, constraint: PlayConstraint, expected: str) -> None:
        """The pile's restriction is stated, not spelled as a dataclass."""
        assert describe_constraint(constraint) == expected

    def test_every_zone_of_the_position_is_reported(self) -> None:
        """Hands, face-up cards, face-down slots, and all three piles appear."""
        result = sync_match(max_play_decisions=6)
        lines = describe_position(result.final_position, match_deck(result.metadata))

        assert lines[0].startswith("position: play | ply 6 |")
        assert "draw pile (" in lines[1]
        assert "discard (" in lines[2]
        assert "burned (" in lines[3]
        assert len(lines) == 4 + result.metadata.player_count

    def test_a_players_line_names_the_cards_behind_its_face_down_slots(self) -> None:
        """The slot identities are the whole point of the omniscient view."""
        result = sync_match(max_play_decisions=6)
        deck = match_deck(result.metadata)
        first = result.final_position.players[0]
        line = describe_position(result.final_position, deck)[4]

        cards = {int(card.id): card for card in deck}
        assert line.strip().startswith("player 0: hand ")
        for slot, card_id in first.face_down:
            assert f"{slot}={card_text(cards[card_id])}" in line

    def test_an_empty_zone_is_marked_rather_than_blank(self) -> None:
        """A finished winner has no cards left, and the line still reads."""
        result = sync_match()
        lines = describe_position(result.final_position, match_deck(result.metadata))
        assert result.outcome is not None
        assert "hand - | face up - | face down -" in lines[4 + result.outcome.winner]


class TestOrdering:
    """What the console sorts, and what it must leave exactly as it is."""

    def test_a_group_of_cards_is_rendered_in_identifier_order(self) -> None:
        """A dealt hand reads the same however the event happened to list it."""
        cards = (DECK[50], DECK[8], DECK[2])
        shuffled = HandDealt(player=PlayerId(0), count=3, cards=cards)
        ordered = HandDealt(player=PlayerId(0), count=3, cards=tuple(sorted(cards, key=key)))

        assert describe_event(shuffled, OMNISCIENT) == "player 0 is dealt 3 cards: 4 10 K"
        assert describe_event(shuffled, OMNISCIENT) == describe_event(ordered, OMNISCIENT)

    def test_ordering_is_by_rank_not_by_the_suit_major_identifier(self) -> None:
        """Identifiers run suit by suit, so identifier order would not look sorted."""
        hand = (DECK[12], DECK[20], DECK[52])  # Ace of clubs, nine of diamonds, joker.
        event = HandDealt(player=PlayerId(0), count=3, cards=hand)

        assert describe_event(event, OMNISCIENT) == "player 0 is dealt 3 cards: 9 A JK"
        assert [card.id for card in sorted(hand, key=key)] == [12, 20, 52]

    def test_showing_suits_keeps_a_rank_together(self) -> None:
        """The identifier tie-break groups one rank by suit rather than scattering it."""
        both = ConsoleStyle(visibility=Visibility.OMNISCIENT, show_suits=True)
        hand = (DECK[35], DECK[9], DECK[22], DECK[49])  # Jh, Jc, Jd, Qs.
        event = HandDealt(player=PlayerId(0), count=4, cards=hand)

        assert describe_event(event, both) == "player 0 is dealt 4 cards: Jc Jd Jh Qs"

    def test_a_hand_is_sorted_but_the_piles_keep_their_order(self) -> None:
        """The two piles are ordered structures; a hand is a bag of cards."""
        position = FinalPosition(
            phase=Phase.PLAY,
            current_player=PlayerId(0),
            current_ply=4,
            constraint=Unrestricted(),
            draw_pile=(CardId(5), CardId(1), CardId(3)),
            discard_pile=(CardId(4), CardId(0)),
            burned_cards=(CardId(8), CardId(6)),
            players=(
                PlayerPosition(
                    player=PlayerId(0),
                    hand=(CardId(9), CardId(2), CardId(7)),
                    face_up=(),
                    face_down=(),
                ),
            ),
        )
        lines = describe_position(position, DECK)

        assert lines[1].endswith("7 3 5")  # draw order: identifiers 5, 1, 3.
        assert lines[2].endswith("6 2")  # play order: identifiers 4, 0.
        assert lines[3].endswith("8 10")  # burned is a bag, so sorted by rank.
        assert lines[4].strip() == "player 0: hand 4 9 J | face up - | face down -"

    def test_sorting_is_presentation_only(self) -> None:
        """The engine's orders are untouched, so a replay still compares them."""
        result = sync_match(max_play_decisions=6)
        describe_position(result.final_position, match_deck(result.metadata))
        assert verify_replay(decode_replay(roundtrip(match_document(result)))).ok


class TestSummaries:
    """The end-of-match block both commands print."""

    def test_a_finished_match_names_its_winner(self) -> None:
        """The status line reports the seat and the participant that won."""
        result = sync_match()
        lines = match_summary(result)
        assert result.outcome is not None
        winner = result.metadata.agents[result.outcome.winner]
        assert any(f"player {result.outcome.winner} ({winner.name}) wins" in line for line in lines)
        assert any("deal seed 42" in line for line in lines)

    def test_a_truncated_match_is_never_reported_as_won(self) -> None:
        """Truncation prints its reason instead of inventing a winner."""
        lines = match_summary(sync_match(max_play_decisions=4))
        assert any("truncated" in line for line in lines)
        assert not any("wins" in line for line in lines)

    def test_a_replay_summary_adds_the_documents_provenance(self) -> None:
        """Reading a file also reports what wrote it."""
        replay = decode_replay(roundtrip(match_document(sync_match(max_play_decisions=4))))
        lines = replay_summary(replay)
        assert lines[0].startswith("replay schema 1 |")
        assert any("shed-v1 | 2 players" in line for line in lines)


class TestPlayCommand:
    """``scripts/play.py``, run as documented."""

    def test_the_documented_command_plays_and_writes_a_replay(self, tmp_path: Path) -> None:
        """A match runs, prints its public actions, and saves a decodable file."""
        output = tmp_path / "results" / "match.json"
        finished = run_script(
            "play.py",
            "--agents",
            "random",
            "greedy",
            "--seed",
            "42",
            *QUICK_MATCH,
            "--output",
            str(output),
        )

        assert finished.returncode == 0, finished.stderr
        assert "deals to 2 seats" in finished.stdout
        assert "shed-v1 | 2 players | deal seed 42" in finished.stdout
        assert str(output) in finished.stdout
        assert json.loads(output.read_text(encoding="utf-8"))["schema"] == 1

    def test_quiet_prints_the_summary_without_the_actions(self) -> None:
        """The action list is the only thing ``--quiet`` removes."""
        quiet = run_script("play.py", "--agents", "random", "random", *QUICK_MATCH, "--quiet")
        loud = run_script("play.py", "--agents", "random", "random", *QUICK_MATCH)

        assert quiet.returncode == 0, quiet.stderr
        assert "deals to 2 seats" not in quiet.stdout
        assert "deals to 2 seats" in loud.stdout
        assert len(quiet.stdout.splitlines()) < len(loud.stdout.splitlines())

    def test_omniscient_shows_what_the_public_log_withholds(self) -> None:
        """The same match, printed twice: the flag adds identities and a position."""
        public = run_script("play.py", "--agents", "random", "greedy", "--seed", "42", *QUICK_MATCH)
        omniscient = run_script(
            "play.py", "--agents", "random", "greedy", "--seed", "42", *QUICK_MATCH, "--omniscient"
        )

        assert omniscient.returncode == 0, omniscient.stderr
        assert "is dealt 3 cards\n" in public.stdout
        assert "is dealt 3 cards: " in omniscient.stdout
        assert "face down 0=" in omniscient.stdout
        assert "face down 0=" not in public.stdout
        assert "position: " not in public.stdout

    def test_omniscient_and_quiet_compose(self) -> None:
        """Together they print the position and the summary, and no action log."""
        finished = run_script(
            "play.py", "--agents", "random", "random", *QUICK_MATCH, "--omniscient", "--quiet"
        )

        assert finished.returncode == 0, finished.stderr
        assert "deals to 2 seats" not in finished.stdout
        assert finished.stdout.startswith("position: ")
        assert "status: " in finished.stdout

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (("--agents", "random"), "supports 2-5 players"),
            (("--agents", "random", "greedy", "--dealer", "5"), "is not a seat"),
        ],
        ids=["one-seat", "impossible-dealer"],
    )
    def test_arguments_describing_no_runnable_match_exit_two(
        self, arguments: tuple[str, ...], message: str
    ) -> None:
        """A lineup or dealer the profile refuses stops before any match starts."""
        finished = run_script("play.py", *arguments, *QUICK_MATCH)
        assert finished.returncode == 2
        assert message in finished.stderr

    @pytest.mark.parametrize(
        "arguments",
        [
            ("--agents", "psychic"),
            ("--seed", "3"),
            ("--agents", "random", "--seconds-per-turn", "0"),
        ],
        ids=["unknown-kind", "no-lineup", "impossible-budget"],
    )
    def test_argparse_refuses_malformed_options(self, arguments: tuple[str, ...]) -> None:
        """Unknown kinds, a missing lineup, and an unusable budget never run."""
        assert run_script("play.py", *arguments).returncode == 2


class TestReplayCommand:
    """``scripts/replay.py``, run as documented."""

    @pytest.fixture
    def saved(self, tmp_path: Path) -> Path:
        """Write a valid replay of a short synchronous match.

        Args:
            tmp_path: Directory for the file.

        Returns:
            Path to the written replay.
        """
        path = tmp_path / "match.json"
        document = match_document(sync_match(max_play_decisions=6))
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_verifying_a_saved_match_succeeds(self, saved: Path) -> None:
        """The documented verify command reports success and exits zero."""
        finished = run_script("replay.py", str(saved), "--verify")

        assert finished.returncode == 0, finished.stderr
        assert "replay schema 1" in finished.stdout
        assert "recorded decisions reproduce this replay exactly" in finished.stdout

    def test_summarizing_without_verifying_prints_the_match(self, saved: Path) -> None:
        """Without ``--verify`` the command only reads and reports."""
        finished = run_script("replay.py", str(saved))

        assert finished.returncode == 0, finished.stderr
        assert "truncated" in finished.stdout
        assert "reproduce" not in finished.stdout

    def test_events_adds_the_public_actions(self, saved: Path) -> None:
        """``--events`` narrates the recording, still without hidden identities."""
        finished = run_script("replay.py", str(saved), "--events")

        assert finished.returncode == 0, finished.stderr
        assert "is dealt 3 cards" in finished.stdout
        assert "deals to 2 seats" in finished.stdout

    def test_omniscient_reveals_the_recording_and_its_position(self, saved: Path) -> None:
        """The flag implies the action log, unredacted, plus the final position."""
        finished = run_script("replay.py", str(saved), "--omniscient")

        assert finished.returncode == 0, finished.stderr
        assert "is dealt 3 cards: " in finished.stdout
        assert "face down 0=" in finished.stdout
        assert "draw pile (" in finished.stdout

    def test_a_corrupted_recording_fails_verification(self, saved: Path, tmp_path: Path) -> None:
        """A tampered decision exits nonzero and says which one disagreed."""
        document = json.loads(saved.read_text(encoding="utf-8"))
        document["result"]["decisions"][3]["events"].clear()
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps(document), encoding="utf-8")

        finished = run_script("replay.py", str(broken), "--verify")
        assert finished.returncode == 1
        assert "verification failed" in finished.stderr
        assert "different events" in finished.stderr

    def test_an_unsupported_schema_exits_two(self, saved: Path, tmp_path: Path) -> None:
        """A future replay version is refused clearly rather than half-read."""
        document = json.loads(saved.read_text(encoding="utf-8"))
        document["schema"] = 99
        future = tmp_path / "future.json"
        future.write_text(json.dumps(document), encoding="utf-8")

        finished = run_script("replay.py", str(future), "--verify")
        assert finished.returncode == 2
        assert "replay schema 99" in finished.stderr

    def test_a_missing_file_exits_two(self, tmp_path: Path) -> None:
        """A path that is not there is an input error, not a verification failure."""
        finished = run_script("replay.py", str(tmp_path / "absent.json"), "--verify")
        assert finished.returncode == 2
        assert "cannot read" in finished.stderr


def test_the_match_status_of_a_short_run_is_reported(tmp_path: Path) -> None:
    """A truncated run still exits zero: the bound is a limit, not a failure."""
    output = tmp_path / "match.json"
    finished = run_script(
        "play.py", "--agents", "greedy", "greedy", "--quiet", *QUICK_MATCH, "--output", str(output)
    )

    assert finished.returncode == 0, finished.stderr
    assert MatchStatus.TRUNCATED.value in finished.stdout
