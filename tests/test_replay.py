"""Replay serialization, decoder rejection, and verification of recorded matches.

The codec tests are the acceptance criterion in section 15: every move and every
event type has to survive a round trip, and the decoder has to refuse raw input
that only looks well formed -- notably a JSON boolean where an integer belongs,
because Python would otherwise play ``True`` as a one.

Most tests here build a match with :class:`SyncRunner`, which decides
synchronously instead of starting workers. That keeps the replay tests about
replay. One test at the end does run a real timed match, because "a saved timed
match replays without agents or timing" is not something a synchronous stand-in
can demonstrate.
"""

from __future__ import annotations

import json
import multiprocessing
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

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
    HandDealt,
    Move,
    ObservedEvent,
    Outcome,
    PickUp,
    PileBurned,
    PilePickedUp,
    Play,
    PlayConstraint,
    PlayerId,
    PlayerView,
    PublicPlayerState,
    Rank,
    Reveal,
    SlotId,
    Suit,
    Unrestricted,
    Zone,
    build_deck,
)
from shed.match import (
    CloseReason,
    MatchConfig,
    MatchResult,
    MatchRunner,
    MatchStatus,
    TurnRecord,
)
from shed.replay import (
    Replay,
    ReplayFormatError,
    decode_card,
    decode_constraint,
    decode_event,
    decode_move,
    decode_replay,
    detect_source_revision,
    encode_card,
    encode_constraint,
    encode_event,
    encode_move,
    match_document,
    read_replay,
    verify_replay,
    write_match,
)

DECK = build_deck()
"""The canonical deck, for building events out of real cards."""

BUDGET = 0.5
"""Acceptance budget for the one test that starts real workers."""


class SyncRunner(MatchRunner):
    """A runner that decides synchronously, so replay tests start no processes.

    Only :meth:`choose_move` is replaced. Everything the replay tests care about
    -- the deal, the applied-decision stream, the events, the metadata, the
    final position, strict-mode aborts, truncation -- is produced by the real
    :meth:`shed.match.MatchRunner.run`.

    Attributes:
        _moves: Generator picking among the offered legal moves.
        _next_decision: Sequential decision identifier.
        _failing: Decisions to report as having accepted no candidate, which is
            what strict mode aborts on.
    """

    def __init__(
        self,
        agents: dict[PlayerId, AgentSpec],
        config: MatchConfig,
        *,
        seed: int = 0,
        failing: Sequence[int] = (),
    ) -> None:
        """Prepare a synchronous runner.

        Args:
            agents: Participant per seat.
            config: Timing, limits, and failure policy.
            seed: Seed for the move-selection generator.
            failing: Decision identifiers to mark as agent failures.
        """
        super().__init__(agents, config)
        self._moves = random.Random(seed)
        self._next_decision = 0
        self._failing = frozenset(failing)

    def choose_move(self, view: PlayerView) -> TurnRecord:
        """Select a move without a worker, clock, or agent.

        Args:
            view: The actor's observation.

        Returns:
            A record with plausible diagnostics, so the encoded replay carries
            the same shape a timed match would produce.
        """
        decision_id = self._next_decision
        self._next_decision += 1
        failed = decision_id in self._failing
        return TurnRecord(
            decision_id=decision_id,
            player=view.viewer,
            phase=view.phase,
            move=self._moves.choice(view.legal_moves),
            reason=CloseReason.DEADLINE if failed else CloseReason.FINAL,
            used_fallback=failed,
            accepted=0 if failed else 1,
            rejected=0,
            failure=None,
            budget_seconds=self._config.seconds_per_turn,
            selection_seconds=0.01,
            cleanup_seconds=0.001,
            agent_seed=1000 + decision_id,
        )


def lineup() -> dict[PlayerId, AgentSpec]:
    """Build the two-seat lineup every synchronous match in this module uses.

    Returns:
        A random agent in seat zero and a greedy agent in seat one.
    """
    return {
        PlayerId(0): AgentSpec(kind="random", name="random-0"),
        PlayerId(1): AgentSpec(kind="greedy", name="greedy-1"),
    }


def sync_match(
    *,
    seed: int = 3,
    deal_seed: int = 42,
    max_play_decisions: int = 10_000,
    strict_failures: bool = False,
    failing: Sequence[int] = (),
) -> MatchResult:
    """Play one match synchronously and return its result.

    Args:
        seed: Seed for the move-selection generator.
        deal_seed: Deck seed.
        max_play_decisions: Bound before the match truncates.
        strict_failures: Whether an agent failure aborts the match.
        failing: Decisions to mark as agent failures.

    Returns:
        The match result, whatever ended it.
    """
    config = MatchConfig(
        seconds_per_turn=1.0,
        max_play_decisions=max_play_decisions,
        strict_failures=strict_failures,
    )
    runner = SyncRunner(lineup(), config, seed=seed, failing=failing)
    return runner.run(deal_seed=deal_seed)


def roundtrip(document: object) -> Any:
    """Send a document through real JSON text, as a file would.

    Encoding to text and back is what turns Python objects into the primitives a
    decoder actually meets: tuples become arrays, and nothing but JSON's own
    types survives.

    Args:
        document: An encoded document or fragment.

    Returns:
        The same document after a JSON text round trip.
    """
    return json.loads(json.dumps(document))


def tamper(document: dict[str, Any], mutate: Callable[[dict[str, Any]], None]) -> Replay:
    """Corrupt a valid document and decode it again.

    Args:
        document: A freshly built document, mutated in place.
        mutate: What to change in it.

    Returns:
        The decoded, corrupted replay.
    """
    mutate(document)
    return decode_replay(roundtrip(document))


MOVES: tuple[Move, ...] = (
    Arrange((CardId(4), CardId(1), CardId(9))),
    Play(Zone.HAND, Rank.SEVEN, 2),
    Play(Zone.FACE_UP, Rank.JOKER, 1),
    Play(Zone.HAND, Rank.ACE, 4),
    Reveal(SlotId(0)),
    Reveal(SlotId(2)),
    PickUp(),
)
"""One instance of every move shape the engine can generate."""

CONSTRAINTS: tuple[PlayConstraint, ...] = (
    Unrestricted(),
    AtLeast(Rank.TWO),
    AtLeast(Rank.ACE),
    AtMost(Rank.SEVEN),
)
"""One instance of every constraint shape the profile can produce."""


def sample_events() -> tuple[ObservedEvent, ...]:
    """Build one instance of every event type, including filtered copies.

    Returns:
        Events covering the full union: the public opening, the private deal and
        draw in both their full and filtered shapes, a commitment, a batch, a
        successful and a failed reveal, a pickup, a burn, and the ending.
    """
    cards = (DECK[0], DECK[1], DECK[2])
    public = PublicPlayerState(
        player=PlayerId(0),
        hand_count=3,
        face_up=cards,
        face_down_slots=(SlotId(0), SlotId(1), SlotId(2)),
    )
    return (
        GameStarted(dealer=PlayerId(0), seat_order=(PlayerId(0), PlayerId(1)), players=(public,)),
        HandDealt(player=PlayerId(1), count=3, cards=cards),
        HandDealt(player=PlayerId(1), count=3, cards=None),
        ArrangementCommitted(player=PlayerId(0), face_up=cards),
        CardsPlayed(player=PlayerId(0), source=Zone.HAND, cards=cards[:2]),
        CardsPlayed(player=PlayerId(1), source=Zone.FACE_UP, cards=(DECK[53],)),
        CardRevealed(player=PlayerId(1), slot=SlotId(2), card=DECK[53], playable=True),
        CardRevealed(player=PlayerId(1), slot=SlotId(0), card=DECK[7], playable=False),
        CardsDrawn(player=PlayerId(0), count=2, cards=cards[:2]),
        CardsDrawn(player=PlayerId(0), count=2, cards=None),
        PilePickedUp(player=PlayerId(0), cards=cards),
        PileBurned(player=PlayerId(1), cards=cards, reason=BurnReason.TEN),
        PileBurned(player=PlayerId(1), cards=cards, reason=BurnReason.FOUR_OF_A_KIND),
        GameEnded(outcome=Outcome(winner=PlayerId(1))),
    )


class TestCardCodec:
    """Physical cards, the vocabulary every other encoder is built from."""

    @pytest.mark.parametrize("card", DECK, ids=lambda card: str(card.id))
    def test_every_canonical_card_round_trips(self, card: Card) -> None:
        """All 54 cards, jokers included, decode back to themselves."""
        assert decode_card(roundtrip(encode_card(card))) == card

    def test_a_card_encodes_its_rank_as_an_integer_and_its_suit_as_a_name(self) -> None:
        """The shape is the documented one, not whatever a dataclass happens to be."""
        assert encode_card(Card(id=CardId(3), rank=Rank.FIVE, suit=Suit.CLUBS)) == {
            "id": 3,
            "rank": 5,
            "suit": "clubs",
        }
        assert encode_card(DECK[52])["suit"] is None

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ({"id": 0, "rank": 2}, "suit is missing"),
            ({"id": True, "rank": 2, "suit": "clubs"}, "must be an integer"),
            ({"id": 0, "rank": "2", "suit": "clubs"}, "must be an integer"),
            ({"id": 0, "rank": 99, "suit": "clubs"}, "must be one of"),
            ({"id": 0, "rank": 2, "suit": "wands"}, "must be one of"),
            ({"id": 0, "rank": 15, "suit": "clubs"}, "is invalid"),
            ({"id": -1, "rank": 2, "suit": "clubs"}, "is invalid"),
            ([0, 2, "clubs"], "must be an object"),
        ],
        ids=[
            "missing-suit",
            "boolean-id",
            "string-rank",
            "unknown-rank",
            "unknown-suit",
            "suited-joker",
            "negative-id",
            "not-an-object",
        ],
    )
    def test_malformed_cards_are_refused(self, raw: object, message: str) -> None:
        """A card is decoded, not coerced: every bad shape is named and refused."""
        with pytest.raises(ReplayFormatError, match=message):
            decode_card(raw)


class TestMoveCodec:
    """Every decision shape, in both directions."""

    @pytest.mark.parametrize("move", MOVES, ids=lambda move: type(move).__name__)
    def test_every_move_type_round_trips(self, move: Move) -> None:
        """Encoding and decoding a move returns an equal typed move."""
        decoded = decode_move(roundtrip(encode_move(move)))
        assert decoded == move
        assert type(decoded) is type(move)

    def test_a_play_encodes_in_the_documented_shape(self) -> None:
        """The example from the design is the shape that is actually written."""
        assert encode_move(Play(Zone.HAND, Rank.SEVEN, 2)) == {
            "type": "play",
            "source": "hand",
            "rank": 7,
            "count": 2,
        }

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ({"type": "play", "source": "hand", "rank": 7, "count": True}, "must be an integer"),
            ({"type": "play", "source": "hand", "rank": True, "count": 1}, "must be an integer"),
            ({"type": "play", "source": "hand", "rank": 7, "count": 1.0}, "must be an integer"),
            ({"type": "play", "source": "hand", "rank": 7}, "count is missing"),
            ({"type": "play", "source": "table", "rank": 7, "count": 1}, "must be one of"),
            ({"type": "play", "source": "face_down", "rank": 7, "count": 1}, "is invalid"),
            ({"type": "play", "source": "hand", "rank": 7, "count": 0}, "is invalid"),
            ({"type": "reveal", "slot": True}, "must be an integer"),
            ({"type": "reveal", "slot": -1}, "is invalid"),
            ({"type": "arrange", "face_up_cards": [1, 2]}, "exactly three cards"),
            ({"type": "arrange", "face_up_cards": [1, 2, 2]}, "is invalid"),
            ({"type": "arrange", "face_up_cards": [1, 2, True]}, "must be an integer"),
            ({"type": "arrange", "face_up_cards": 3}, "must be an array"),
            ({"type": "shuffle"}, "not a known move"),
            ({"source": "hand", "rank": 7, "count": 1}, "type is missing"),
            ({"type": 4}, "must be a string"),
            ("play", "must be an object"),
        ],
        ids=[
            "boolean-count",
            "boolean-rank",
            "float-count",
            "missing-count",
            "unknown-zone",
            "unplayable-zone",
            "zero-count",
            "boolean-slot",
            "negative-slot",
            "short-arrangement",
            "duplicate-arrangement",
            "boolean-card-id",
            "scalar-arrangement",
            "unknown-tag",
            "missing-tag",
            "non-string-tag",
            "not-an-object",
        ],
    )
    def test_malformed_moves_are_refused(self, raw: object, message: str) -> None:
        """Shape, tag, and primitive type are all checked before construction."""
        with pytest.raises(ReplayFormatError, match=message):
            decode_move(raw)


class TestConstraintCodec:
    """Play constraints, tagged rather than inferred from their fields."""

    @pytest.mark.parametrize("constraint", CONSTRAINTS, ids=lambda item: repr(item))
    def test_every_constraint_type_round_trips(self, constraint: PlayConstraint) -> None:
        """Encoding and decoding a constraint returns an equal typed constraint."""
        decoded = decode_constraint(roundtrip(encode_constraint(constraint)))
        assert decoded == constraint
        assert type(decoded) is type(constraint)

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ({"type": "at_least"}, "rank is missing"),
            ({"type": "at_least", "rank": True}, "must be an integer"),
            ({"type": "at_most", "rank": 15}, "is invalid"),
            ({"type": "at_least", "rank": 1}, "must be one of"),
            ({"type": "exactly", "rank": 7}, "not a known constraint"),
        ],
        ids=["missing-rank", "boolean-rank", "joker-bound", "unknown-rank", "unknown-tag"],
    )
    def test_malformed_constraints_are_refused(self, raw: object, message: str) -> None:
        """A joker is never a bound, and a boolean is never a rank."""
        with pytest.raises(ReplayFormatError, match=message):
            decode_constraint(raw)


class TestEventCodec:
    """Every event type, in both its full and its filtered shape."""

    @pytest.mark.parametrize("event", sample_events(), ids=lambda event: type(event).__name__)
    def test_every_event_type_round_trips(self, event: ObservedEvent) -> None:
        """Encoding and decoding an event returns an equal typed event."""
        decoded = decode_event(roundtrip(encode_event(event)))
        assert decoded == event
        assert type(decoded) is type(event)

    def test_a_filtered_private_event_keeps_its_missing_identities(self) -> None:
        """A ``cards=None`` event is a distinct, faithful shape, not an error."""
        filtered = HandDealt(player=PlayerId(1), count=3, cards=None)
        assert encode_event(filtered)["cards"] is None
        assert decode_event(roundtrip(encode_event(filtered))) == filtered

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            ({"type": "cards_drawn", "player": 0, "count": True, "cards": None}, "an integer"),
            ({"type": "cards_drawn", "player": True, "count": 1, "cards": None}, "an integer"),
            ({"type": "cards_drawn", "player": 0, "count": 1}, "cards is missing"),
            (
                {"type": "card_revealed", "player": 0, "slot": 0, "card": None, "playable": True},
                "must be an object",
            ),
            (
                {
                    "type": "card_revealed",
                    "player": 0,
                    "slot": 0,
                    "card": {"id": 0, "rank": 2, "suit": "clubs"},
                    "playable": 1,
                },
                "must be a boolean",
            ),
            (
                {"type": "pile_burned", "player": 0, "cards": [], "reason": "nine"},
                "must be one of",
            ),
            ({"type": "cards_played", "player": 0, "source": "hand"}, "cards is missing"),
            ({"type": "game_ended", "outcome": {"winner": True}}, "must be an integer"),
            ({"type": "card_flipped"}, "not a known event"),
        ],
        ids=[
            "boolean-count",
            "boolean-player",
            "missing-cards",
            "null-card",
            "integer-flag",
            "unknown-burn-reason",
            "missing-played-cards",
            "boolean-winner",
            "unknown-tag",
        ],
    )
    def test_malformed_events_are_refused(self, raw: object, message: str) -> None:
        """Every field of every event is type-checked at the boundary."""
        with pytest.raises(ReplayFormatError, match=message):
            decode_event(raw)


class TestDocument:
    """The whole replay document, written and read back."""

    def test_a_finished_match_round_trips_completely(self) -> None:
        """Decoding a written match reproduces every record the runner made."""
        result = sync_match()
        assert result.status is MatchStatus.FINISHED
        replay = decode_replay(roundtrip(match_document(result)))

        assert replay.metadata == result.metadata
        assert replay.status is result.status
        assert replay.outcome == result.outcome
        assert replay.failure == result.failure
        assert replay.play_decisions == result.play_decisions
        assert replay.initial_events == result.initial_events
        assert replay.decisions == result.decisions
        assert replay.final_position == result.final_position
        assert replay.turns == result.turns

    def test_the_recorded_deck_is_the_deal_the_match_was_played_from(self) -> None:
        """The canonical order is stored, so replay never re-runs the shuffle."""
        result = sync_match()
        replay = decode_replay(roundtrip(match_document(result)))
        assert sorted(replay.deck, key=lambda card: card.id) == list(DECK)
        assert len(replay.deck) == len(DECK)

    def test_the_header_records_the_profile_schema_and_versions(self) -> None:
        """Provenance is part of the artifact, not something a reader infers."""
        document = match_document(sync_match(), source_revision="deadbeef")
        assert document["schema"] == 1
        assert document["rules"] == {
            "id": "shed-v1",
            "min_players": 2,
            "max_players": 5,
            "joker_count": 2,
            "initial_hand_size": 3,
            "initial_face_up_count": 3,
            "initial_face_down_count": 3,
            "refill_target": 3,
        }
        header = decode_replay(roundtrip(document)).header
        assert (header.schema, header.source_revision) == (1, "deadbeef")
        assert header.package_version and header.python_version

    def test_selected_but_unapplied_moves_stay_out_of_the_applied_stream(self) -> None:
        """A strict abort records its selection without replaying it."""
        result = sync_match(strict_failures=True, failing=(4,))
        assert result.status is MatchStatus.AGENT_FAILED
        replay = decode_replay(roundtrip(match_document(result)))

        assert len(replay.unapplied_turns) == 1
        assert replay.unapplied_turns[0].decision_id == 4
        assert all(decision.turn.decision_id != 4 for decision in replay.decisions)
        assert [turn.decision_id for turn in replay.turns] == list(range(5))

    def test_a_written_file_is_utf8_json_that_reads_back(self, tmp_path: Path) -> None:
        """Writing creates missing directories and produces a decodable file."""
        result = sync_match(max_play_decisions=6)
        destination = tmp_path / "nested" / "results" / "match.json"
        written = write_match(result, destination)

        assert written == destination
        assert json.loads(destination.read_text(encoding="utf-8"))["schema"] == 1
        assert read_replay(destination).final_position == result.final_position

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda doc: doc.__setitem__("schema", 2), "replay schema 2"),
            (lambda doc: doc.__setitem__("schema", "1"), "must be an integer"),
            (lambda doc: doc.pop("schema"), "schema is missing"),
            (lambda doc: doc["rules"].__setitem__("id", "shed-v2"), "rules profile 'shed-v2'"),
            (lambda doc: doc["rules"].__setitem__("joker_count", 4), "is invalid"),
            (lambda doc: doc["rules"].__setitem__("refill_target", True), "must be an integer"),
            (lambda doc: doc.pop("result"), "result is missing"),
            (lambda doc: doc["setup"]["agents"][0].__setitem__("kind", "psychic"), "is invalid"),
            (
                lambda doc: doc["setup"]["config"].__setitem__("seconds_per_turn", 0),
                "is invalid",
            ),
            (
                lambda doc: doc["setup"]["config"].__setitem__("strict_failures", 0),
                "must be a boolean",
            ),
            (lambda doc: doc["result"].__setitem__("status", "abandoned"), "must be one of"),
            (
                lambda doc: doc["result"]["decisions"][0]["turn"].__setitem__("accepted", True),
                "must be an integer",
            ),
            (
                lambda doc: doc["result"]["final_position"].__setitem__("current_ply", "12"),
                "must be an integer",
            ),
            (
                lambda doc: doc["result"]["final_position"]["players"][0].__setitem__(
                    "face_down", [[0]]
                ),
                "\\[slot, card\\] pair",
            ),
        ],
        ids=[
            "unsupported-schema",
            "string-schema",
            "missing-schema",
            "unsupported-profile",
            "modified-profile",
            "boolean-profile-field",
            "missing-result",
            "unknown-agent-kind",
            "impossible-budget",
            "integer-flag",
            "unknown-status",
            "boolean-counter",
            "string-ply",
            "short-slot-pair",
        ],
    )
    def test_unsupported_or_malformed_documents_are_refused(
        self, mutate: Callable[[dict[str, Any]], None], message: str
    ) -> None:
        """Every external claim is checked before anything is reconstructed."""
        document = roundtrip(match_document(sync_match(max_play_decisions=4)))
        mutate(document)
        with pytest.raises(ReplayFormatError, match=message):
            decode_replay(document)

    def test_a_document_that_is_not_an_object_is_refused(self) -> None:
        """A JSON array is not a replay, and says so rather than raising later."""
        with pytest.raises(ReplayFormatError, match="must be an object"):
            decode_replay([1, 2, 3])

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("{not json}", "not valid JSON"),
            ('{"schema": NaN}', "not valid JSON in a replay"),
            ('{"schema": 1, "package_version": Infinity}', "not valid JSON in a replay"),
        ],
        ids=["malformed", "nan", "infinity"],
    )
    def test_unreadable_files_are_refused(self, tmp_path: Path, text: str, message: str) -> None:
        """A replay is ordinary JSON: the parser's extensions are not accepted."""
        path = tmp_path / "broken.json"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ReplayFormatError, match=message):
            read_replay(path)


class TestVerification:
    """Replaying recorded decisions through the engine and comparing."""

    def test_a_finished_match_replays_to_the_same_events_and_outcome(self) -> None:
        """The recorded stream reproduces exactly, decision for decision."""
        result = sync_match()
        replay = decode_replay(roundtrip(match_document(result)))
        check = verify_replay(replay)

        assert check.ok
        assert check.problems == ()
        assert check.applied == len(replay.decisions)
        assert check.outcome == result.outcome

    def test_a_truncated_match_replays_to_the_position_it_stopped_in(self) -> None:
        """Truncation has no winner, and verification does not invent one."""
        result = sync_match(max_play_decisions=5)
        assert result.status is MatchStatus.TRUNCATED
        check = verify_replay(decode_replay(roundtrip(match_document(result))))

        assert check.ok
        assert check.outcome is None
        assert check.applied == len(result.decisions)

    def test_an_aborted_match_replays_without_its_unapplied_selection(self) -> None:
        """Strict mode's last selection was never played, so it is never replayed."""
        result = sync_match(strict_failures=True, failing=(6,))
        assert result.status is MatchStatus.AGENT_FAILED
        replay = decode_replay(roundtrip(match_document(result)))
        check = verify_replay(replay)

        assert check.ok
        assert check.applied == 6
        assert check.outcome is None

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (
                lambda doc: doc["result"]["decisions"][0]["turn"].__setitem__(
                    "move", {"type": "reveal", "slot": 0}
                ),
                "not playable",
            ),
            (
                lambda doc: doc["result"]["decisions"][2]["turn"].__setitem__("player", 9),
                "recorded for player 9",
            ),
            (
                lambda doc: doc["result"]["decisions"][2]["turn"].__setitem__("phase", "setup"),
                "recorded in phase setup",
            ),
            (
                lambda doc: doc["result"]["decisions"][2]["events"].clear(),
                "different events",
            ),
            (
                lambda doc: doc["result"]["initial_events"].pop(),
                "deal events differ",
            ),
            (
                lambda doc: doc["setup"]["deck"].pop(),
                "cannot be rebuilt",
            ),
            (
                lambda doc: doc["result"]["final_position"].__setitem__("current_ply", 99),
                "final position differs",
            ),
            (
                lambda doc: doc["result"].__setitem__("play_decisions", 99),
                "99 were recorded",
            ),
            (
                lambda doc: doc["result"].__setitem__("status", "truncated"),
                "recorded status is truncated",
            ),
            (
                lambda doc: doc["result"].__setitem__("outcome", {"winner": 0}),
                "not the recorded",
            ),
        ],
        ids=[
            "illegal-move",
            "wrong-player",
            "wrong-phase",
            "wrong-events",
            "wrong-deal",
            "incomplete-deck",
            "wrong-position",
            "wrong-ply-count",
            "wrong-status",
            "wrong-outcome",
        ],
    )
    def test_corrupted_replays_fail_verification_with_a_reason(
        self, mutate: Callable[[dict[str, Any]], None], message: str
    ) -> None:
        """A replay that disagrees with the engine is reported, never crashed on."""
        replay = tamper(roundtrip(match_document(sync_match())), mutate)
        check = verify_replay(replay)

        assert not check.ok
        assert any(message in problem for problem in check.problems), check.problems

    def test_a_decision_recorded_after_the_game_ended_is_refused(self) -> None:
        """The applied stream cannot continue past a finished game."""
        document = roundtrip(match_document(sync_match()))
        decisions = document["result"]["decisions"]
        decisions.append(decisions[-1])
        check = verify_replay(decode_replay(document))

        assert not check.ok
        assert "already finished" in check.problems[0]


class TestSourceRevision:
    """Provenance detection, which is best effort by design."""

    def test_a_loose_branch_reference_is_read(self, tmp_path: Path) -> None:
        """The ordinary case: HEAD points at a branch whose ref file exists."""
        git = tmp_path / ".git"
        (git / "refs" / "heads").mkdir(parents=True)
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n", encoding="utf-8")
        assert detect_source_revision(tmp_path) == "a" * 40

    def test_a_detached_head_is_read(self, tmp_path: Path) -> None:
        """A detached HEAD holds the commit directly."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("b" * 40, encoding="utf-8")
        assert detect_source_revision(tmp_path) == "b" * 40

    def test_a_tree_without_a_repository_reports_nothing(self, tmp_path: Path) -> None:
        """Provenance is optional: an unknown revision is recorded as null."""
        assert detect_source_revision(tmp_path / "empty") is None


class TestTimedMatchReplay:
    """The acceptance case: a real timed match, replayed without agents."""

    def test_a_saved_timed_match_replays_without_agents_or_timing(self, tmp_path: Path) -> None:
        """A recorded timed match reproduces from its file alone.

        The match really starts a worker per decision. Verification then runs
        far inside a single decision's budget and leaves no children behind,
        which is what "no agent was built" looks like from the outside.
        """
        runner = MatchRunner(lineup(), MatchConfig(seconds_per_turn=BUDGET))
        result = runner.run(deal_seed=7)
        assert result.status is MatchStatus.FINISHED
        assert result.outcome is not None

        path = write_match(result, tmp_path / "results" / "match.json")
        replay = read_replay(path)
        started = time.perf_counter()
        check = verify_replay(replay)
        elapsed = time.perf_counter() - started

        assert check.ok, check.problems
        assert check.outcome == result.outcome
        assert check.applied == len(result.decisions)
        assert elapsed < BUDGET
        assert not multiprocessing.active_children()
