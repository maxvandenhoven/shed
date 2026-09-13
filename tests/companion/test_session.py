"""Tests for the session document: codec, replay, undo, corrections, round trips.

A session is the whole recovery story, so these tests exercise it as one: a
document written out and read back must fold to the same position, an undo must be
the log's own shape rather than an inverse operation, a correction must survive a
replay like any other entry, and a document this release cannot read must be
refused rather than half-understood.
"""

from __future__ import annotations

import json

import pytest

from shed.companion.codec import (
    CompanionDataError,
    decode_event,
    decode_state,
    encode_event,
    encode_state,
)
from shed.companion.observed import (
    ME,
    OPPONENT,
    AtLeast,
    CorrectState,
    ObservationError,
    ObservationEvent,
    ObservedState,
    Phase,
    PickUpPile,
    PlayCards,
    Rank,
    RecordCards,
    RevealFaceDown,
    StatePatch,
)
from shed.companion.session import (
    COMPANION_SCHEMA_VERSION,
    Session,
    decode_session,
    derive,
    describe_event,
    encode_session,
    render,
    revision,
)
from tests.companion.conftest import craft, session_for


def _fold(session: Session) -> ObservedState:
    """Fold a session and require that the whole log applied.

    Args:
        session: The document.

    Returns:
        The position it describes.

    Raises:
        AssertionError: If any entry failed, which no test here expects silently.
    """
    derivation = derive(session)
    assert derivation.error is None, derivation.error
    return derivation.state


def test_a_state_survives_a_json_round_trip_without_learning_anything() -> None:
    """Encoding and decoding is lossless in both directions, ignorance included."""
    state = craft(
        my_hand=(Rank.THREE, Rank.SEVEN),
        my_face_up=(Rank.FOUR,),
        my_face_down=2,
        opponent_hand_known=(Rank.KING,),
        opponent_hand_unknown=3,
        opponent_face_up=(Rank.FIVE,),
        opponent_face_down=2,
        deck_count=5,
        pile=(None, Rank.ACE),
        constraint=AtLeast(Rank.ACE),
    )
    restored = decode_state(json.loads(json.dumps(encode_state(state))))
    assert restored == state
    assert restored.pile == (None, Rank.ACE)
    assert restored.seat(OPPONENT).hand_unknown == 3


@pytest.mark.parametrize(
    "event",
    [
        PlayCards(ME, Rank.SEVEN, 2),
        PickUpPile(OPPONENT),
        RevealFaceDown(ME, Rank.JOKER),
        RecordCards((Rank.TEN, Rank.TWO)),
        CorrectState(StatePatch(deck_count=4, my_hand=(Rank.THREE,)), note="miscount"),
    ],
)
def test_every_event_survives_a_json_round_trip(event: ObservationEvent) -> None:
    """The log is the document, so each entry has to encode exactly."""
    assert decode_event(json.loads(json.dumps(encode_event(event)))) == event


def test_a_correction_encodes_only_the_fields_it_set() -> None:
    """A history entry shows what was corrected, not every field it could have been."""
    encoded = encode_event(CorrectState(StatePatch(deck_count=4)))
    assert encoded["patch"] == {"deck_count": 4}


def test_an_unknown_correction_field_is_refused() -> None:
    """A typo in a correction must not look as though it had been applied."""
    with pytest.raises(CompanionDataError, match="unknown correction field"):
        decode_event({"kind": "correct", "patch": {"my_hnad": ["3"]}})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"kind": "play", "player": "me", "rank": "Z", "count": 1}, "is not a rank"),
        ({"kind": "play", "player": "nobody", "rank": "3", "count": 1}, "is not a seat"),
        ({"kind": "play", "player": "me", "rank": "3", "count": "two"}, "whole number"),
        ({"kind": "play", "player": "me", "rank": "3", "count": True}, "whole number"),
        ({"kind": "play", "player": "me", "rank": "3", "count": 0}, "at least 1"),
        ({"kind": "shuffle"}, "is not an observation"),
    ],
)
def test_malformed_events_are_refused_with_the_field_named(
    payload: dict[str, object], message: str
) -> None:
    """Decoding is the one place untyped data is checked, so it names the field."""
    with pytest.raises(CompanionDataError, match=message):
        decode_event(payload)


def test_a_document_from_another_schema_version_is_refused() -> None:
    """A future document is not read optimistically."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1)
    document = encode_session(session_for(state))
    document["schema_version"] = COMPANION_SCHEMA_VERSION + 1
    with pytest.raises(CompanionDataError, match="reads version"):
        decode_session(document)


def test_a_session_round_trips_through_json_and_folds_to_the_same_position() -> None:
    """This is the export/import and refresh path, and the restart path with it."""
    session = session_for(
        craft(
            my_hand=(Rank.THREE, Rank.SEVEN, Rank.TEN),
            my_face_down=2,
            opponent_hand_unknown=3,
            opponent_face_down=2,
            deck_count=10,
        )
    )
    for event in (
        PlayCards(ME, Rank.THREE, 1),
        RecordCards((Rank.FIVE,)),
        PlayCards(OPPONENT, Rank.KING, 1),
        PlayCards(ME, Rank.TEN, 1),
    ):
        session = session.appended(event)
    restored = decode_session(json.loads(json.dumps(encode_session(session))))
    assert restored == session
    assert _fold(restored) == _fold(session)
    assert revision(restored) == revision(session)


def test_undo_removes_exactly_the_last_entry() -> None:
    """Undo is a shorter log, which is why it works on every kind of entry."""
    session = session_for(
        craft(my_hand=(Rank.THREE, Rank.SEVEN), my_face_down=1, opponent_hand_unknown=2)
    )
    played = session.appended(PlayCards(ME, Rank.THREE, 1))
    assert played.undone() == session
    assert _fold(played.undone()).seat(ME).hand_known == (Rank.THREE, Rank.SEVEN)


def test_undo_takes_back_a_correction_like_any_other_entry() -> None:
    """Corrections are recorded, not applied behind the log's back."""
    session = session_for(craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1, deck_count=3))
    corrected = session.appended(CorrectState(StatePatch(deck_count=2, burned_count=50)))
    assert _fold(corrected).deck_count == 2
    assert _fold(corrected.undone()).deck_count == 3


def test_undo_on_an_empty_log_is_refused() -> None:
    """The interface disables the button; the model still says no."""
    session = session_for(craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1))
    with pytest.raises(ObservationError, match="nothing to undo"):
        session.undone()


def test_a_correction_appears_in_the_history_with_its_note() -> None:
    """The operator can read back why the position was changed."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1, deck_count=3)
    line = describe_event(CorrectState(StatePatch(deck_count=2), note="miscounted deck"), state)
    assert line == "Correction recorded -- miscounted deck"


def test_the_history_says_how_big_a_picked_up_pile_was() -> None:
    """A line is written against the position it applied to, so it can say that."""
    state = craft(
        my_hand=(Rank.THREE,),
        opponent_hand_unknown=1,
        pile=(Rank.KING, Rank.ACE),
        constraint=AtLeast(Rank.ACE),
        to_act=OPPONENT,
    )
    assert describe_event(PickUpPile(OPPONENT), state) == "Opponent picked up the pile (2 cards)"


def test_a_log_that_no_longer_folds_is_reported_rather_than_raised() -> None:
    """A damaged session still has to be readable, so it can be undone or exported."""
    session = session_for(
        craft(my_hand=(Rank.THREE, Rank.SEVEN), my_face_down=1, opponent_hand_unknown=2)
    )
    broken = session.appended(PlayCards(ME, Rank.FOUR, 1))
    derivation = derive(broken)
    assert derivation.error is not None
    assert derivation.applied == 0
    assert derivation.state == session.initial


def test_an_initial_position_that_describes_no_table_does_raise() -> None:
    """There is nothing to show or undo, so this one failure is not recoverable."""
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=1, deck_count=3)
    document = encode_session(session_for(state))
    document["initial"]["deck_count"] = 40
    with pytest.raises(ObservationError, match="add up to"):
        derive(decode_session(document))


def test_the_revision_changes_whenever_anything_does() -> None:
    """A recommendation is tagged with this, so it must not survive a change."""
    session = session_for(
        craft(my_hand=(Rank.THREE, Rank.SEVEN), my_face_down=1, opponent_hand_unknown=2)
    )
    played = session.appended(PlayCards(ME, Rank.THREE, 1))
    assert revision(session) != revision(played)
    assert revision(session) == revision(session_for(session.initial))


def test_a_rendered_screen_is_a_pure_function_of_the_document() -> None:
    """Two renders of one document agree, agent tie-breaking included."""
    session = session_for(
        craft(my_face_down=3, opponent_hand_unknown=2, opponent_face_down=1, deck_count=0)
    )
    assert render(session) == render(session)


def test_a_rendered_screen_offers_one_reveal_for_indistinguishable_cards() -> None:
    """Three face-down cards are one decision, so they are one button."""
    session = session_for(
        craft(my_face_down=3, opponent_hand_unknown=2, opponent_face_down=1, deck_count=0)
    )
    options = render(session)["options"]
    assert [option["kind"] for option in options] == ["reveal"]


def test_a_rendered_screen_reports_uncertainty_rather_than_hiding_it() -> None:
    """Counts of unknown cards reach the screen as counts."""
    session = session_for(
        craft(
            my_hand=(Rank.THREE,),
            opponent_hand_known=(Rank.KING,),
            opponent_hand_unknown=2,
            pile=(None, Rank.ACE),
            constraint=AtLeast(Rank.ACE),
            deck_count=4,
        )
    )
    rendered = render(session)
    assert rendered["state"]["pile"] == {
        "size": 2,
        "unknown": 1,
        "top": "A",
        "ranks": [None, "A"],
    }
    theirs = rendered["state"]["seats"][1]
    assert theirs["hand_unknown"] == 2
    assert theirs["hand_known"] == [{"rank": "K", "count": 1, "label": "K"}]


def test_a_rendered_screen_replaces_advice_with_what_to_record_next() -> None:
    """No move is shown when the position cannot support one."""
    session = session_for(
        craft(my_hand=(Rank.THREE,), opponent_hand_unknown=2, deck_count=4, to_act=OPPONENT)
    )
    rendered = render(session)
    assert rendered["recommendation"] is None
    assert [blocker["code"] for blocker in rendered["blockers"]] == ["not_my_turn"]
    assert rendered["options"] == []


def test_a_rendered_screen_asks_for_an_uncounted_deck() -> None:
    """The incomplete ongoing-game case names the observation it needs."""
    session = session_for(craft(my_hand=(Rank.THREE,), opponent_hand_unknown=2, deck_count=None))
    rendered = render(session)
    assert rendered["recommendation"] is None
    assert rendered["state"]["deck_count"] is None
    assert any(blocker["code"] == "deck_unknown" for blocker in rendered["blockers"])


def test_a_game_played_down_to_a_face_down_card_replays_from_its_log() -> None:
    """The document is the game: a long log folds to one position, every time.

    The sequence walks the whole zone ordering for one seat -- hand, then face-up once
    the hand and deck are gone, then a blind reveal that fails and takes the pile --
    which is the path most likely to drift from the engine if a rule were restated.
    """
    session = session_for(
        craft(
            my_hand=(Rank.THREE,),
            my_face_up=(Rank.NINE,),
            my_face_down=2,
            opponent_hand_unknown=4,
            opponent_face_up=(Rank.FIVE,),
            opponent_face_down=2,
            deck_count=0,
        )
    )
    log = (
        PlayCards(ME, Rank.THREE, 1),
        PlayCards(OPPONENT, Rank.FOUR, 1),
        PlayCards(ME, Rank.NINE, 1),
        PlayCards(OPPONENT, Rank.EIGHT, 1),
        RevealFaceDown(ME, Rank.SIX),
    )
    for event in log:
        session = session.appended(event)
    state = _fold(session)
    assert derive(session).applied == len(log)
    assert state.seat(ME).face_down == 1
    assert state.seat(ME).hand_known == (Rank.THREE, Rank.FOUR, Rank.SIX, Rank.EIGHT, Rank.NINE)
    assert state.pile == ()
    assert state.to_act == OPPONENT
    assert _fold(decode_session(json.loads(json.dumps(encode_session(session))))) == state


def test_a_win_is_rendered_as_a_finished_game() -> None:
    """The last card out ends the game on screen as well as in the model."""
    session = session_for(
        craft(my_hand=(Rank.FOUR,), opponent_hand_unknown=2, deck_count=0)
    ).appended(PlayCards(ME, Rank.FOUR, 1))
    rendered = render(session)
    assert rendered["state"]["finished"] is True
    assert rendered["state"]["winner"] == "me"
    assert _fold(session).phase is Phase.FINISHED
