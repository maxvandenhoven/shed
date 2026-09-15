"""Tests for the local HTTP surface and the launcher.

The API is driven twice over: as plain function calls through
:func:`~shed.companion.api.handle_api`, which is where the endpoint behaviour is
pinned down, and over a real socket for the handful of properties that only exist
once HTTP is involved -- the bundled assets, the status codes, the body limit, and
the launcher's own behaviour when a port is taken.
"""

from __future__ import annotations

import errno
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import pytest

from shed.agents import AGENT_KINDS
from shed.companion.api import ASSETS, MAX_BODY_BYTES, handle_api
from shed.companion.codec import CompanionDataError
from shed.companion.observed import ObservationError
from shed.companion.server import build_parser, build_server, main
from shed.companion.session import COMPANION_SCHEMA_VERSION, decode_session, derive

NEW_GAME: dict[str, Any] = {
    "my_hand": ["3", "7", "10"],
    "my_face_up": ["4", "9", "A"],
    "opponent_face_up": ["5", "6", "K"],
    "starting_player": "me",
}
"""A new-game request body, matching the ``opening`` fixture's cards."""


def _event(session: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    """Post one observation through the API.

    Args:
        session: The session document the browser holds.
        event: The observation to append.

    Returns:
        The reply envelope.
    """
    return handle_api("/api/event", {"session": session, "event": event})


def test_health_reports_the_schema_and_the_profile() -> None:
    """The one endpoint a page can call before it has a game."""
    health = handle_api("/api/health", {})
    assert health["ok"] is True
    assert health["schema_version"] == COMPANION_SCHEMA_VERSION
    assert health["rules"] == "standard"


def test_a_new_game_comes_back_with_a_document_and_a_recommendation() -> None:
    """Setup, folding, and advice are one round trip."""
    reply = handle_api("/api/new", NEW_GAME)
    assert reply["session"]["schema_version"] == COMPANION_SCHEMA_VERSION
    assert reply["session"]["events"] == []
    assert reply["state"]["deck_count"] == 36
    assert reply["recommendation"]["headline"].startswith("Play")
    assert "not optimal play" in reply["recommendation"]["caveat"]


def test_every_reply_carries_the_session_and_the_request_id() -> None:
    """The browser saves whatever the last accepted reply carried, and drops stale ones."""
    reply = handle_api("/api/new", {**NEW_GAME, "request_id": 42})
    assert reply["request_id"] == 42
    assert "session" in reply
    assert reply["revision"].startswith("0-")


def test_an_observation_is_appended_and_the_screen_comes_back_with_it() -> None:
    """One tap, one request, one saved document."""
    started = handle_api("/api/new", NEW_GAME)
    played = _event(started["session"], {"kind": "play", "player": "me", "rank": "3", "count": 1})
    assert played["event_count"] == 1
    assert played["applied"] == 1
    assert played["state"]["pending"][0]["count"] == 1
    assert played["history"][0]["text"] == "You played 1 x 3"


def test_a_contradicted_observation_is_refused_without_touching_the_document() -> None:
    """The rejected entry never reaches the log, so the browser keeps what it had."""
    started = handle_api("/api/new", NEW_GAME)
    with pytest.raises(ObservationError, match="cannot play"):
        _event(started["session"], {"kind": "play", "player": "me", "rank": "3", "count": 2})
    assert started["session"]["events"] == []


def test_an_observation_out_of_turn_names_whose_turn_it_is() -> None:
    """The message is what the interface shows beside the operator's input."""
    started = handle_api("/api/new", NEW_GAME)
    with pytest.raises(ObservationError, match="not your opponent's"):
        _event(started["session"], {"kind": "play", "player": "opponent", "rank": "5", "count": 1})


def test_undo_shortens_the_log() -> None:
    """Recovery is the document's own shape, so the endpoint is trivial."""
    started = handle_api("/api/new", NEW_GAME)
    played = _event(started["session"], {"kind": "play", "player": "me", "rank": "3", "count": 1})
    undone = handle_api("/api/undo", {"session": played["session"]})
    assert undone["event_count"] == 0
    assert undone["session"] == started["session"]


def test_loading_a_document_reproduces_the_screen_it_was_saved_from() -> None:
    """This is the refresh, the browser restart, and the Python restart, all at once."""
    started = handle_api("/api/new", NEW_GAME)
    played = _event(started["session"], {"kind": "play", "player": "me", "rank": "3", "count": 1})
    reloaded = handle_api("/api/state", {"session": json.loads(json.dumps(played["session"]))})
    assert reloaded["state"] == played["state"]
    assert reloaded["revision"] == played["revision"]
    assert reloaded["recommendation"] == played["recommendation"]


def test_joining_an_ongoing_game_without_a_deck_count_asks_for_it() -> None:
    """Advice is withheld and the missing observation is named."""
    reply = handle_api(
        "/api/join",
        {
            "my_hand": ["3", "4"],
            "my_face_up": ["5"],
            "my_face_down": 2,
            "opponent_hand_count": 4,
            "opponent_hand_known": [],
            "opponent_face_up": ["6"],
            "opponent_face_down": 2,
            "deck_count": None,
            "pile": [None, "A"],
            "constraint": {"kind": "at_least", "rank": "A"},
            "to_act": "me",
        },
    )
    assert reply["state"]["deck_count"] is None
    assert reply["recommendation"] is None
    assert any(blocker["code"] == "deck_unknown" for blocker in reply["blockers"])


def test_a_correction_then_advice_recovers_an_incomplete_join() -> None:
    """The documented way out: record the count, and the recommendation appears."""
    joined = handle_api(
        "/api/join",
        {
            "my_hand": ["3", "9"],
            "my_face_up": ["5"],
            "my_face_down": 2,
            "opponent_hand_count": 3,
            "opponent_hand_known": [],
            "opponent_face_up": ["6"],
            "opponent_face_down": 2,
            "deck_count": None,
            "pile": [],
            "constraint": {"kind": "unrestricted"},
            "to_act": "me",
        },
    )
    fixed = _event(
        joined["session"],
        {
            "kind": "correct",
            "patch": {"deck_count": 20, "burned_count": 54 - 20 - 11},
            "note": "counted the deck",
        },
    )
    assert fixed["state"]["deck_count"] == 20
    assert fixed["recommendation"] is not None
    assert fixed["history"][0]["text"] == "Correction recorded, counted the deck"


def test_an_event_on_a_log_that_no_longer_folds_is_refused() -> None:
    """Appending onto a damaged document would bury the real problem one entry deeper."""
    started = handle_api("/api/new", NEW_GAME)
    damaged = dict(started["session"])
    damaged["events"] = [{"kind": "play", "player": "me", "rank": "4", "count": 1}]
    with pytest.raises(ObservationError, match="no longer applies"):
        _event(damaged, {"kind": "play", "player": "me", "rank": "3", "count": 1})


def test_a_damaged_log_can_still_be_loaded_and_undone() -> None:
    """A session that stops folding has to stay recoverable, not become unreadable."""
    started = handle_api("/api/new", NEW_GAME)
    damaged = dict(started["session"])
    damaged["events"] = [{"kind": "play", "player": "me", "rank": "4", "count": 1}]
    loaded = handle_api("/api/state", {"session": damaged})
    assert loaded["replay_error"] is not None
    assert loaded["applied"] == 0
    assert handle_api("/api/undo", {"session": damaged})["event_count"] == 0


@pytest.mark.parametrize(
    ("path", "payload", "message"),
    [
        ("/api/state", {}, "must carry a 'session'"),
        ("/api/event", {"session": None}, "must be an object"),
        ("/api/new", {"my_hand": ["3", "7", "10"]}, "must be an array"),
        ("/api/nope", {}, "is not an endpoint"),
    ],
)
def test_malformed_requests_are_refused_with_the_field_named(
    path: str, payload: dict[str, Any], message: str
) -> None:
    """Decoding failures are told apart from contradicted observations."""
    with pytest.raises(CompanionDataError, match=message):
        handle_api(path, payload)


def test_a_recommendation_is_tagged_with_the_revision_it_was_asked_for() -> None:
    """The browser drops a reply whose revision has moved on; the tag makes that possible."""
    started = handle_api("/api/new", NEW_GAME)
    played = _event(started["session"], {"kind": "play", "player": "me", "rank": "3", "count": 1})
    assert started["revision"] != played["revision"]
    assert derive(decode_session(played["session"])).applied == 1


# --------------------------------------------------------------------- over HTTP


@pytest.fixture
def companion() -> Iterator[str]:
    """Serve the companion on a free loopback port for one test.

    Yields:
        The base URL, such as ``http://127.0.0.1:54321``.
    """
    server = build_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(base: str, path: str) -> tuple[int, str, bytes]:
    """Fetch one path.

    Args:
        base: The server's base URL.
        path: The path to fetch.

    Returns:
        The status, the content type, and the body.
    """
    try:
        with urllib.request.urlopen(base + path) as response:  # noqa: S310 - loopback only.
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def _post(base: str, path: str, body: bytes) -> tuple[int, dict[str, Any]]:
    """Post one raw body.

    Args:
        base: The server's base URL.
        path: The endpoint.
        body: The encoded request body.

    Returns:
        The status and the decoded reply.
    """
    request = urllib.request.Request(  # noqa: S310 - loopback only.
        base + path, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310 - loopback only.
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_every_bundled_asset_is_served_from_the_package(companion: str) -> None:
    """No CDN, no remote font: the allowlist is the whole inventory."""
    for path, (_, content_type) in ASSETS.items():
        status, served_type, body = _get(companion, path)
        assert status == 200
        assert served_type == content_type
        assert body


def test_the_page_loads_only_same_origin_assets(companion: str) -> None:
    """A phone in aeroplane mode has to render the same page as a connected one."""
    page = _get(companion, "/")[2].decode("utf-8")
    assert "http://" not in page.replace("http://127.0.0.1:8000", "")
    assert "https://" not in page
    assert 'src="app.js"' in page
    assert 'href="styles.css"' in page


def test_an_unknown_path_is_a_json_404(companion: str) -> None:
    """Nothing outside the allowlist is reachable, traversal attempts included."""
    status, _, body = _get(companion, "/../pyproject.toml")
    assert status == 404
    assert "error" in json.loads(body)


def test_a_contradicted_observation_is_a_400_the_page_can_show(companion: str) -> None:
    """The status separates 'fix your entry' from 'this page is out of date'."""
    status, started = _post(companion, "/api/new", json.dumps(NEW_GAME).encode())
    assert status == 200
    status, error = _post(
        companion,
        "/api/event",
        json.dumps(
            {
                "session": started["session"],
                "event": {"kind": "play", "player": "me", "rank": "3", "count": 2},
            }
        ).encode(),
    )
    assert status == 400
    assert error["kind"] == "observation"
    assert "cannot play" in error["error"]


def test_a_malformed_body_is_a_400_and_says_so(companion: str) -> None:
    """A client bug is reported as one, rather than as a rejected observation."""
    status, error = _post(companion, "/api/state", b"{not json")
    assert status == 400
    assert error["kind"] == "data"


def test_an_oversized_body_is_refused_before_it_is_read(companion: str) -> None:
    """A malformed length cannot make the server allocate without limit."""
    request = urllib.request.Request(  # noqa: S310 - loopback only.
        companion + "/api/state",
        data=b"{}",
        headers={"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request)  # noqa: S310 - loopback only.
    assert raised.value.code == 400


def test_assets_are_served_with_no_store(companion: str) -> None:
    """An updated checkout must take effect on the next reload, not eventually."""
    with urllib.request.urlopen(companion + "/") as response:  # noqa: S310 - loopback only.
        assert response.headers.get("Cache-Control") == "no-store"
        assert response.headers.get("X-Content-Type-Options") == "nosniff"


def test_a_full_turn_works_over_http(companion: str) -> None:
    """Setup, recommendation, my play, the drawn rank, and their play in one session."""
    _, started = _post(companion, "/api/new", json.dumps(NEW_GAME).encode())
    session = started["session"]
    suggested = started["recommendation"]["move"]
    assert suggested["kind"] == "play"

    def send(event: dict[str, Any]) -> dict[str, Any]:
        """Post one observation and keep the returned document.

        Args:
            event: The observation.

        Returns:
            The reply.
        """
        nonlocal session
        status, reply = _post(
            companion, "/api/event", json.dumps({"session": session, "event": event}).encode()
        )
        assert status == 200, reply
        session = reply["session"]
        return reply

    played = send(
        {
            "kind": "play",
            "player": "me",
            "rank": suggested["rank"],
            "count": suggested["count"],
        }
    )
    assert played["state"]["pending"]
    recorded = send({"kind": "record", "ranks": ["2"]})
    assert not recorded["state"]["pending"]
    theirs = send({"kind": "play", "player": "opponent", "rank": "9", "count": 1})
    assert theirs["state"]["my_turn"] is True
    assert theirs["recommendation"] is not None
    undone = _post(companion, "/api/undo", json.dumps({"session": session}).encode())[1]
    assert undone["event_count"] == 2


def test_a_taken_port_is_reported_rather_than_raised(
    companion: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator sees one line and another port to try, not a traceback."""
    port = int(companion.rsplit(":", 1)[1])
    assert main(["--port", str(port)]) == 1
    assert "already in use" in capsys.readouterr().err


def test_a_host_the_phone_cannot_bind_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    """An unresolvable host is refused before a socket is opened."""
    assert main(["--host", "not.a.host.invalid"]) == 1
    assert "not an address" in capsys.readouterr().err


def test_the_launcher_defaults_to_loopback_and_the_documented_port() -> None:
    """The defaults are the ones the Termux instructions tell the operator to use."""
    args = build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.log_requests is False


@pytest.mark.parametrize("value", ["-1", "65536", "eight"])
def test_the_launcher_refuses_a_port_that_is_not_one(value: str) -> None:
    """A mistyped port fails at the command line rather than at bind time."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--port", value])


def test_a_bound_server_releases_its_port_on_close() -> None:
    """An immediate restart on the same port has to work, not report it in use."""
    server = build_server("127.0.0.1", 0)
    port = server.server_address[1]
    server.server_close()
    again = build_server("127.0.0.1", port)
    try:
        assert again.server_address[1] == port
    finally:
        again.server_close()
    assert errno.EADDRINUSE  # The code the launcher translates into its message.


def test_the_agent_catalogue_comes_from_the_package() -> None:
    """The picker cannot fall behind the agents the package actually builds."""
    catalogue = handle_api("/api/agents", {})
    assert [entry["kind"] for entry in catalogue["agents"]] == list(AGENT_KINDS)
    assert catalogue["default"] in AGENT_KINDS
    for entry in catalogue["agents"]:
        assert entry["label"] and entry["summary"] and entry["caveat"]


def test_health_says_which_document_versions_it_reads() -> None:
    """A page holding an older document can tell whether this server will take it."""
    health = handle_api("/api/health", {})
    assert health["schema_version"] == COMPANION_SCHEMA_VERSION
    assert 1 in health["reads_schema_versions"]


def test_a_new_game_records_the_chosen_agent() -> None:
    """The choice is made at setup and stored with the game."""
    reply = handle_api("/api/new", {**NEW_GAME, "agent": {"kind": "random", "name": "random"}})
    assert reply["session"]["agent"]["kind"] == "random"
    assert reply["agent"]["label"] == "Random"
    assert reply["recommendation"]["agent"]["kind"] == "random"
    assert "sampled uniformly" in reply["recommendation"]["reasoning"]


def test_a_new_game_without_a_chosen_agent_uses_the_default() -> None:
    """Never opening the picker gives the shedding baseline, not a failure."""
    reply = handle_api("/api/new", NEW_GAME)
    assert reply["agent"]["kind"] == "greedy"
    assert (
        "retention" in reply["recommendation"]["reasoning"]
        or "only rank" in (reply["recommendation"]["reasoning"])
    )


def test_a_joined_game_records_the_chosen_agent() -> None:
    """Both setup paths take a choice."""
    reply = handle_api(
        "/api/join",
        {
            "my_hand": ["3", "9"],
            "my_face_up": ["5"],
            "my_face_down": 2,
            "opponent_hand_count": 3,
            "opponent_hand_known": [],
            "opponent_face_up": ["6"],
            "opponent_face_down": 2,
            "deck_count": 20,
            "pile": [],
            "constraint": {"kind": "unrestricted"},
            "to_act": "me",
            "agent": {"kind": "random", "name": "random", "seed": 11},
        },
    )
    assert reply["session"]["agent"] == {
        "kind": "random",
        "name": "random",
        "seed": 11,
        "label": "Random",
        "summary": reply["agent"]["summary"],
        "caveat": reply["agent"]["caveat"],
    }


def test_the_chosen_agent_survives_every_later_request() -> None:
    """Observations, undo, and a reload all keep the strategy the game was started with."""
    started = handle_api("/api/new", {**NEW_GAME, "agent": {"kind": "random", "name": "random"}})
    played = _event(started["session"], {"kind": "play", "player": "me", "rank": "3", "count": 1})
    assert played["agent"]["kind"] == "random"
    undone = handle_api("/api/undo", {"session": played["session"]})
    assert undone["agent"]["kind"] == "random"
    reloaded = handle_api("/api/state", {"session": json.loads(json.dumps(undone["session"]))})
    assert reloaded["agent"]["kind"] == "random"


def test_a_setup_naming_an_unknown_agent_is_refused() -> None:
    """A typed or stale kind fails at setup rather than at the first suggestion."""
    with pytest.raises(CompanionDataError, match="not an agent this release ships"):
        handle_api("/api/new", {**NEW_GAME, "agent": {"kind": "mcts", "name": "mcts"}})


def test_a_version_1_document_is_accepted_over_the_api() -> None:
    """A game saved before the picker existed keeps working across the upgrade."""
    started = handle_api("/api/new", NEW_GAME)
    older = dict(started["session"])
    older["schema_version"] = 1
    older.pop("agent", None)
    reloaded = handle_api("/api/state", {"session": older})
    assert reloaded["session"]["schema_version"] == COMPANION_SCHEMA_VERSION
    assert reloaded["agent"]["kind"] == "greedy"


def test_the_agent_catalogue_is_reachable_before_a_game_exists(companion: str) -> None:
    """The picker is drawn on the setup screen, which has no session to send."""
    status, content_type, body = _get(companion, "/api/agents")
    assert status == 200
    assert content_type.startswith("application/json")
    assert [entry["kind"] for entry in json.loads(body)["agents"]] == list(AGENT_KINDS)
