"""The local HTTP surface: bundled assets and a stateless JSON API.

The server is a pure function of the request. It keeps no session, no cache, and
no file on disk: every endpoint takes the whole session document, folds it, and
answers. What that buys is the reliability property the companion needs most --
the Python process is disposable. Kill it mid-game, restart it, run it on a
different port, and the browser resends the document it has been saving all along.

Every asset is served from this package. There is no CDN, no remote font, no
analytics, and nothing in the page reaches the network except ``fetch`` calls to
this same origin, so a phone in aeroplane mode plays exactly as well as one with a
signal. Assets are an explicit allowlist rather than a directory walk, which is
both the traversal defence and the inventory.

Responses set no CORS headers, so a page on another origin can post to this server
but cannot read the answer. Combined with the server holding no state, there is
nothing for a hostile page to steal or corrupt; the default bind is loopback
regardless.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import shed
from shed.companion.advice import DEFAULT_AGENT, agent_catalogue
from shed.companion.codec import (
    CompanionDataError,
    decode_agent,
    decode_constraint,
    decode_event,
    decode_optional_ranks,
    decode_ranks,
    decode_seat,
)
from shed.companion.observed import (
    ObservationError,
    apply_event,
    join_game,
    new_game,
)
from shed.companion.session import (
    COMPANION_SCHEMA_VERSION,
    READABLE_SCHEMA_VERSIONS,
    Session,
    decode_session,
    derive,
    encode_session,
    render,
)

__all__ = ["ASSETS", "MAX_BODY_BYTES", "CompanionHandler", "handle_api"]

STATIC_ROOT: Path = Path(__file__).parent / "static"
"""Directory holding the bundled page, stylesheet, and script."""

ASSETS: Mapping[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
"""Every path this server serves as a file, mapped to its name and media type.

An allowlist rather than a root directory: a request can only ever name one of
these, so no path can escape :data:`STATIC_ROOT`, and the table doubles as the
list of what ships.
"""

MAX_BODY_BYTES = 1 << 20
"""Largest request body accepted, one mebibyte.

A session document is a few kilobytes even after a long game; this bound exists so
a malformed ``Content-Length`` cannot make the server allocate without limit.
"""


def _session_from(payload: Mapping[str, Any]) -> Session:
    """Decode the session a request carried.

    Args:
        payload: The decoded request body.

    Returns:
        The session.

    Raises:
        CompanionDataError: If the body carries no readable session.
    """
    if "session" not in payload:
        raise CompanionDataError("request must carry a 'session'")
    return decode_session(payload["session"], "session")


def _reply(session: Session, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the envelope every successful endpoint returns.

    The session travels back with the rendered screen so the browser's persistence
    is one rule -- save whatever the last accepted reply carried -- whether the reply
    came from loading, an observation, an undo, or a fresh setup.

    Args:
        session: The session the answer is about.
        payload: The request, read for the optional ``request_id``.

    Returns:
        The rendered screen, the document that produced it, and the request id the
        browser uses to drop replies that arrive out of order.
    """
    rendered = render(session)
    rendered["session"] = encode_session(session)
    rendered["request_id"] = payload.get("request_id")
    return rendered


def _new_session(payload: Mapping[str, Any]) -> Session:
    """Build a session for a game that is starting now.

    Args:
        payload: The request body, carrying both face-up sets, my hand, and who
            actually played first.

    Returns:
        The new session.

    Raises:
        CompanionDataError: If a field is missing or the wrong shape.
        ObservationError: If the entered cards could not be a real deal.
    """
    return Session(
        schema_version=COMPANION_SCHEMA_VERSION,
        initial=new_game(
            my_hand=decode_ranks(payload.get("my_hand"), "my_hand"),
            my_face_up=decode_ranks(payload.get("my_face_up"), "my_face_up"),
            opponent_face_up=decode_ranks(payload.get("opponent_face_up"), "opponent_face_up"),
            starting_player=decode_seat(payload.get("starting_player"), "starting_player"),
        ),
        events=(),
        agent=decode_agent(payload.get("agent")),
    )


def _joined_session(payload: Mapping[str, Any]) -> Session:
    """Build a session for a game already under way.

    Args:
        payload: The request body, carrying every count and public card the operator
            could read off the table.

    Returns:
        The new session.

    Raises:
        CompanionDataError: If a field is missing or the wrong shape.
        ObservationError: If the entered position could not describe a real table.
    """
    deck = payload.get("deck_count")
    return Session(
        schema_version=COMPANION_SCHEMA_VERSION,
        initial=join_game(
            my_hand=decode_ranks(payload.get("my_hand"), "my_hand"),
            my_face_up=decode_ranks(payload.get("my_face_up"), "my_face_up"),
            my_face_down=_count(payload, "my_face_down"),
            opponent_hand_count=_count(payload, "opponent_hand_count"),
            opponent_hand_known=decode_ranks(
                payload.get("opponent_hand_known", []), "opponent_hand_known"
            ),
            opponent_face_up=decode_ranks(payload.get("opponent_face_up"), "opponent_face_up"),
            opponent_face_down=_count(payload, "opponent_face_down"),
            deck_count=None if deck is None else _count(payload, "deck_count"),
            pile=decode_optional_ranks(payload.get("pile", []), "pile"),
            constraint=decode_constraint(
                payload.get("constraint", {"kind": "unrestricted"}), "constraint"
            ),
            to_act=decode_seat(payload.get("to_act"), "to_act"),
        ),
        events=(),
        agent=decode_agent(payload.get("agent")),
    )


def _count(payload: Mapping[str, Any], name: str) -> int:
    """Read one non-negative count from a request body.

    Args:
        payload: The decoded body.
        name: Field name.

    Returns:
        The count.

    Raises:
        CompanionDataError: If the field is missing, not a whole number, or
            negative. ``True`` is an ``int`` in Python, so booleans are refused
            explicitly rather than counted as one.
    """
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompanionDataError(f"{name} must be a whole number")
    if value < 0:
        raise CompanionDataError(f"{name} must not be negative, got {value}")
    return value


def handle_api(path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one decoded API request.

    Separating this from the HTTP plumbing is what lets the tests drive every
    endpoint as a function call, and what keeps the transport free of game logic.

    Args:
        path: The request path, such as ``"/api/event"``.
        payload: The decoded request body.

    Returns:
        The reply envelope; see :func:`_reply`.

    Raises:
        CompanionDataError: If the path is not an endpoint, or the body is
            malformed.
        ObservationError: If the observation contradicts the tracked game. The
            caller turns this into a 400 whose message the interface shows beside
            the operator's unchanged input.
    """
    match path:
        case "/api/health":
            return {
                "ok": True,
                "schema_version": COMPANION_SCHEMA_VERSION,
                "reads_schema_versions": sorted(READABLE_SCHEMA_VERSIONS),
                "version": shed.__version__,
                "rules": "shed-v1",
            }
        case "/api/agents":
            # Driven by the package, never by a list here: an agent registered in
            # shed.agents appears in the picker without a change to this file.
            return {
                "default": DEFAULT_AGENT,
                "agents": [
                    {
                        "kind": profile.kind,
                        "label": profile.label,
                        "summary": profile.summary,
                        "caveat": profile.caveat,
                    }
                    for profile in agent_catalogue()
                ],
            }
        case "/api/state":
            return _reply(_session_from(payload), payload)
        case "/api/new":
            return _reply(_new_session(payload), payload)
        case "/api/join":
            return _reply(_joined_session(payload), payload)
        case "/api/event":
            session = _session_from(payload)
            if "event" not in payload:
                raise CompanionDataError("request must carry an 'event'")
            current = derive(session)
            if current.error is not None:
                # Appending onto a log that no longer folds would bury the real
                # problem one entry deeper; say so and let Undo or export out.
                raise ObservationError(current.error)
            event = decode_event(payload["event"], "event")
            # Applying against the derived position is what rejects the entry, and
            # it raises the reducer's own message rather than a replay summary.
            apply_event(current.state, event)
            return _reply(session.appended(event), payload)
        case "/api/undo":
            return _reply(_session_from(payload).undone(), payload)
    raise CompanionDataError(f"{path} is not an endpoint")


class CompanionHandler(BaseHTTPRequestHandler):
    """Serves the bundled page and the JSON API over one local connection.

    Attributes:
        server_version: Product token in the ``Server`` header, so a stray request
            in a log is identifiable as this companion.
        protocol_version: HTTP/1.1, which keeps the phone's connection alive between
            the several requests one tap makes. Every reply therefore sets an exact
            ``Content-Length``.
        log_requests: Whether to print a line per request. Off by default: the
            Termux session is the operator's game console, not a web log.
    """

    server_version = "shed-companion"
    protocol_version = "HTTP/1.1"
    log_requests = False

    def do_GET(self) -> None:  # noqa: N802 - the name is BaseHTTPRequestHandler's.
        """Serve a bundled asset, or the health endpoint."""
        path = self.path.split("?", 1)[0]
        if path in ("/api/health", "/api/agents"):
            self._send_json(HTTPStatus.OK, handle_api(path, {}))
            return
        asset = ASSETS.get(path)
        if asset is None:
            self._send_json(
                HTTPStatus.NOT_FOUND, {"error": f"{path} is not served by the companion"}
            )
            return
        name, content_type = asset
        try:
            body = (STATIC_ROOT / name).read_bytes()
        except OSError as error:  # A broken install, not a bad request.
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"cannot read bundled asset {name}: {error}"},
            )
            return
        self._send_bytes(HTTPStatus.OK, body, content_type)

    def do_POST(self) -> None:  # noqa: N802 - the name is BaseHTTPRequestHandler's.
        """Decode one JSON request, dispatch it, and answer.

        Errors are told apart on the way out: a malformed body or an unknown
        endpoint is the caller's mistake, a contradicted observation is the
        operator's to resolve, and anything else is a bug that must not take the
        server down mid-game.
        """
        path = self.path.split("?", 1)[0]
        try:
            payload = self._read_json()
        except CompanionDataError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error), "kind": "data"})
            return
        try:
            self._send_json(HTTPStatus.OK, handle_api(path, payload))
        except CompanionDataError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error), "kind": "data"})
        except ObservationError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error), "kind": "observation"})
        except Exception as error:  # noqa: BLE001 - one bad request must not end the game.
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"unexpected server error: {error}", "kind": "server"},
            )

    def _read_json(self) -> dict[str, Any]:
        """Read and decode the request body.

        Returns:
            The decoded object.

        Raises:
            CompanionDataError: If the length is missing or implausible, the body is
                not valid UTF-8 JSON, or the payload is not a JSON object.
        """
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise CompanionDataError("request needs a Content-Length")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise CompanionDataError("Content-Length is not a number") from error
        if length < 0 or length > MAX_BODY_BYTES:
            raise CompanionDataError(f"request body must be at most {MAX_BODY_BYTES} bytes")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CompanionDataError(f"request body is not valid JSON: {error}") from error
        if not isinstance(payload, dict):
            raise CompanionDataError("request body must be a JSON object")
        return payload

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        """Send one JSON reply.

        Args:
            status: The status code.
            payload: The object to encode.
        """
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        """Send one reply with an exact length and no caching.

        ``no-store`` is deliberate on the assets too: an updated checkout should take
        effect on the next reload, and a stale script cached against a changed API is
        the one failure the operator cannot diagnose from the phone.

        Args:
            status: The status code.
            body: The bytes to send.
            content_type: The media type.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base signature.
        """Print a request line only when asked to.

        Args:
            format: Printf-style format from the base class.
            args: Its arguments.
        """
        if self.log_requests:
            super().log_message(format, *args)
