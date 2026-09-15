from __future__ import annotations

import argparse
import errno
import socket
import sys
from collections.abc import Sequence
from http.server import ThreadingHTTPServer

from shed.companion.api import CompanionHandler

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "build_parser", "build_server", "main", "serve"]

DEFAULT_HOST = "127.0.0.1"
"""Loopback only.

The companion is a personal scorekeeper for the phone it runs on, and it accepts
whatever a client sends. Binding a phone's Wi-Fi interface would put that on the
network, so reaching past loopback has to be asked for explicitly.
"""

DEFAULT_PORT = 8000
"""The port the documentation tells the operator to open."""


def build_server(host: str, port: int, *, log_requests: bool = False) -> ThreadingHTTPServer:
    """Bind the HTTP server.

    Args:
        host: Interface to bind.
        port: TCP port to bind; ``0`` asks the operating system for a free one,
            which is what the tests use.
        log_requests: Whether each request prints a line.

    Returns:
        The bound server, not yet serving.

    Raises:
        OSError: If the address cannot be bound. :func:`serve` turns the
            already-in-use case into a readable message; other causes, a host
            that does not resolve, a privileged port, surface as they are.
    """
    handler = type("_BoundHandler", (CompanionHandler,), {"log_requests": log_requests})
    return ThreadingHTTPServer((host, port), handler)


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    log_requests: bool = False,
) -> int:
    """Run the companion until interrupted.

    Args:
        host: Interface to bind.
        port: TCP port to bind.
        log_requests: Whether each request prints a line.

    Returns:
        The process exit code: ``0`` after a clean stop, ``1`` when the address
        could not be bound.
    """
    try:
        server = build_server(host, port, log_requests=log_requests)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            print(
                f"error: port {port} on {host} is already in use.\n"
                f"       Stop whatever is using it, or start the companion on "
                f"another port:\n"
                f"           python -m shed.companion --port {port + 1}",
                file=sys.stderr,
            )
        else:
            print(f"error: cannot serve on {host}:{port}: {error}", file=sys.stderr)
        return 1

    bound_host, bound_port = server.server_address[:2]
    shown = host if host not in ("", "0.0.0.0") else str(bound_host)  # noqa: S104
    print(f"Shed companion running on http://{shown}:{bound_port}")
    print("Open that address in Chrome on this phone. Press Ctrl-C to stop.")
    if host != DEFAULT_HOST:
        print(
            f"warning: bound to {host}, not {DEFAULT_HOST}. Anything that can reach "
            "this phone on that interface can reach the companion."
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping. Your game is saved in the browser, so it will still be there next time.")
    finally:
        # shutdown() ends the accept loop; server_close() releases the socket, so an
        # immediate restart on the same port works instead of reporting it in use.
        server.shutdown()
        server.server_close()
    return 0


def _port(value: str) -> int:
    """Parse a port number.

    Args:
        value: The command-line argument.

    Returns:
        The port; ``0`` asks the operating system for a free one.

    Raises:
        argparse.ArgumentTypeError: If it is not a port number.
    """
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not a port number") from error
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be between 0 and 65535, got {port}")
    return port


def build_parser() -> argparse.ArgumentParser:
    """Build the launcher's argument parser.

    Returns:
        The parser, with every option the companion accepts.
    """
    parser = argparse.ArgumentParser(
        prog="python -m shed.companion",
        description=(
            "Serve the offline phone companion for tracking a physical Shed game "
            "and asking the greedy agent what it would play."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"interface to bind (default {DEFAULT_HOST}, loopback only)",
    )
    parser.add_argument(
        "--port", type=_port, default=DEFAULT_PORT, help=f"port to bind (default {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--log-requests", action="store_true", help="print a line for every HTTP request"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the companion command.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        The process exit code: ``0`` after a clean stop, ``1`` when the address
        could not be bound.
    """
    args = build_parser().parse_args(argv)
    try:
        socket.getaddrinfo(args.host, args.port)
    except OSError as error:
        print(
            f"error: {args.host!r} is not an address this phone can bind: {error}",
            file=sys.stderr,
        )
        return 1
    return serve(args.host, args.port, log_requests=args.log_requests)
