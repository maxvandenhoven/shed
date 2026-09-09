"""Shed: an engine, agents, and gauntlet for the hidden-information card game Shed.

The package is being built in milestones. This module exports project metadata
only; import the engine from :mod:`shed.engine` and the agents from
:mod:`shed.agents`.

Implemented so far: the complete deterministic engine -- the canonical deck,
dealing, authoritative state, immutable player views, the event vocabulary with
private-event filtering, legal-move generation, and atomic setup and play
transitions with snapshot undo -- the agent layer on top of it, and the timed
match runner in :mod:`shed.match`, which owns clocks, worker processes, the
selection policy, and the records a match leaves behind. On top of those,
:mod:`shed.replay` writes and verifies versioned JSON replays and is the
project's only serialization boundary, and :mod:`shed.cli` holds what the
command-line scripts share. The gauntlet described in ``docs/implementation.md``
is not implemented yet.

Importing this package is deliberately cheap. ``__version__`` is resolved on
first access rather than at import time, because reading distribution metadata
pulls in ``importlib.metadata`` and its dependencies -- measured at roughly 60 ms
of the 115 ms it took to import :mod:`shed.match`. The match runner starts a
fresh interpreter per decision, so that import cost was being paid on every
single timed decision to compute a string almost nothing reads. Only the replay
writer needs the version, once per match.

Attributes:
    RULES_PROFILE_ID: Identifier of the fixed rules profile this package targets.
    REPLAY_SCHEMA_VERSION: Version of the JSON replay schema this package targets.
    __version__: Installed distribution version, or a sentinel when the package
        is imported from a source tree without installed metadata. Resolved on
        first access and not cached, which is why nothing hot should read it.
"""

__all__ = ["REPLAY_SCHEMA_VERSION", "RULES_PROFILE_ID", "__version__"]

RULES_PROFILE_ID = "shed-v1"
REPLAY_SCHEMA_VERSION = 1


def __getattr__(name: str) -> str:
    """Resolve a lazily computed package attribute.

    Args:
        name: Attribute being accessed on the package.

    Returns:
        The installed distribution version for ``__version__``, or the sentinel
        ``"0.0.0+unknown"`` when the package is imported from a source tree
        without installed metadata.

    Raises:
        AttributeError: For every other name, so ordinary attribute errors keep
            their usual behaviour.
    """
    if name != "__version__":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("shed")
    except PackageNotFoundError:  # pragma: no cover - only hit outside an install.
        return "0.0.0+unknown"
