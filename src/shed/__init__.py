"""Shed: an engine, agents, and gauntlet for the hidden-information card game Shed.

The package is being built in milestones. This module exports project metadata
only; import the engine from :mod:`shed.engine`.

Implemented so far: the canonical deck, deterministic dealing, authoritative
state, immutable player views, the event vocabulary with private-event
filtering, legal-move generation, and the SETUP transition with snapshot undo.
Ordinary PLAY resolution is the remaining engine work, and the ``agents``
subpackage, match runner, replay format, and gauntlet described in
``docs/implementation.md`` are not implemented yet.

Attributes:
    RULES_PROFILE_ID: Identifier of the fixed rules profile this package targets.
    REPLAY_SCHEMA_VERSION: Version of the JSON replay schema this package targets.
    __version__: Installed distribution version, or a sentinel when the package
        is imported from a source tree without installed metadata.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["REPLAY_SCHEMA_VERSION", "RULES_PROFILE_ID", "__version__"]

RULES_PROFILE_ID = "shed-v1"
REPLAY_SCHEMA_VERSION = 1

try:
    __version__ = version("shed")
except PackageNotFoundError:  # pragma: no cover - only hit outside an install.
    __version__ = "0.0.0+unknown"
