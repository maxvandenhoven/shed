"""Shed: an engine, agents, and gauntlet for the hidden-information card game Shed.

The package is being built in milestones. Only project metadata is exported so
far; the ``engine``, ``agents``, match runner, replay, and gauntlet modules
described in ``docs/implementation.md`` are not implemented yet.

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
