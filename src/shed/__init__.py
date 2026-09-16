__all__ = ["REPLAY_SCHEMA_VERSION", "RULES_PROFILE_ID", "__version__"]

RULES_PROFILE_ID = "standard"
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
