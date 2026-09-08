"""Smoke tests for the installed ``shed`` package."""

import shed


def test_package_imports_with_declared_metadata() -> None:
    """The installed package imports and exposes its profile metadata.

    This guards the packaging setup: the test relies on ``shed`` being
    importable from the environment, never on ``sys.path`` manipulation.
    """
    assert shed.RULES_PROFILE_ID == "shed-v1"
    assert shed.REPLAY_SCHEMA_VERSION == 1
    assert shed.__version__


def test_public_exports_are_intentional() -> None:
    """Every name in ``__all__`` exists on the package."""
    for name in shed.__all__:
        assert hasattr(shed, name), name
