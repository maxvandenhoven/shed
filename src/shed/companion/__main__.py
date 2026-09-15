"""Entry point for ``python -m shed.companion``.

Kept to one line of work so the launcher stays importable and testable: the
argument parsing, the bind, and the shutdown all live in
:mod:`shed.companion.server`.
"""

from shed.companion.server import main

if __name__ == "__main__":
    raise SystemExit(main())
