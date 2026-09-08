# Shed

An engine, interchangeable agents, and an evaluation gauntlet for **Shed**, a
hidden-information card game for 2–5 players.

The full design contract lives in [`docs/implementation.md`](docs/implementation.md).

## Status

The engine is being built in milestones. What exists today:

| Area | State |
| --- | --- |
| Cards, moves, constraints, fixed `shed-v1` profile | Implemented |
| Canonical 54-card deck, deterministic shuffle and deal | Implemented |
| `GameState`, `PlayerView`, and the information boundary | Implemented |
| Events, private-event filtering, transition/undo shape | Implemented |
| Legal-move generation for setup, hand, face-up, face-down, forced pickup | Implemented |
| SETUP transition: private submissions, collective commit, opener choice | Implemented |
| **Ordinary PLAY resolution** — batch transfer, reveals, burns, pickup, refill, termination | **Remaining engine work** |
| Agents, match runner, replay format, gauntlet, scripts | Not started |

The engine works on strictly typed domain objects: constructors take a `Rank`,
not an integer they convert into one, and check domain invariants only — `ty`
enforces the annotations. Validating untyped external data and building these
objects from it is the job of the future replay and transport layers, so the
engine stays free of JSON and decoding.

`Ruleset.apply_move()` therefore resolves arrangements only. A legal PLAY move
is validated and then refused with `NotImplementedError`: the engine never
reports a transition that did not happen. Everything else about a PLAY position
already works, so legality can be inspected on crafted states:

```python
from shed.engine import PlayerId, Ruleset

ruleset = Ruleset()
state = ruleset.create_initial_state(3, seed=42)  # SETUP, arrangements pending
view = ruleset.observe(state, PlayerId(1))  # immutable, hides everything private
moves = view.get_legal_moves()  # the 20 arrangements
ruleset.apply_move(state, moves[0])  # stored privately until all players choose
```

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages the Python 3.12 toolchain, the
  virtual environment, and the committed `uv.lock`)

## Install

```bash
uv sync            # create .venv and install the project with dev tools
uv sync --locked   # CI: install exactly the committed lockfile
```

## Development

```bash
uv run ruff format .        # format
uv run ruff check . --fix   # lint and order imports
uv run ty check             # type check
uv run pytest               # tests
uv build                    # build the sdist and wheel
```

The review gates are the same commands with `ruff format --check .` in place of
`ruff format .` and without `--fix`.

## Layout

| Path | Contents |
| --- | --- |
| `src/shed/` | The `shed` package |
| `src/shed/engine/` | Types, state and views, events, rules and `Ruleset` |
| `tests/` | pytest suite |
| `scripts/` | Command-line entry points (none yet) |
| `docs/implementation.md` | Implementation specification |
| `results/` | Generated local outputs, ignored by Git |

## Planned commands (not implemented)

The specification defines these interfaces for later milestones. They do not
exist yet and will fail if run:

```bash
uv run scripts/play.py --agents random greedy --seed 42 --output results/match.json
uv run scripts/gauntlet.py --agents random greedy --deals 100 --seed 42 --output results/gauntlet.json
uv run scripts/replay.py results/match.json --verify
uv run scripts/benchmark.py --iterations 10000
```

## License

MIT. See [`LICENSE`](LICENSE).
