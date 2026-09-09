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
| `GameState` operations, `PlayerView`, and the information boundary | Implemented |
| Events, private-event filtering, transition/undo shape | Implemented |
| Legal-move generation for setup, hand, face-up, face-down, forced pickup | Implemented |
| SETUP transition: private submissions, collective commit, opener choice | Implemented |
| PLAY resolution: batch transfer, reveals, burns, pickup, refill, termination | Implemented |
| Snapshot undo for every transition, including terminal ones | Implemented |
| Agents, match runner, replay format, gauntlet, scripts | Not started |

The engine works on strictly typed domain objects: constructors take a `Rank`,
not an integer they convert into one, and check domain invariants only — `ty`
enforces the annotations. Validating untyped external data and building these
objects from it is the job of the future replay and transport layers, so the
engine stays free of JSON and decoding.

`GameState.apply_move()` is atomic. It validates the move against the position
as it is now, snapshots the state, and resolves the whole chain — transfer or
reveal, burn or rank effect, pickup, replenishment, termination, and the next
actor — before returning. An illegal move mutates nothing, and any failure
inside that boundary, the closing invariant check included, rolls the snapshot
back. `undo_move()` restores fields on the existing object, LIFO.

Timing, processes, and agents are deliberately outside the engine: the tests
drive complete games with a small seeded helper that picks among the engine's
own legal moves.

`GameState` is the entry point — it carries its own `RulesConfig`, so there is
no separate rules object:

```python
from shed.engine import GameState, PlayerId

state = GameState.create(3, seed=42)  # SETUP, arrangements pending
moves = state.get_legal_moves()  # the actor's 20 arrangements
view = state.observe(PlayerId(1))  # immutable; hides everything private
view.legal_moves  # the same tuple, for the actor only
state.apply_move(moves[0])  # stored privately until all players choose
```

Once every seat has arranged, the game is in PLAY and the same four operations
carry it to a result. A play names a zone, a rank, and a batch size — such as
`Play(Zone.HAND, Rank.SEVEN, 2)` — and the engine picks the physical cards by
ascending card ID, so suits never multiply the action space:

```python
transition = state.apply_move(state.get_legal_moves()[0])
transition.events  # full events: cards played, burns, draws, the ending
state.constraint  # what the next ordinary rank must satisfy
state.current_ply  # 1: PLAY decisions only, setup does not count
state.undo_move(transition)  # back to the position before the decision
state.is_finished, state.outcome  # False, None until somebody sheds everything
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
| `src/shed/engine/` | Value types, events, and state with the game operations |
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
