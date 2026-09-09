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
| Agent interface, `AgentSpec`, and the built-in factory | Implemented |
| Random and greedy baselines across every phase | Implemented |
| Timed match runner: selection policy, spawned workers, records | Implemented |
| Replay format, gauntlet, scripts | Not started |

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

Timing and processes are deliberately outside the engine and outside agents:
they live in `shed.match`, so the engine never learns that a decision was timed
and an agent never learns that it runs in a worker.

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

An agent sees one observation and a turn-scoped submission channel, and
nothing else. It reads its options from `view.legal_moves` — the engine is the
only legality authority — and closes the decision with a final submission:

```python
from shed.agents import AgentSpec, build_agent
from shed.engine import GameState, PlayerId

state = GameState.create(2, seed=42)
view = state.observe(PlayerId(1))  # seat 1 arranges first when seat 0 deals
agent = build_agent(AgentSpec(kind="greedy", name="greedy-1"), seed=7)

# `turn` comes from whoever runs the decision: the match runner's pipe-backed
# context inside a worker, or an in-memory fake in a test. The agent submits
# into it, and turn.submit(move, final=True) closes the decision with that move.
agent.think(view, turn)
```

Agents are built fresh for every decision from a serializable `AgentSpec` and an
explicit seed, so no live object and no generator state is ever reused: a spec
carries a kind and a label, never a deck or fallback seed. `RandomAgent` samples
the legal rank/count actions uniformly. `GreedyAgent` keeps the cards that are
hardest to shed — sevens, nines, twos, jokers, and tens score above every
ordinary rank — sheds the largest batch it can, spends the cheapest cards among
equally sized plays, and settles genuine ties with its seeded generator.

`MatchRunner` puts a clock around that. Each decision gets one freshly spawned
worker, one fresh pipe, and one fresh agent seed; the actor's `legal_moves` are
frozen as the acceptance set before the worker starts; and exactly one move is
applied afterwards, which the engine revalidates independently:

```python
from shed.agents import AgentSpec
from shed.engine import PlayerId
from shed.match import MatchConfig, MatchRunner

runner = MatchRunner(
    {
        PlayerId(0): AgentSpec(kind="greedy", name="greedy-0"),
        PlayerId(1): AgentSpec(kind="random", name="random-1"),
    },
    MatchConfig(seconds_per_turn=2.0),
)
result = runner.run(deal_seed=42)
result.status, result.outcome  # FINISHED, Outcome(winner=...)
result.turns[0].reason  # why that decision closed: final, returned, deadline, failed
result.decisions[-1].events  # the full events the last applied decision resolved into
```

An agent may submit as often as it likes; the runner keeps the latest legal
candidate, closes early on a legal final one, and otherwise closes at the
deadline. A decision that reaches the deadline holding a legal candidate is an
ordinary decision, not a failure. Only a decision that accepts nothing at all
uses the seeded legal fallback, and rejected submissions, worker crashes, and
fallbacks are all counted in the turn record — `MatchConfig(strict_failures=True)`
aborts the match on any of them instead of playing on.

The budget is an *acceptance* deadline: worker startup counts against it, and
scheduling and reaping add latency after it, so `choose_move()` does not return
at exactly that instant. The parent enforces the deadline itself — it
terminates, then kills, and always reaps the worker — because polling inside an
agent cannot interrupt an infinite loop. A worker receives only its
specification, a fresh seed, one observation, and its deadline. The supported
threat model is trusted local agents sending small, well-formed messages:
submissions are validated and illegal ones rejected, but this is not a sandbox
for hostile code.

No worker is ever forked from the runner, because that would hand it a copy of
the parent's memory, where the authoritative state lives. Workers come from a
`forkserver` where the platform has one, and from `spawn` otherwise. The
forkserver keeps the same boundary — its server process is created by fork *and
immediate exec* of a fresh interpreter, so it holds none of the runner's
objects, and workers fork from that server rather than from the runner — while
cutting worker startup from about 120 ms to about 14 ms, because the server has
`shed.match` already imported. A test asserts the boundary directly: a value the
parent assigns after import is visible to a plain `fork` child and invisible to a
real worker.

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
| `src/shed/agents/` | Agent interface, specification, factory, and the baselines |
| `src/shed/match.py` | Timed decisions: selection policy, worker, pipe context, runner |
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
