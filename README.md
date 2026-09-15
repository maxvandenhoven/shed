# Shed

Game engine, agents, and evaluation tooling for Shed, a hidden-information
card game for 2 to 5 players. The package contains an authoritative engine,
interchangeable agents behind a small interface, a match runner that enforces
a wall-time budget per decision, versioned JSON replays, and a gauntlet that
compares agents on shared deals with rotated seats.

There is also a phone companion for games played with real cards.
`python -m shed.companion` serves an offline page that tracks a physical
two-player game and asks any of the shipped agents what it would play. See
[`docs/companion.md`](docs/companion.md).

[`docs/design.md`](docs/design.md) covers the architecture and the full rules
of the implemented profile. [`docs/agent-baselines.md`](docs/agent-baselines.md)
records what the shipped baselines score and where a stronger agent should
start.

## The rules

Shed is played with 54 uniquely identified cards, the 52 ordinary cards plus
two distinct jokers. Ranks run 2 to ace, ace high. Suits decide nothing,
neither legality nor strength, so a play names a rank and a count rather than
a set of cards, and the engine picks the physical cards by ascending
identifier.

- Setup. Three cards go face down, three face up, and three to the hand.
  Every player then privately picks which three of their six hand and face-up
  cards end up face up. Keeping the deal as dealt is legal. Submissions stay
  private and are committed together, so nobody can react to a choice while
  still making their own. The opener is found by scanning ranks in the order
  3,4,5,6,7,8,9,10,J,Q,K,A,2,joker (specials last) and taking the first rank
  anybody holds. Ties among its holders break clockwise after the dealer. The
  opening pile is unrestricted, so the opener need not lead that card.
- Zones, in order. You play from your hand. When hand and deck are both
  empty, you play from your face-up cards, and only then from your face-down
  slots, one blind reveal at a time. A revealed card is tested against the
  constraint that stood before the reveal. On a miss you take it and the
  pile, and a failed final reveal does not win.
- A play is one rank and a count from one zone, never a mix of hand and
  table. The pile's constraint is what an ordinary rank must satisfy. An
  empty pile accepts anything, an ordinary rank r requires at least r after
  it, and a seven requires at most seven after it.
- The exceptions. A two is always legal and resets the requirement to "at
  least two". A nine is always legal and leaves the constraint exactly as it
  was, so 7 then 9 still requires at most seven. A joker is always legal and
  clears the constraint. A ten is always legal and burns the pile. Four of a
  rank played in one action burns the pile, but only if that rank was legal
  to begin with. Four accumulating across separate turns does not burn.
- Blocked means pickup, and pickup is never voluntary. When no batch is
  playable, taking the pile is the only legal move.
- After a play the hand refills to three while the deck lasts. Drawing is
  automatic and never a decision. A burn gives the same player another
  decision with a fresh budget, and anything else passes to the next seat.
- Winning is checked after the refill. The first player with no cards
  anywhere wins and the game ends. A final burn wins rather than earning an
  extra turn. There are no eliminations, no last-player-loses rule, no
  passing, and no off-turn responses.

The profile is fixed and identified as `standard`. `RulesConfig` records it
and refuses a changed field or an unknown identifier, so a game can never
quietly claim the standard rules while playing something else.

## The engine

`GameState.apply_move()` is atomic. It validates the move against the current
position, snapshots the state, and resolves the whole chain (transfer or
reveal, burn or rank effect, pickup, replenishment, termination, and the next
actor) before returning. An illegal move mutates nothing, and any failure
inside that boundary rolls the snapshot back. `undo_move()` restores fields
on the existing object, LIFO.

Timing and process management live in `shed.match`, outside the engine and
outside agents. This keeps the engine testable without real clocks and lets
agents be written without knowledge of their execution environment.

`GameState` is the entry point. It carries its own `RulesConfig`, so there is
no separate rules object.

```python
from shed.engine import GameState, PlayerId

state = GameState.create(3, seed=42)  # SETUP, arrangements pending.
moves = state.get_legal_moves()  # The actor's 20 arrangements.
view = state.observe(PlayerId(1))  # Immutable, hides everything private.
view.legal_moves  # The same tuple, for the actor only.
state.apply_move(moves[0])  # Stored privately until all players choose.
```

Once every seat has arranged, the game is in PLAY and the same four
operations carry it to a result. A play names a zone, a rank, and a batch
size, such as `Play(Zone.HAND, Rank.SEVEN, 2)`.

```python
transition = state.apply_move(state.get_legal_moves()[0])
transition.events  # Full events, cards played, burns, draws, the ending.
state.constraint  # What the next ordinary rank must satisfy.
state.current_ply  # 1. PLAY decisions only, setup does not count.
state.undo_move(transition)  # Back to the position before the decision.
state.is_finished, state.outcome  # False, None until somebody sheds everything.
```

## Agents

An agent sees one observation and a turn-scoped submission channel, and
nothing else. It reads its options from `view.legal_moves` (the engine is the
only legality authority) and closes the decision with a final submission.

```python
from shed.agents import AgentSpec, build_agent
from shed.engine import GameState, PlayerId

state = GameState.create(2, seed=42)
view = state.observe(PlayerId(1))  # Seat 1 arranges first when seat 0 deals.
agent = build_agent(AgentSpec(kind="greedy", name="greedy-1"), seed=7)

# `turn` comes from whoever runs the decision, either the match runner's
# pipe-backed context inside a worker or an in-memory fake in a test.
agent.think(view, turn)
```

Agents are built fresh for every decision from a serializable `AgentSpec` and
an explicit seed, so no live object and no generator state is ever reused. A
spec carries a kind and a label, never a deck or fallback seed. `RandomAgent`
samples the legal rank/count actions uniformly. `GreedyAgent` keeps the cards
that are hardest to shed (sevens, nines, twos, jokers, and tens score above
every ordinary rank), sheds the largest batch it can, spends the cheapest
cards among equally sized plays, and settles genuine ties with its seeded
generator.

### What an agent may see

A `PlayerView` is the whole of an agent's input. It carries the rules
profile, the public position, the viewer's own hand, the viewer's filtered
history, and, for the acting seat only, that decision's legal moves. Every
collection in it is a tuple of frozen values, and none of them aliases
anything inside `GameState`.

It never contains another seat's hand, any face-down identity, the deck
order, the shuffle seed, any RNG state, another player's pending setup
submission, or the replay being recorded. History preserves what was once
visible, so a card seen face up before the arrangements is still in the
record after it moves, and a private event reaches everyone but its recipient
as a count with `cards=None`. Swapping two face-down cards between seats
cannot change what any viewer observes or what moves they are offered. The
engine tests assert exactly that.

There is no agent memory between decisions. A worker is created for one
decision, builds the agent, and exits. Nothing an agent stores on `self`
survives, and nothing it writes reaches the parent. The filtered history in
each view is how an agent reconstructs what it knew.

## Timed matches

`MatchRunner` puts a clock around a decision. Each decision gets one freshly
spawned worker, one fresh pipe, and one fresh agent seed. The actor's
`legal_moves` are frozen as the acceptance set before the worker starts, and
exactly one move is applied afterwards, which the engine revalidates
independently.

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
result.turns[0].reason  # Why that decision closed: final, returned, deadline, failed.
result.decisions[-1].events  # The events the last applied decision resolved into.
```

An agent may submit as often as it likes. The runner keeps the latest legal
candidate, closes early on a legal final one, and otherwise closes at the
deadline. A decision that reaches the deadline holding a legal candidate is
an ordinary decision. Only a decision that accepts nothing at all uses the
seeded legal fallback, and rejected submissions, worker crashes, and
fallbacks are all counted in the turn record. `MatchConfig(strict_failures=True)`
aborts the match on any of them instead of playing on.

The budget is an acceptance deadline. Worker startup counts against it, and
scheduling and reaping add latency after it, so `choose_move()` does not
return at exactly that instant. The parent enforces the deadline itself
(terminate, then kill, and always reap the worker), because polling inside an
agent cannot interrupt an infinite loop. A worker receives only its
specification, a fresh seed, one observation, and its deadline. The supported
threat model is trusted local agents sending small, well-formed messages.
Submissions are validated and illegal ones rejected, but this is not a
sandbox for hostile code.

No worker is ever forked from the runner, because that would hand it a copy
of the parent's memory, where the authoritative state lives. Workers come
from a `forkserver` where the platform has one, and from `spawn` otherwise.
The forkserver keeps the same boundary, since its server process is created
by fork and immediate exec of a fresh interpreter and so holds none of the
runner's objects, while cutting worker startup from about 120 ms to about
14 ms. A test asserts the boundary directly.

## Replays

A finished match is written as versioned JSON and read back through the only
serialization boundary the project has. `shed.replay` holds every encoder and
decoder. The engine, the agents, and the runner never see JSON, and nothing
they own imports this module.

```python
from pathlib import Path

from shed.replay import read_replay, verify_replay, write_match

write_match(result, Path("results/match.json"))
replay = read_replay(Path("results/match.json"))  # Validates before it decodes.
check = verify_replay(replay)
check.ok, check.applied, check.outcome  # True, 58, Outcome(winner=1)
```

Encoding takes typed records and writes explicit tags. A move is
`{"type": "play", "source": "hand", "rank": 7, "count": 2}`, never a pickled
object. Decoding takes a document that merely claims to be a replay and
settles every external question before a domain object exists, checking the
schema and rules profile it targets, the shape of each record, the tag of
each union, and the primitive type of each field. A JSON boolean is refused
where an integer is expected, even though Python treats `True` as `1`.

Verification replays the recording. It deals the recorded deck order with the
same pure helper the runner deals with, applies each recorded move through
`GameState.apply_move()`, and compares the resolved events, the outcome, and
a digest of the final position. No agent is built and no worker is started,
so a replay is deterministic even though the timed match that produced it was
not. A move the engine now refuses, a tampered event stream, or a position
that does not match come back as reported problems, never as an exception.
Truncated and aborted matches replay too. A selection that was chosen but
never applied stays out of the applied stream, so a replay can never play a
move the match did not.

A replay reproduces the recording, and only that. Rerunning the same match
from the same seeds is not expected to produce the same game, because the
agents think against a wall clock and a different candidate can be the latest
one when the deadline arrives. Timings are recorded as diagnostics and are
never compared during verification. A test that needs bit-for-bit
repeatability of the decisions can drive the baselines synchronously with a
fake turn context, which removes the clock from the loop entirely.

A complete replay file holds hidden information (every face-down identity,
the deck order, and each private draw) and is a trusted post-match artifact.
The console output is the opposite. `describe_event` reports private events
by their public count, so summarizing a replay never dumps what the players
could not see.

## The gauntlet

`shed.gauntlet` compares agents by playing many matches. It schedules, seeds,
and counts. Every authoritative state still lives inside a `MatchRunner`, and
no rule is reimplemented there.

```python
from shed.cli import build_participants
from shed.gauntlet import GauntletConfig, build_schedule, run_gauntlet

agents = build_participants(["random", "greedy"])  # random-0, greedy-1
config = GauntletConfig(deals=10, seed=42, seconds_per_turn=0.5)

build_schedule(agents, config)  # Pure: 20 entries, seeds included.
run = run_gauntlet(agents, config)
run.report.status.finished  # 20. Truncated and failed matches are counted apart.
run.report.agents[1].wins  # Rate(count=16, total=20)
```

Each deal in the bank is played once per cyclic seat rotation, so a lineup of
two agents and 10 deals is 20 matches with the same deck and the same dealer
and the participants shifted one seat. Every participant gets every seat on
every deal, which makes the seat breakdown a fair comparison. It is not every
seating permutation for three or more agents, since the participants keep
their cyclic order relative to each other, so it controls for seat advantage
and not for who sits to whose left.

Matches are played one at a time. The budget an agent is given is wall time,
so two matches thinking at once would measure the machine's load rather than
the strategies.

Seeds come from `derive_seed`, a SHA-256 over a canonical JSON payload rather
than Python's per-interpreter `hash()`, so a schedule reproduces in a fresh
process. The deck, agent, and fallback streams are derived under separate
purposes. Rotations of one deal share the deck seed and nothing else, and no
agent's seed is a function of the deal it is playing.

Finished, truncated, agent-failed, and engine-failed matches are counted
separately and must add up to the number scheduled, or `summarize` refuses
the run. Win rates are measured over finished matches only, so a truncated
match is never quietly a loss, and every rate carries the denominator it came
from, including `0/0`, which reports as `n/a` rather than as a zero win rate.

## Requirements

- [uv](https://docs.astral.sh/uv/), which manages the Python 3.12 toolchain,
  the virtual environment, and the committed `uv.lock`.

## Install

```bash
git clone https://github.com/maxvandenhoven/shed
cd shed
uv sync            # create .venv and install the project with dev tools
uv sync --locked   # CI: install exactly the committed lockfile
```

`uv sync` creates `.venv`, installs the pinned Python 3.12 toolchain, and
installs `shed` in editable mode with `pytest`, `ruff`, and `ty`. There are
no runtime dependencies, the package needs only the standard library, so
installing the built wheel on its own is enough to use the library.

```bash
uv build                                  # dist/shed-0.1.0-py3-none-any.whl
uv run --isolated --no-project --python 3.12 \
    --with dist/shed-0.1.0-py3-none-any.whl \
    python -c "from shed.engine import GameState; print(GameState.create(2, seed=1).phase)"
```

`--python 3.12` is required there. The wheel declares `requires-python
>=3.12`, and outside the project directory `uv` would otherwise resolve
against whatever interpreter it finds first.

Every command below is written for a checkout, because the scripts under
`scripts/` are part of the repository rather than console entry points.

## Development

```bash
uv run ruff format .        # format
uv run ruff check . --fix   # lint and order imports
uv run ty check             # type check
uv run pytest               # tests
uv build                    # build the sdist and wheel
```

CI runs the same commands with `ruff format --check .` in place of
`ruff format .` and without `--fix`. See [Contributing](#contributing).

## Layout

| Path | Contents |
| --- | --- |
| `src/shed/` | The `shed` package |
| `src/shed/engine/` | Value types, events, and state with the game operations |
| `src/shed/agents/` | Agent interface, specification, factory, and the baselines |
| `src/shed/match.py` | Timed decisions: selection policy, worker, pipe context, runner |
| `src/shed/gauntlet.py` | Sequential evaluation: schedules, seeds, accounting, report file |
| `src/shed/replay.py` | Versioned JSON replay: the whole codec, and verification |
| `src/shed/benchmark.py` | Engine-only benchmark: fixtures, measurements, report |
| `src/shed/cli.py` | Shared command-line logic: lineups, narration, summaries |
| `src/shed/companion/` | Offline phone companion: observed state, agent adapter, local server, bundled page |
| `tests/` | pytest suite |
| `scripts/` | Command-line entry points: `play.py`, `replay.py`, `gauntlet.py`, `benchmark.py` |
| `docs/design.md` | Architecture and the full rules of the profile |
| `docs/agent-baselines.md` | What the shipped baselines score, and why |
| `docs/companion.md` | Installing, running, and using the phone companion |
| `results/` | Generated local outputs, ignored by Git |

## Commands

Play one match and save its replay, then verify that the replay reproduces
it:

```bash
uv run scripts/play.py --agents random greedy --seed 42 --seconds-per-turn 2 --output results/match.json
uv run scripts/replay.py results/match.json --verify
```

`play.py` prints the public actions and the outcome, and exits nonzero only
when a match aborts. A truncated match is a limit being reached, and is
reported as such. `--quiet` drops the action log, `--strict-failures` aborts
on any agent failure, and `--dealer`, `--max-play-decisions`, `--agent-seed`,
and `--fallback-seed` expose the rest of the runner's configuration.
`replay.py` summarizes a saved file, adds the recorded actions with
`--events`, and with `--verify` replays it. It exits `1` when verification
fails and `2` when the file cannot be read or is not a replay this version
supports.

### Watching a match with full information

Both commands take `--omniscient`, which prints what the players could not
see, meaning the dealt hands, every replenishment draw, and the position the
match stopped in, face-down slots and undrawn deck included.

```
$ uv run scripts/play.py --agents random greedy --seed 42 --max-play-decisions 10 --omniscient
player 0 deals to 2 seats; player 0 shows 6 7 10, player 1 shows 3 8 10
player 1 is dealt 3 cards: 4 5 K
player 0 is dealt 3 cards: 2 3 A
player 1 settles on 8 10 K face up
player 1 plays 3 from hand
player 1 draws 1 card: K
…
player 0 plays 10 from hand
player 0 burns 8 cards (ten)

position: play | ply 10 | to act: player 0 | constraint: at least 5
  draw pile (26, next draw last): 2 10 7 6 8 J 5 7 4 J K Q 5 Q 9 5 9 8 6 JK Q Q J 4 K 2
  discard (2): 7 5
  burned (8): 2 3 4 6 8 10 K A
  player 0: hand 9 A JK | face up 3 7 A | face down 0=9 1=10 2=4
  player 1: hand 2 J A | face up 8 10 K | face down 0=3 1=3 2=6
```

This is the view for reading back why an agent played what it did. It is
orthogonal to `--quiet`, so together they print the position and the summary
and no action log. It is also purely a console setting. Events always carry
their identities, a runner records them, and a replay file stores them, so
the public view is redaction applied on the way to the terminal, and
`--omniscient` simply declines to apply it. Nothing about what an agent is
given changes. A strategy sees a `PlayerView`, which never contains another
seat's cards whatever the operator asked to print.

Cards are spelled by rank alone, because a suit decides nothing in this
profile and `J J 7` reads better than `Jc Jd 7h`. Both commands take
`--show-suit` when you do want them, for example to track one physical card
through a pickup. The suits are in the replay file either way. The flag only
changes the spelling on your terminal.

Every group of cards is printed in reading order, by rank with the card
identifier breaking ties, so the output stays deterministic and keeps one
rank's suits together. The two exceptions are the discard and draw piles,
which are printed exactly as stored, because position decides what happens
next in both. Sorting is presentation only. The engine's orders are untouched
and a replay still compares them exactly.

### Running a gauntlet

```bash
uv run scripts/gauntlet.py --agents random greedy --deals 10 --seed 42 --seconds-per-turn 0.5 --output results/gauntlet.json
```

The lineup holds two to five agents and repeated kinds get distinct labels,
so `--agents greedy greedy` is a readable mirror match. `--deals` sizes the
bank, `--dealer`, `--max-play-decisions`, and `--strict-failures` reach the
rest of the runner's configuration, and `--quiet` silences the per-match
progress, which otherwise goes to standard error while the table goes to
standard output. The command exits `2` on arguments that describe no runnable
gauntlet and `1` when a match aborted or the report could not be written.

```
$ uv run scripts/gauntlet.py --agents random greedy --deals 10 --seed 42 --seconds-per-turn 0.5 --quiet
standard | 2 agents | 10 deals x 2 rotations = 20 matches | seed 42 | dealer 0 | 0.5s per decision
status: 20 finished | 0 truncated | 0 agent failed | 0 engine failed (of 20 scheduled)

wins among finished matches:
agent       kind           wins        seat 0        seat 1
random-0  random   4/20 (0.200)  2/10 (0.200)  2/10 (0.200)
greedy-1  greedy  16/20 (0.800)  8/10 (0.800)  8/10 (0.800)

random-0 against greedy-1: 4/20 (0.200)
greedy-1 against random-0: 16/20 (0.800)

decisions:
agent     decisions  fallbacks  rejected  crashes   mean  median
random-0       1271          0         0        0  37 ms   36 ms
greedy-1       1298          0         0        0  37 ms   36 ms
all            2569          0         0        0  37 ms   36 ms

play decisions per match: mean 126.5 | median 107.5
```

Those numbers are one 20-match run against a random baseline, not a
benchmark of the heuristic. Rotations share deals, so results across them are
correlated. Treating each rotation as an independent sample would understate
the uncertainty.

`--output` writes the same numbers as JSON and embeds each match's full
replay document, the same format `play.py` writes, so a finished match from a
gauntlet file verifies exactly like a saved single match.

```python
import json
from pathlib import Path

from shed.replay import decode_replay, verify_replay

document = json.loads(Path("results/gauntlet.json").read_text(encoding="utf-8"))
for entry in document["matches"]:
    if entry["status"] == "finished":
        assert verify_replay(decode_replay(entry["replay"])).ok
```

That costs size, roughly 180 KB per match, so `--no-replays` drops the
embedded documents when only the aggregate is wanted. What is left keeps each
match's schedule entry, seeds, status, and winner, but can no longer be
replayed.

### Measuring the engine

```bash
uv run scripts/benchmark.py --iterations 10000
uv run scripts/benchmark.py --transitions 500 --playouts 50 --output results/benchmark.json
```

The benchmark measures the engine and nothing else. `shed.benchmark` does not
import `shed.match`, and a test asserts that by reading its imports, so no
worker startup, no transport, and no timed wait can end up inside a reported
number. Four costs are reported separately, because they are charged
separately.

| Measurement | One operation | Why it is on its own |
| --- | --- | --- |
| legal moves | `state.get_legal_moves()` | No view is built, so this is grouping and rank comparison alone |
| transitions | `state.apply_move()` then `state.undo_move()` | The pair a search spends; both ends copy the whole position |
| observations | `state.observe()` | Includes generating the acting seat's legal moves |
| playouts | one complete random game, and one decision of one | Engine-only throughput: deal, legality, apply |

The observation rows are printed under the legality rows. `observe()` fills
`PlayerView.legal_moves` for the acting seat on every call, so an observation
is a legal-move generation plus a snapshot of the public position, and the
two tables together show how much of it is which. History is passed in
already filtered and stored by reference, so its length does not drive that
cost. Filtering it is the runner's work.

Fixtures are discovered rather than hand-written. Seeded games are played and
the first position of each shape is copied out, covering the opening
arrangement, the first play, a hand grown past the refill target by a pickup,
a forced pickup, a burn, the face-up collection, and a blind reveal. The same
`--seed` finds the same positions, and their sizes are printed before any
duration, because a timing without the position it was measured on says
nothing.

```
$ uv run scripts/benchmark.py --iterations 2000 --transitions 100 --playouts 10
standard engine benchmark | CPython 3.12.3 on Linux 6.18.44-fc-v24 (x86_64) | shed 0.1.0
seed 0 | 3 players | 3 repeats | per repeat: 2,000 calls, 100 apply/undo pairs, 10 playouts

fixtures:
fixture     phase  hand  face up  face down  draw  discard  moves  history
setup       setup     3        3          3    27        0     20        4
opening      play     3        3          3    27        0      3        7
grown-hand   play    12        3          3    16        2      4       30
pickup       play     3        3          3    18        9      1       25
burn         play     3        3          3     9        4      3       51
face-up      play     0        3          3     0       10      3       87
face-down    play     0        0          3     0        2      3      106

legal moves (get_legal_moves; no view is built):
fixture     calls  per call  calls/s  spread
setup       2,000  25.57 us   39,114    1.23
opening     2,000   6.28 us  159,316    1.01
grown-hand  2,000   9.89 us  101,150    1.03
…
```

`--iterations` sizes the per-call measurements, `--transitions` the
apply/undo pairs, `--playouts` the complete games, and `--repeats` how many
times each runs. `--players` and `--seed` choose the games. `--output` writes
the same report as JSON, with the raw per-repeat totals alongside the derived
figures.

Every printed duration is the fastest repeat, which is the one least
contaminated by scheduling noise, and the `spread` column is the slowest
divided by the fastest, so a busy machine is visible rather than averaged
away. These numbers describe one machine and one interpreter, and no test
asserts a duration.

One result worth knowing before optimizing anything is that snapshot undo
dominates, by about two orders of magnitude. Generating legal moves costs 2
to 26 µs depending on the position, building an observation 11 to 35 µs, and
an apply/undo pair about 1 ms, almost all of it the two deep copies that make
undo work. A future search should look there first. Those figures are from
one machine, so run it on yours.

## The phone companion

For a game played with real cards across a table:

```bash
python -m shed.companion      # then open http://127.0.0.1:8000 in Chrome
```

It serves one page from this repository, with no CDN, no remote font, and no
request off the device, that tracks a physical two-player game and shows what
`GreedyAgent` would play. It is built for Termux on Android, but it runs
anywhere Python 3.12 does. [`docs/companion.md`](docs/companion.md) is the
full walkthrough, including the Termux install and the Android battery
settings that keep it running.

The central constraint is that the companion does not know the cards. The
engine deals, so it knows every hidden assignment. Across a real table nobody
does. `shed.companion.observed` therefore models what a player can actually
observe, ranks that were seen and counts for everything else, and never
converts a count into an identity. Your opponent's hand is a number plus
whatever a public transfer proved. Face-down cards are a tally. A pile card
you never saw stays `None`. When something needed is missing, the companion
names the observation to record and withholds the recommendation rather than
guessing at it.

Two things keep that honest. The rules are not restated. Legality, the
constraint transition, the burn rule, batch generation, and the active-zone
ordering are imported from `shed.engine` as `can_play_rank`,
`constraint_after`, `burn_reason`, `legal_batches`, and
`active_zone_for_counts`, each of which takes counts and ranks rather than a
`GameState`. And the agent is given a `PlayerView` built strictly from
observations. No unobserved card becomes a `Card`, and the identifiers and
single suit such a view carries are bookkeeping the type demands, never a
claim about a physical card. The agent is called directly in-process, since
a phone asking a one-pass heuristic for a hint does not need the timed
multiprocessing runner.

The game itself lives in the browser as a versioned document, the position
you entered plus an ordered log of observations. The server folds that
document and holds nothing, which is why stopping Python mid-game costs
nothing. The page shows a reconnect banner, keeps the game and anything
half-typed, and recovers by itself when the server comes back. Undo is the
log minus its last entry, corrections are recorded entries rather than
silent rewrites, and export is the document written out.

Any agent the package builds can advise, chosen before the game starts and
recorded in the document. The picker is driven by `AGENT_KINDS` rather than a
list in the page, so a new agent shows up on the phone as soon as it is
registered. The heading, the reasoning, and the caveat are all that agent's
own, and a strategy the companion ships no description for says so rather
than borrowing greedy's.

That every agent can advise is checked, not assumed. Both baselines read only
`legal_moves`, `hand`, `me.face_up` and `viewer`, all of which a
companion-built view fills as truthfully as the engine does. The fields it
cannot fill are named in `FAITHFUL_VIEW_FIELDS`, and a test traces every
shipped agent's field accesses and fails if one reaches outside that set. A
future agent that starts reading the pile therefore finds out, instead of
quietly getting a thinner truth.

The advice is still a baseline and is labelled as one on screen. Greedy is
one pass over the legal moves, biggest batch first, then the rank it least
wants to keep. No card counting, no lookahead, no opponent model, and no win
probability.

## Contributing

Run the same gates CI runs, from a clean checkout:

```bash
uv sync --locked            # install exactly the committed lockfile
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
uv build
```

`uv.lock` is generated. Change dependencies with `uv add` / `uv remove` and
let uv rewrite it. Never edit it by hand.

Every class, function, method, property, private helper, script, and test
carries a Google-style docstring, and Ruff's `D` rules with
`convention = "google"` are part of the lint gate. The lint checks that a
docstring is present and well formed. Describing contracts, units, side
effects, and information boundaries is the actual requirement. Do not
blanket-disable the checks, and do not silence a type error or a missing
implementation with `noqa` or an ignore comment.

The suite takes about a minute. Most of it is the process tests, which start
real workers and wait on real deadlines rather than faking either. The
selection policy is also tested against a fake clock and a fake transport, so
a change to it can be checked in milliseconds before the slow tests confirm
it end to end. Nothing asserts a duration, because a timing threshold fails
on a loaded machine and says nothing about the code.

Everything the commands write goes to `results/`, which Git ignores.

## License

MIT. See [`LICENSE`](LICENSE).
