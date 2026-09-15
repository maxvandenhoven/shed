# Design

Architecture and rules reference for the `shed` package. The README covers
installation and the commands. This document covers how the pieces fit
together and the exact rules the engine implements.

## Architecture

| Piece | Owns | Main consumer |
| --- | --- | --- |
| `GameState` | All physical cards, hidden assignments, deck order, phase, actor; creation, observation, legality, resolution, undo, outcome | Runner, trusted simulations |
| `PlayerView` | Immutable information available to one player, including its legal moves | Agents |
| `Move` | One decision: arrange, play, reveal, or pick up | `GameState` and runner |
| `Agent` | Strategy; submits candidate decisions | Runner's worker process |
| `TurnContext` | Agent-facing submission channel and remaining-time query | Agent |
| `MatchRunner` | Timing, worker lifecycle, selection, fallbacks, history, replay | Scripts and gauntlet |
| Gauntlet | Match schedules, independent seeds, seat rotations, statistics | Evaluation scripts |

Flow: the gauntlet schedules a match, the runner builds a player view, the
agent submits candidates, the runner selects one, the engine resolves it, and
the gauntlet records the result.

Dependencies point inward. The engine depends only on the standard library.
Agents depend on engine types. The match runner depends on both, and the
gauntlet depends on the match runner. Engine code never imports agents,
timing, scripts, or multiprocessing.

Inside the engine the direction is `types` to `events` to `state`. `types.py`
defines the value types, moves, configuration, and errors. `events.py`
records what happened using them. `state.py` builds state, observations, and
the game operations on both. Engine modules import from the defining module
rather than from the re-exporting `shed/engine/__init__.py`. There is no
separate rules object. `GameState` carries its `RulesConfig`, so a second
instance holding the same fixed profile would only add ceremony.

The full import order is `engine.types` to `engine.events` to
`engine.state`, then `agents.base` to the strategies to `agents.factory`,
then `match` to `replay` to `gauntlet`, each depending on the layers before
it, with `cli` last, above all of them. `benchmark` sits outside that stack
and depends on the engine alone. It does not import `match`, so no
measurement can include worker startup, and `cli` imports it only to render
its report. Two function-local imports exist in shipped code.
`shed/__init__.py` defers `importlib.metadata` so importing the package stays
cheap in a per-decision worker, and `engine/events.py` imports `GameState`
only under `TYPE_CHECKING`.

## The rules profile

The package implements one fixed, versioned profile for 2 to 5 players,
identified as `standard`. `RulesConfig` records it and rejects a changed
field value or an unknown identifier rather than silently claiming the
standard rules. Every public entry point that consumes a profile, deck
construction included, validates it. New profiles should only be added
alongside explicit rules and tests.

### Deck and setup

1. The deck is 52 ordinary cards and two distinct jokers, 54 uniquely
   identified physical cards.
2. Ordinary ranks are 2 through ace, ace high. Suits affect neither strategy
   nor legality.
3. Seats are ordered `0..player_count-1`. The default dealer is seat `0`,
   and an explicit valid dealer argument is supported.
4. The unshuffled deck is built deterministically. Suits run clubs,
   diamonds, hearts, spades, with ranks ascending 2 to ace within each suit,
   then two jokers. IDs `0..53` are assigned before shuffling.
5. The deck is shuffled once with a dedicated `random.Random(seed)`. The end
   of the deck list is the next card to draw.
6. Dealing goes clockwise beginning after the dealer. Three rounds to
   face-down slots `0,1,2`, then three rounds face up, then three rounds to
   hands, one card per seat per round.
7. All remaining cards form the draw pile. The game starts with an empty
   discard pile, an empty burned collection, and an unrestricted constraint.
8. Every player privately chooses exactly three physical cards from their
   six hand and face-up cards to become their final face-up cards. The other
   three become their hand. Keeping the original arrangement is legal.
9. Arrangements are requested in clockwise order after the dealer.
   Submissions are stored privately without being applied, and all
   arrangements are committed together only after everyone has chosen.
10. The opening player is selected from post-arrangement hands by searching
    ranks in the order 3,4,5,6,7,8,9,10,J,Q,K,A,2,joker. For the first rank
    present, player ties break clockwise after the dealer.
11. No particular opening card is forced. The opening pile is unrestricted.

Face-down card identities are unknown even to their owner. Face-down slot
IDs remain stable when other slots are removed. Face-up cards are a
collection and do not block the specific face-down slot underneath them.

### Active zone and decisions

| Condition, checked in order | Available decision source |
| --- | --- |
| Hand nonempty | Hand |
| Hand empty but draw pile nonempty | Engine refills before requesting a decision |
| Hand and deck empty, face-up cards remain | Face-up collection |
| Hand, deck, and face-up collection empty | Any remaining face-down slot |
| All personal zones empty | Player has finished; no new decision |

For hand or face-up play, a move chooses a rank and a count from one active
zone. A batch contains only that rank, and the engine chooses the physical
cards by ascending card ID. Hand and table cards are never combined in one
move.

If no playable hand or face-up batch exists, `PickUp()` is the only legal
action. Voluntary pickup does not exist in this profile. Pickup transfers the
whole discard pile to the actor's hand, clears the constraint, and passes
play to the next seat. An empty pile is unrestricted, so an ordinary live
card decision cannot legitimately be blocked on an empty pile.

For face-down play, every remaining slot is legal to select. One card is
revealed and its rank tested against the pre-reveal constraint. On success it
resolves as a single-card play. On failure the actor collects both that card
and the old discard pile into their hand, the constraint clears, and play
advances. A failed final reveal does not win.

### Rank legality and effects

| Rank or action | Legality | Successful effect when no burn occurs |
| --- | --- | --- |
| Ordinary rank, including seven | Must satisfy current constraint | Normally `AtLeast(rank)`; seven sets `AtMost(SEVEN)` |
| Two | Always legal | `AtLeast(TWO)` |
| Nine | Always legal | Preserves the previous constraint exactly |
| Ten | Always legal | Burns the entire pile, including the played cards |
| Joker | Always legal | Resets to `Unrestricted()` |
| Four-card batch of one rank | Must first satisfy that rank's legality | Burns the entire pile, including the played cards |

`Unrestricted` accepts any ordinary rank. `AtLeast(r)` accepts ranks greater
than or equal to `r`. `AtMost(r)` accepts ranks less than or equal to `r`.
The always-playable exceptions apply before ordinary comparisons, and a
joker's enum value never determines its game strength.

The seven restriction is represented by the current constraint, never a
countdown of players. A nine preserves it, and an ordinary successful play
replaces it. After 7 then 9 the requirement is still at most 7. After 7 then
5 it is at least 5, and after 7 then 2 it is at least 2.

Only a batch of four played in a single action triggers the four-card burn.
Four equal ranks accumulated across separate actions do not burn. A ten's
burn takes precedence when selecting the recorded burn reason.

### Resolution, replenishment, and termination

After a successful play, burn or constraint changes resolve first, then the
actor's hand refills to three while the deck has cards. Drawing is automatic,
never an agent decision. Pickup and failed reveal finish with the same refill
helper, which never removes cards from a hand already holding at least
three.

A burn moves the entire discard pile to the burned cards, clears the
constraint, and grants the same actor a new decision with a fresh time
budget. Otherwise play advances one seat clockwise.

Whether the actor has zero cards across all personal zones is checked after
replenishment, before scheduling another actor. The first such player wins
and the game ends immediately. A successful final burn wins rather than
granting an extra turn. There are no eliminations, no last-player-loses
rule, no passing, no jokers skipping players, and no chained off-turn
responses.

Potentially cyclic play is bounded by the runner's action limit. Reaching it
is a truncation, never a rules-level draw or win.

## Types

`types.py` defines the identifier newtypes (`PlayerId`, `CardId`, `SlotId`),
the `Suit`, `Rank`, `Zone`, and `Phase` enums, the frozen `Card`, the
constraint union (`Unrestricted | AtLeast | AtMost`), the move union
(`Arrange | Play | Reveal | PickUp`), and `RulesConfig`.

Jokers have no suit and ordinary cards have one, which construction
validates. `Play` accepts only hand or face-up sources and positive counts.
`Arrange` requires three distinct non-negative IDs and normalizes their
order ascending. Legality then checks ownership and availability.

Engine constructors take correctly typed domain objects and assume their
annotations hold. `Play` takes a `Rank`, never an integer it converts into
one. `__post_init__` checks domain invariants only (non-negative identifiers
and slots, positive play counts, playable sources, distinct arrangement IDs,
joker and suit consistency, supported profiles) and never coerces or
runtime-type-checks its inputs. `ty` enforces the annotations for engine
callers, so a decoding failure is a decoder bug rather than something every
constructor re-checks.

Serialization belongs entirely outside the engine, in the replay and
transport layers. Those layers validate untyped external data, decoded JSON
and agent messages, where it enters the process and construct domain objects
before calling the engine. The engine itself stays free of JSON, transport,
and replay decoding.

All moves and cards are immutable and hashable. Canonicalization ensures
equivalent arrangements compare equal, and rank/count moves already
eliminate suit permutations. Face-up public cards and hand tuples are sorted
by card ID for stable representation. Discard and draw order is meaningful
and preserved.

## State and observations

`GameState` holds the rules, per-player zones, seat order, dealer, phase,
current actor, ply counter, draw pile, discard pile, burned cards, the
constraint, the pending setup submissions, and the outcome.

`current_ply` counts only resolved PLAY decisions, including pickup and
failed reveal, incrementing exactly once per such decision. Setup decisions
do not increment it. The runner separately numbers all decisions, including
setup and extra turns.

`current_player` is the next arrangement submitter in SETUP, the current
actor in PLAY, and `None` in FINISHED. The setup record exists only in
SETUP, and there is no independently mutable active-zone field.

### Information boundary

`PlayerView` contains no mutable references into `GameState`. Nested exposed
collections are tuples and their elements are frozen. It never exposes
opponents' hidden cards, face-down identities, shuffle seeds, complete
replays, RNG state, deck order, or setup submission contents.

An observation contains public cards and the viewer's current hand. History
preserves previously known information even after a card changes zones.
Agent policy must depend only on the view, the public rules, its
configuration, and its independent random seed.

`observe()` is read-only. It never resolves a refill or mutates the game.
Invalid decision-boundary states raise an invariant error, and automatic
work belongs inside creation or `apply_move()`. Generating the actor's moves
is what performs that check, so an unresolved position is refused whoever
asks to observe it.

### History and events

Growing history lives in the runner, outside `GameState`. The frozen event
types are:

| Event | Fields and visibility |
| --- | --- |
| `GameStarted` | Dealer, seats, initial public player states; public |
| `HandDealt` | Player, count, cards; card identities visible only to recipient |
| `ArrangementCommitted` | Player and final face-up cards; public, emitted only at collective commit |
| `CardsPlayed` | Player, source, selected physical cards; public |
| `CardRevealed` | Player, slot, card, playable flag; public |
| `CardsDrawn` | Player, count, cards; identities visible only to drawing player |
| `PilePickedUp` | Player and transferred cards, including a failed reveal when applicable; public |
| `PileBurned` | Player, cards, reason `TEN` or `FOUR_OF_A_KIND`; public |
| `GameEnded` | Outcome; public |

`ObservedEvent` is the union of these dataclasses. For private events,
public recipients receive `cards=None` while retaining the count. Full
internal events retain identities for replay. A successful reveal is
represented by `CardRevealed` followed by any burn, draw, or end events, and
no `CardsPlayed` is emitted for the same physical transfer.

The initial public events let agents remember cards visible before the
arrangements. Setup choices remain private until the collective commit, so
pending arrangement transitions emit no public commitment events. Each
player's filtered history is passed into their next observation, and only
theirs.

## Game operations

`GameState` is the engine's entry point. `create` builds a seeded game,
`get_legal_moves` generates the actor's options, `observe` builds a view,
`initial_events` returns the deal's events, and `apply_move` and `undo_move`
resolve and revert transitions. `deal_initial_state(deck, ...)` is a
module-level helper so a replay can rebuild the opening position from a
recorded deck order, and `create` is the seeded path to the same function.

`get_legal_moves()` is the single legality implementation. It reads the
actor's own cards, the public constraint, and the size of the draw pile
straight from the state, and builds no observation. The dependency runs one
way, from `observe()` to `get_legal_moves()`.

`PlayerView.legal_moves` is populated by `observe()` with the actor's moves
and left empty for every other viewer. Agents read it directly. The field
describes the state that was observed and grants no authority to mutate a
later one, since `apply_move()` always revalidates against the current
position, so a stale tuple can only produce an `IllegalMoveError`. The view
is the only place an agent reads legality from. The turn context does not
mirror the tuple, so there is one source of truth per decision.

### Generation algorithm

1. Return `()` for FINISHED. A live phase with no scheduled actor is an
   invariant error, and so is any other invalid live decision boundary, such
   as a pending refill or an actor holding no cards at all. A view for a
   non-actor carries `()` because `observe()` gives it none.
2. In SETUP, return all three-card combinations from the actor's six hand
   and face-up cards, with sorted IDs and deterministic combination order,
   which yields 20 arrangements.
3. In PLAY, derive the active zone as described above.
4. For FACE_DOWN, return `Reveal(slot)` for each remaining slot ascending,
   never inspecting or filtering by hidden rank.
5. For HAND or FACE_UP, count cards by rank.
6. For each rank ascending, apply the shared rank legality predicate. If
   playable, emit `Play(source, rank, count)` for every count from 1 through
   the available count.
7. If no such moves exist, return `(PickUp(),)`.

Count and source legality are separate from rank legality. `apply_move()`
validates against current legal moves before any mutation. A stale or
illegal action raises `IllegalMoveError` and leaves state byte-for-byte
equivalent under canonical serialization.

Rank/count actions keep the action space small. Suits do not affect
outcomes, so four cards of one rank produce four strategic actions instead
of fifteen subsets. Physical card identities still exist for arrangement,
observation, transfer, and replay. For an active zone of H cards, grouping
and generating rank/count actions costs O(H + M) with M ≤ H. Deterministic
physical-card selection preserves reproducible state transitions.

### Transitions and undo

Only three phases are stored. Revealing, drawing, burning, and pickup are
internal operations completed atomically inside `apply_move()`.

A setup transition validates and stores the actor's arrangement and removes
that player from the pending list. While submissions remain pending, the
next submitter becomes the actor with no visible change to hands or tables.
Once the last one arrives, every arrangement is applied, all commitment
events are emitted together, the setup record is discarded, the opener is
selected, and the game enters PLAY with `current_ply=0`.

A play transition validates the move, performs the transfer, reveal, or
pickup, resolves any burn or rank effect, refills the actor's hand, checks
the win condition, and schedules the next actor. `current_ply` increments
once per successful PLAY decision, events are emitted in physical resolution
order, and FINISHED always has `current_player=None`.

`Transition` carries the decision, an undo record, and the full resolved
events, which only trusted callers see. The undo record is an independent
deep snapshot captured before mutation. `undo_move` restores fields on the
existing `GameState` object and restores another copy of the snapshot, so
subsequent mutation cannot corrupt the record. Undo is LIFO on the
originating state, with no promise of arbitrary out-of-order undo. If an
internal exception occurs during resolution after mutation has begun, the
snapshot is restored before the error propagates. Undo snapshots are never
serialized into replays or exposed to agents.

### Decision-boundary invariants

- Each of the 54 physical cards occurs in exactly one deck, player, discard,
  or burned zone.
- IDs, suits, and ranks agree with the canonical deck.
- Seats and the current actor are valid, and no pending refill or card
  effect exists.
- Every live actor has at least one legal move.
- An empty discard implies an unrestricted constraint, and stored
  constraints match resolved effects.
- Hidden assignment changes cannot affect a viewer's legal moves when their
  observable information is unchanged.
- SETUP has pending submissions and no prematurely applied arrangements.
- FINISHED has an outcome, no setup state, no actor, and a winner with no
  personal cards.

## Agents

Agents receive immutable observations and a turn-scoped capability, never
the authoritative state or the runner.

```python
class TurnContext(Protocol):
    def remaining_seconds(self) -> float: ...
    def submit(self, move: Move, *, final: bool = False) -> None: ...


class Agent(ABC):
    def __init__(self, *, seed: int) -> None: ...

    @abstractmethod
    def think(self, view: PlayerView, turn: TurnContext) -> None: ...
```

The turn is a channel and nothing more, submission plus a remaining-time
query. Legal choices come from `view.legal_moves`. There is no acceptance
acknowledgement and no parallel `choose_move()` interface. The base class
owns only the agent's dedicated `random.Random`, because every strategy
needs seeded tie-breaking and the factory always supplies a seed.

`submit(move, final=True)` finalizes. Submission is fire and forget, and the
runner remains the legality authority. An agent should submit a cheap legal
baseline immediately, then improve it, checking `remaining_seconds()`
between small work units. The value is a cooperative hint, and the runner
enforces the deadline independently. "Latest" means the latest legal
received submission, regardless of quality, so agents choose when an
improvement deserves submission.

### Baselines

`RandomAgent` samples uniformly from `view.legal_moves` using its dedicated
per-decision RNG and submits final immediately. Uniform rank/count actions
avoid overweighting equivalent suit subsets.

`GreedyAgent` handles every phase. During setup it scores candidate face-up
sets by fixed card retention scores and chooses the largest sum. During play
it prefers the largest count shed, and among equally sized plays it prefers
to spend cards with a lower retention score. Ordinary ranks score their
numeric value, seven scores 17, nine 20, two 21, joker 22, and ten 23. For
blind reveals it chooses a seeded random slot, and it picks up when that is
the only action. Ties are settled with seeded random choice over a
deterministically ordered tied list. This is a baseline heuristic, not a
claim of strong play. [`agent-baselines.md`](agent-baselines.md) measures
it.

Agents are described by a serializable `AgentSpec` holding a kind and a
stable evaluation label, never a deck seed. `build_agent(spec, seed=...)` is
a small explicit factory. It lives in its own module above the strategies so
it can import every built-in agent at module scope. `base` holds the
interface and imports nothing from the package, each strategy imports
`base`, and `factory` imports both. The spec validates its kind against the
built-in list when it is constructed, so a mistyped lineup fails where it is
written rather than inside a worker at decision time.

Baselines can be driven synchronously with a fake in-memory turn context
that captures submissions and dictates the clock. A whole match runs that
way with no production timing code.

## Timed decisions and the match runner

### Selection semantics

| Event | Selection behavior |
| --- | --- |
| Legal candidate before deadline | Replace latest candidate |
| Illegal or malformed candidate | Record rejection; retain previous candidate |
| Legal final candidate before deadline | Select it and close immediately |
| Deadline with accepted candidate | Select latest candidate; ordinary completion |
| Deadline without accepted candidate | Choose seeded random legal fallback |
| Worker returns | Close early using latest candidate or fallback |
| Worker crashes | Close with latest candidate or fallback; record failure |
| Message after deadline or after finalization | Ignore; never change selection |

Returning early intentionally closes the turn, and no further submissions
can arrive. It does not turn an illegal final move into a legal one.

No game mutation occurs during thinking. Legal moves are precomputed from
the decision state once, and exactly one selected move is applied after the
decision closes. An accepted candidate may be used after a crash, but the
crash still counts in the failure metrics.

There is no application-level cap on the number of submissions. Transport
has finite capacity and may exert backpressure. The runner keeps only the
latest legal candidate and counters, never an unbounded archive.

### Process model

One newly spawned process serves each decision. It receives only
`AgentSpec`, a fresh independent agent seed, the `PlayerView` (which already
carries that decision's legal moves), its deadline, and a dedicated send
endpoint. The agent is instantiated in the worker. Live agents, `GameState`,
match results, and replay seeds are never passed.

No worker is ever forked from the runner, which would hand it a copy of the
parent's memory, where the authoritative state lives. Workers come from a
`forkserver` where the platform has one, and from `spawn` otherwise. The
forkserver's server process is created by fork and immediate exec of a fresh
interpreter, so workers fork from a process that never held the game.
Preloading the runner module in the server is what makes per-decision
startup roughly 14 ms rather than 120 ms.

Building a fresh agent with a fresh recorded seed per decision avoids
repeatedly resetting the same RNG state, which copying a live agent into
each process would cause.

There is no persistent agent memory between decisions. Public and private
observed history lets an agent reconstruct knowledge. Persistent search
state would need one persistent worker per seat and an explicit recovery
story.

The default budget is 2 seconds, configurable, strictly positive, and
finite. The deadline begins immediately before `process.start()`, so process
startup, argument transport, and agent construction count against it. This
keeps the model simple but makes it unsuitable for very short budgets.
Initialization failures before the worker starts are infrastructure errors,
never ordinary agent strategy failures.

### Pipes and the turn context

Each decision gets one fresh `multiprocessing.Pipe(duplex=False)`. The
worker uses the sender and the parent keeps the receiver. After spawn, the
parent closes its copy of the sender so EOF can be detected, and the worker
closes its sender in a `finally` block.

The pipe-backed `TurnContext` stores the sender, the monotonic deadline, and
a local closed flag. `remaining_seconds()` clamps the clock difference to
zero. `submit()` sends a small `Submission(move, final)` message unless
locally expired or closed, and sets its closed flag after sending final.
Broken-pipe errors mean the turn is closed and do not crash cleanup.

The worker calls `think()`, then sends `WorkerFinished`. On an ordinary
agent exception it sends a small `WorkerFailed` record with the exception
type and message, then closes. These three tagged frozen message types live
in `match.py`. The parent validates message type, final flag, canonical move
shape, and legal membership.

### Parent selection loop

The parent polls the receiver until the deadline, accepting complete
messages one at a time. Receipt means a complete message has arrived and the
runner checks its monotonic clock before validation. Equality with the
deadline is late. The runner's timestamp governs eligibility, never an
agent-supplied one. Buffered moves not received before expiry do not count,
and the parent never continues draining buffered candidates after expiry to
choose a newer one. The single-threaded parent avoids lock races, and the
selection policy is testable independently with an injected clock.

### Enforcement and cleanup

Polling in the worker cannot stop infinite loops, so the parent
independently stops the worker on deadline or accepted finalization. It
requests termination, joins with a short bounded grace, kills if still
alive, and always reaps the process. Cleanup runs in `finally`, including on
keyboard interruption and validation failure. A pipe never outlives its
decision.

The budget is an acceptance deadline, never a guarantee that `choose_move()`
returns at exactly that instant. Spawn, OS scheduling, message receipt, and
process cleanup add latency. The supported threat model is trusted local
Python agents sending small well-formed messages. Hostile or arbitrary
payloads would need a watchdog, bounded nonblocking transport, and a real
sandbox, which this project does not provide. Agents must not spawn their
own child processes, since terminating one worker does not automatically
kill arbitrary descendants.

### Runner API

`MatchConfig` holds the per-turn budget, the play-decision limit, the
fallback and agent seeds, and the strict-failure flag. `MatchRunner` takes
the seat-to-spec mapping and a config, validates that agent seats are
exactly `0..len(agents)-1`, and `run(deal_seed=..., dealer=...)` plays a
match. It creates the fresh state through `GameState.create`, records the
initialization metadata, reconstructs the initial shuffled deck order for
replay through the shared deal helper, and initializes filtered histories
from `state.initial_events()`.

At each decision the runner checks the action limit, builds the actor's view
with filtered history, selects a move, enforces the strict-failure policy if
enabled, applies the move once, appends the full events and the turn record,
and updates all filtered histories. It returns when the rules finish the
game, a failure aborts it, or the action limit truncates it. The runner's
terminal status is separate from `GameState.phase`, and truncation never
fabricates an `Outcome`.

The default fallback mode continues after agent errors using the latest
accepted candidate, or a uniform random legal action if none exists. In
strict mode, any rejected submission, worker failure, or absence of a valid
submission aborts the match as `AGENT_FAILED` before applying the selected
action. A deadline with a legal candidate is not a failure in either mode.

## Results, replay, and seeds

`TurnRecord` records the decision ID, player, phase, selected move, close
reason, fallback flag, accepted and rejected counts, any worker failure, the
budget, selection and cleanup elapsed times, and the per-decision agent
seed. `MatchStatus` is `FINISHED`, `TRUNCATED`, `AGENT_FAILED`, or
`ENGINE_FAILED`. `MatchResult` holds the status, the outcome or `None`, the
turn records, the initial and per-decision full events, the play decision
count, failure detail, and replay metadata.

Engine and runner infrastructure errors stop a match and are reported
distinctly from an agent exception. Invalid engine states are never
converted into random moves. For strict failure results,
selected-but-unapplied decisions are excluded from the replay's
applied-decision stream.

Deck shuffle, agent decisions, and fallback selection use separate seed
streams, and no module-global randomness exists anywhere. The gauntlet
generates independent streams from its experiment seed with a stable
algorithm, a SHA-256 of a canonical JSON tuple
`(experiment_seed, purpose, match_index, seat, decision_id)` interpreted as
an integer. Python's randomized `hash()` is never used for seeds.

Wall-time search is not bit-for-bit reproducible, since scheduling changes
how many improvements finish. Recorded-move replay is deterministic even
when rerunning the timed agents would choose differently.

### Replay format

Replays are UTF-8 JSON with explicit tags for moves, constraints, and
events. Decoding is where external data is validated. It checks tags, types,
and ranges, then builds the engine's domain objects and hands those to the
engine, which assumes them well typed. Pickle is never used as a replay
format.

Metadata includes the replay schema, rules ID and config, package version,
source revision when available, Python version, player count, dealer,
canonical initial deck order, deal seed, agent specs and seeds, timing and
failure policy, and final match status. Storing the initial deck order
alongside the seed makes replay independent of future shuffle changes.
Neither is exposed to agents.

Applied decisions are stored in order with full resolved events and
selection diagnostics. Verification reconstructs initialization from the
recorded deck order using the deterministic deal helper, applies recorded
actions without invoking agents, and compares events and the final state and
outcome. Unsupported schemas or profiles are rejected clearly, and timing
measurements are never compared during replay.

Complete replay files contain hidden information and are trusted post-match
artifacts. Agent observations contain only filtered history.

## Gauntlet

Matches run sequentially so concurrent agents never compete for CPU during
wall-time comparisons. The lineup holds 2 to 5 agents.

For each seed in the deal bank, the gauntlet creates a fixed deal and
rotates agent assignments through the seats, keeping the same dealer and
deck for each rotation. Cyclic rotations give every participant every seat.
They do not cover every multiplayer seating permutation, since participants
keep their cyclic order relative to each other.

Fresh agent specs and workers are created for each decision, with separate
seeds derived per seat and decision. Deck randomness is unchanged by agent
failures or fallback draws. All match records are preserved, including
failed and truncated matches.

The report covers scheduled, finished, failed, and truncated counts, wins
and win rate per agent among finished matches with explicit denominators,
win rates by seat and opponent, fallback, rejection, and crash counts, and
mean and median selection time. Failed and truncated matches are never
silently excluded or labelled losses. Rotated results share deals and are
correlated, so an uncertainty interval, if one is ever needed, should
resample whole deal blocks rather than treating each rotation as
independent.

## Benchmark

`shed.benchmark` measures legal generation, apply/undo pairs, observation
construction, and engine-only full random playouts separately, excluding
worker startup and timed waiting from every number. It uses
`time.perf_counter()`, reports platform and fixture sizes, and no unit test
asserts a duration.

Two details keep the rows readable:

- Observation includes legality. `observe()` fills `PlayerView.legal_moves`
  for the acting seat on every call, so an observation row is the matching
  legality row plus the snapshot of the public position around it. The two
  are printed together so the inclusion is visible. History is passed in
  already filtered and stored by reference, so its length does not appear in
  that cost.
- The legality rows build no view at all. They call `get_legal_moves()` on
  the state directly.

Fixtures are discovered from seeded games rather than hand-written, so they
are positions the engine actually reaches and the same seed measures the
same ones. The engine-only claim is structural. The module does not import
`shed.match`, `shed.agents`, or `multiprocessing`, and a test asserts that
by reading its imports.

On one machine and one interpreter (CPython 3.12.3, Linux, three players),
snapshot undo dominates everything else by roughly two orders of magnitude.
Legal-move generation costs 2 to 26 µs across the fixtures, an observation
11 to 35 µs, and an apply/undo pair about 1 ms, most of it the two deep
copies undo needs, since a bare apply inside a playout costs roughly half
the pair. That is where a future search implementation should look first.

## Future search

Search must handle hidden information, more than two players, and extra
turns after a burn, so two-player negamax does not transfer unchanged. A
belief-sampling interface can look like:

```python
class BeliefSampler(Protocol):
    def sample(self, view: PlayerView, rng: random.Random) -> GameState: ...
```

The sample is hypothetical and consistent with observed cards and history,
never the true hidden state. Simulated opponents also receive their own
observations rather than direct access to the sample. Monte Carlo rollouts
are a sensible starting point before information-set search. Hidden
assignment changes must not affect a viewer's observation or legal moves,
and a belief sampler must respect that boundary.

## The phone companion

`src/shed/companion/` tracks a game played with physical cards. Its own
document is [`companion.md`](companion.md). Two of its decisions constrain
the rest of the package and are recorded here for that reason.

First, it imports the profile's individual rules from `shed.engine`
(`can_play_rank`, `constraint_after`, `burn_reason`, `legal_batches`,
`active_zone_for_counts`) instead of restating them. This is why those five
are public and take counts, ranks, and constraints rather than a
`GameState`. A change to any of them is a change to both callers.

Second, it never fabricates a hidden card to satisfy an engine signature. It
builds a `PlayerView` from observed information alone, leaving unobserved
cards out as counts, and calls an agent directly rather than through the
timed runner. That view is honest about the fields listed in
`shed.companion.advice.FAITHFUL_VIEW_FIELDS` and thinner than the engine's
everywhere else. This constrains future agents as much as the companion. A
strategy that comes to depend on `discard_pile`, `burned_cards`, `history`,
`current_ply`, or `dealer` is reading an observed game as though it were an
omniscient one, and `tests/companion/test_agents.py` fails when one starts
to.

The companion depends on `engine` and `agents` and on nothing else in the
package, and on no third-party package at all.

| Path | Responsibility |
| --- | --- |
| `src/shed/companion/__init__.py` | Public companion exports |
| `src/shed/companion/observed.py` | Observed state, its events, and the reducer over engine rules |
| `src/shed/companion/advice.py` | Observed state to `PlayerView`, and the recommendation |
| `src/shed/companion/codec.py` | JSON vocabulary and the decoding boundary |
| `src/shed/companion/session.py` | Versioned session document, replay, and screen rendering |
| `src/shed/companion/api.py` | Stateless JSON endpoints and the asset allowlist |
| `src/shed/companion/server.py` | `python -m shed.companion`: bind, report, shut down |
| `src/shed/companion/static/` | The bundled page, stylesheet, and script |
| `tests/companion/` | Observed-state, adapter, session, and HTTP tests |

## Conventions

- Small functions, dataclasses, tuples, enums, and one concrete rules
  class.
- PascalCase classes, snake_case functions and modules, UPPER_CASE
  constants. All shipped code passes `ty`.
- Frozen, slotted dataclasses for cards, actions, observations, specs, and
  events. Mutable state dataclasses use slots and `default_factory`.
- Explicit move unions and pattern matching. `None` is never overloaded to
  mean pass or pickup.
- Clocks, processes, file I/O, and randomness stay out of legal-move
  generation. Independent RNGs are injected or constructed explicitly, and
  global `random` is never seeded.
- Deterministic iteration and canonical serialized forms everywhere. Set
  iteration never chooses moves or determines card transfer order.
- External and agent input is validated at the decoding and transport
  boundaries where it arrives. Engine code assumes its annotated types and
  checks domain invariants only. Assertions cover internal invariants only.
- `IllegalMoveError(ValueError)` and `StateInvariantError(RuntimeError)` are
  the engine's error types. Agent exceptions are caught only at the worker
  boundary and engine failures at the runner boundary.
- Google-style docstrings on classes, functions, methods, properties,
  private helpers, scripts, and tests, documenting contracts, ownership,
  hidden-information restrictions, and timing semantics.
- `pathlib.Path`, UTF-8, and explicit JSON encoders and decoders for
  artifacts.
- No compact bitfields, custom undo deltas, or shared-memory workers before
  profiling justifies them.
