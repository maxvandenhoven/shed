# Baseline agent analysis

Measurements from one gauntlet run, plus a reading of the shipped heuristic.
This is not a benchmark and not a claim about optimal play. It exists so that
work on a stronger agent starts from evidence. It records what the baselines
do, what they score, which parts of the result are skill and which are
structural to the game, and where the cheapest gains are.
[`design.md`](design.md) covers the eventual search interfaces. This document
is about what to do before reaching for search.

## What was measured

Two gauntlets, each 100 deals played once per cyclic seat rotation, 200
matches each, sharing the same root seed, so both matchups played the same
100 deals:

```bash
uv run scripts/gauntlet.py --agents random greedy --deals 100 --seed 42 \
    --seconds-per-turn 0.5 --quiet --no-replays --output results/random-vs-greedy.json
uv run scripts/gauntlet.py --agents greedy greedy --deals 100 --seed 42 \
    --seconds-per-turn 0.5 --quiet --no-replays --output results/greedy-vs-greedy.json
```

No match failed in either run. There were no agent failures, no engine
failures, no rejected submissions, and no worker crashes. The mirror run did
truncate 13 matches, which is a finding rather than a fault and is covered
below. Both baselines submit one final move per decision, so the budget
almost never bound and the selection-time column measures worker startup
rather than thinking.

## Results

Greedy against random, as the command printed it:

```
standard | 2 agents | 100 deals x 2 rotations = 200 matches | seed 42 | dealer 0 | 0.5s per decision
status: 200 finished | 0 truncated | 0 agent failed | 0 engine failed (of 200 scheduled)

wins among finished matches:
agent       kind             wins          seat 0          seat 1
random-0  random   40/200 (0.200)  16/100 (0.160)  24/100 (0.240)
greedy-1  greedy  160/200 (0.800)  76/100 (0.760)  84/100 (0.840)
```

Two things derived from the same report file:

- Match length splits with the result. Median play decisions is 96.5 when
  greedy wins and 114 when it loses, so greedy loses the long games. The
  mirror run below shows what happens when nothing stops a long game.
- Per deal, greedy took both rotations on 63 deals, split 34, and lost both
  on 3. The deal itself rarely decides the matchup. What happens inside a
  particular game does.

The greedy-vs-greedy mirror is a control rather than a comparison. With the
same strategy, the same deals, and rotated seats, it measures how much of a
result is seat and deal luck rather than play. It passed as a control, and
failed as a game:

```
standard | 2 agents | 100 deals x 2 rotations = 200 matches | seed 42 | dealer 0 | 0.5s per decision
status: 187 finished | 13 truncated | 0 agent failed | 0 engine failed (of 200 scheduled)

wins among finished matches:
agent       kind             wins         seat 0         seat 1
greedy-0  greedy   84/187 (0.449)  42/94 (0.447)  42/93 (0.452)
greedy-1  greedy  103/187 (0.551)  51/93 (0.548)  52/94 (0.553)
```

The 84 to 103 split is within noise for a fair coin (z = 1.39, |z| < 1.96),
and the seat breakdown is flat. Each participant scores about the same in
both seats, so the small difference sits with the participant rather than
the seat. Cyclic rotation is therefore not manufacturing a seat advantage,
which is what the control was for.

### The mirror does not terminate

13 of 200 mirror matches hit the 10,000 play-decision limit and were
truncated. The random matchup truncated none, and its longest game was 921
play decisions.

| | random vs greedy | greedy vs greedy |
| --- | --- | --- |
| Truncated | 0 / 200 | 13 / 200 |
| Median play decisions (finished) | 102 | 75 |
| Mean play decisions (finished) | 122.8 | 417.7 |
| Longest finished match | 921 | 9962 |
| Matches over 1000 play decisions | 0 | 28 |

The distribution is bimodal rather than merely heavy-tailed. Mirror games
are shorter than random-matchup games at the median and enormously longer in
the tail. Eight deals produced a truncation and five of them truncated in
both rotations (deals 10, 16, 25, 27 and 44 at seed 42), so the runaway is a
property of the position rather than of a seating or a coin flip. Those
deals make a ready-made regression fixture for a replacement agent.

The likely mechanism, stated as a hypothesis because it has not been
isolated, is that the retention table makes both agents hoard exactly the
cards that end piles. Tens score 23, jokers 22, twos 21 and nines 20, above
every ordinary rank, so both agents spend them last, and a ten is the burn.
Two mirror hoarders keep feeding each other always-playable cheap cards
while neither clears the pile, and once the deck is exhausted the pickups
recirculate the same cards. Against random, the random agent plays its tens
whenever chance says so, piles clear, and games end. If that is right it is
the same defect as "burning is not valued at all" below, and this is direct
evidence for it.

One note on that run's diagnostics. 128 fallbacks appeared across 208506
decisions (0.06%). A fallback means no candidate arrived before the deadline
and the runner played a seeded legal move instead, which is the documented
behaviour, but it does mean about 128 moves in that run were effectively
random. The random matchup had none. The mirror's far longer games and
larger hands make a decision occasionally overrun a 0.5 s budget that
includes worker startup.

## Why greedy wins only ~80% against random

Three separate causes, and only the third is fixable by writing a better
agent.

### 1. The endgame is blind for everybody

The last three cards are face-down and their identity is unknown even to
their owner. Legal-move generation offers every remaining slot and never
filters by hidden rank, so `Reveal` moves are genuinely indistinguishable.
There is nothing to choose between them. Every game ends in a stretch where
skill contributes nothing and a failed reveal collects the whole pile. That
caps any heuristic, and any search, well short of 100%.

The lever that is available is reaching that phase in a better position,
with fewer cards, a small pile, and a favourable constraint.

### 2. A random legal move is often a fine move

Legality in this game is permissive. Twos, nines, tens and jokers are always
playable, and `AtLeast(r)` accepts everything at or above `r`, so a uniform
choice among legal actions is frequently reasonable. The worst thing that
happens in this game is also forced, never chosen. `PickUp()` is the only
legal move when it appears at all, and voluntary pickup does not exist in
this profile. Both agents therefore inherit piles on exactly the same
trigger, and random never blunders into one that greedy avoids.

### 3. The heuristic is shallow

`GreedyAgent` scores each candidate with `(-count, RETENTION_SCORE[rank])`
and takes the minimum. See `_score` in
[`src/shed/agents/greedy.py`](../src/shed/agents/greedy.py). Four
consequences, roughly in order of how much they appear to cost:

- Batch size dominates unconditionally. `-count` is the first key, so the
  agent dumps three kings when one would do, spending three turns' worth of
  safe plays to shed two extra cards. In a game whose losing move is "have
  nothing playable", having a legal answer each turn is usually worth more
  than card count.
- Burning is not valued at all. A ten scores 23 purely as "expensive to
  spend", and the four-of-a-kind burn is invisible to the score except as a
  large count. So the agent hoards the one move that reliably rescues a bad
  position, and burns mostly by accident.
- The pile is ignored. Pile size is nowhere in the score, so the agent will
  not take a cheap play now to avoid inheriting fifteen cards next turn.
  This is the most plausible explanation for losing the long games. Long
  games are pile churn, and nothing in the heuristic responds to it.
- Opponents are ignored. The observation carries every opponent's hand
  count, face-up cards and the shared history, and none of it is read. The
  agent plays identically whether an opponent has eight cards or one.

Setup is a separate question. The agent keeps the highest-retention cards
face up, which is defensible, since face-up cards are played after the hand
and before the blind cards and so meet a late, hostile pile, but it also
strips the opening hand of exactly the cards that prevent an early pickup.
Whether that trade is right has not been measured.

## What these numbers do not establish

- The rotations are correlated. 200 matches are 100 deals played twice, so
  the effective sample is nearer 100 than 200. A normal approximation on
  160/200 gives roughly 74 to 85% at 95%, and the true interval is wider. If
  an interval is ever needed, resample whole deal blocks, not individual
  rotations.
- The gauntlet computes no uncertainty interval, for the reason above.
- One run, one machine, one seed. The seat split (76/100 versus 84/100) is
  well inside noise at this sample size and should not be read as a seat
  effect until the mirror control says otherwise.
- The timing column is not a measure of thinking. Both baselines decide in
  well under a millisecond. The ~35 ms is worker startup.

## Where to start on a stronger agent

Ordered by expected gain per unit of work, and each one is testable on its
own:

1. Make batch size conditional rather than dominant. Shed the largest batch
   when it burns (four of a kind) or when the cards are dead weight, and
   otherwise prefer keeping a legal answer for the next turn. This is a
   change to `_score` and nothing else.
2. Value the burn explicitly. Treat a ten, or a four-card batch, as worth
   playing when the pile is large or the constraint is hostile. A burn
   clears the pile, clears the constraint, and grants the same actor another
   decision with a fresh budget, which is the strongest tempo swing in the
   game. The mirror truncations are the evidence, since two agents that both
   hoard their tens can fail to finish a game at all. Treat "the mirror
   match terminates" as an acceptance test, not just a win rate. Deals 10,
   16, 25, 27 and 44 at seed 42 run away in both rotations today.
3. Read the pile. `view.discard_pile` and `view.constraint` are right there.
   Avoiding one pickup is worth more than shedding two cards.
4. Keep a legal answer in reserve. Retention should depend on the constraint
   the agent expects to face, not on a fixed table. A seven that is the only
   card able to answer `AtMost(SEVEN)` is not interchangeable with a seven
   held against an empty pile.
5. Use the opponents. Hand counts and face-up cards are public. Playing to
   deny an opponent on their last card is a different objective from
   shedding fast, and the observation already supports it.
6. Revisit setup as its own experiment, once play is stronger. The current
   arrangement rule has never been tested against an alternative.
7. Only then reach for search. Hidden information, more than two players,
   and extra turns after a burn all mean two-player negamax does not
   transfer. [`design.md`](design.md) sketches the belief-sampling interface
   and recommends Monte Carlo rollouts before information-set search.

## What a long match costs, and why

The mirror run took 3 h 22 m against the random matchup's 16 minutes. Three
things multiply:

1. Decision count. The mirror averaged 1042 play decisions per match against
   the random matchup's 123. The 13 truncated matches alone account for
   130000 decisions, 62% of the run's 208506, from 6.5% of its matches.
2. One process per decision. Roughly 35 ms of worker startup, paid whether
   the agent thinks for a microsecond or the whole budget. It buys the
   isolation the design asks for, and it means runtime tracks the decision
   count, not the thinking.
3. The observation grows with the match. The runner ships the actor's whole
   filtered history inside the `PlayerView` to a fresh worker every
   decision, so the per-decision transport cost is linear in the match
   length and the cost of a match is quadratic in it:

   | history events | pickled view | pickle + unpickle |
   | --- | --- | --- |
   | 0 | 1.0 KB | 0.18 ms |
   | 1000 | 20.8 KB | 4.17 ms |
   | 5000 | 97.2 KB | 21.40 ms |
   | 10000 | 192.7 KB | 50.56 ms |

At ordinary lengths this is invisible. Measured across a 20-match run,
decisions 0 to 24 cost a median 35.7 ms against 36.8 ms at decision 150 and
beyond. At 10000 decisions the last decisions pay about 50 ms of
serialization on top of startup, and one truncated match pushes on the order
of a gigabyte through its pipes. The mean selection time rising from 35 ms
to 56 ms between the two runs is this effect showing up in the report.

Two consequences for anyone running the next comparison:

- Set `--max-play-decisions` to something near the real distribution. The
  median finished mirror game is 75 decisions. A limit of 1000 still catches
  every genuine game, flags a runaway just as clearly, and would have cut
  that run from three hours to well under one. The 10000 default is a safety
  net for a single match, not a budget for two hundred.
- The history-per-decision copy in `shed.match` is worth revisiting. This is
  the first measurement of where it stops being cheap. A fix means a
  persistent worker per seat fed history deltas, or bounding what the view
  carries. Both are design changes, and both should start from the engine
  benchmark (`scripts/benchmark.py`). Note what it measures and what it does
  not. It times `observe()`, which stores an already-filtered history tuple
  by reference, so the cost this section is about, building that tuple per
  decision and shipping it through a pipe, is the runner's and sits outside
  the engine numbers. It does tell you the floor those numbers sit on. An
  apply/undo pair costs about a millisecond, so a decision spending tens of
  milliseconds on transport is not paying for the engine.

## How to measure a change cheaply

Do not use the timed gauntlet as the inner loop. It starts a worker per
decision, so 200 matches take ~35 minutes, and almost all of that is process
startup.

- Fast loop: the synchronous harness in
  [`tests/agents/conftest.py`](../tests/agents/conftest.py).
  `play_baseline_match` drives complete games straight through the engine
  with a `FakeTurn`, building a fresh agent per decision and keeping each
  seat's filtered history, with no timing and no processes. Thousands of
  games run in seconds, and the information boundary is identical to a real
  match.
- Slow loop, for the final claim: `scripts/gauntlet.py`, which is the only
  path that exercises real timed decisions, fallbacks and worker failures.
- Method: compare on the same deals with rotated seats, against both
  `random` and the current `greedy`, and report the denominators. Beating
  random by more than 80% means little on its own. Beating the current
  greedy head to head is the real bar.
- A `--no-replays` report keeps the aggregate and each match's schedule
  entry, status and winner. Drop the flag when the matches themselves need
  verifying. Every embedded document decodes with
  `shed.replay.decode_replay` and checks with `verify_replay`.

## Reproducing

The two commands at the top of this document, then:

```python
import json, statistics
from pathlib import Path

doc = json.loads(Path("results/random-vs-greedy.json").read_text(encoding="utf-8"))
names = [p["name"] for p in doc["participants"]]
wins = [m["play_decisions"] for m in doc["matches"] if names[m["winner"]] == "greedy-1"]
losses = [m["play_decisions"] for m in doc["matches"] if names[m["winner"]] != "greedy-1"]
statistics.median(wins), statistics.median(losses)
```

`results/` is ignored by Git, so the report files themselves are local
artifacts. The numbers quoted above are the record.
