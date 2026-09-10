"""Sequential evaluation: match schedules, independent seeds, and accounting.

The gauntlet is the layer above the match runner and it owns nothing the runner
already owns. It decides *which* matches to play -- one fixed deal per entry in
the deal bank, played once per cyclic seat rotation -- derives the independent
seed streams each of those matches runs on, plays them one after another, and
adds up what came back. Every authoritative :class:`~shed.engine.GameState`
still lives inside a :class:`~shed.match.MatchRunner`, and no rule, legality
check, or transition is reimplemented here.

Three properties are deliberate:

* **Sequential.** Matches are played one at a time in :func:`run_gauntlet`,
  because the budget a timed agent is given is wall time. Two matches running
  side by side would have their agents competing for the same cores, so the
  comparison would measure the machine's load rather than the strategies.
* **Deterministic.** :func:`build_schedule` is pure: the same participants and
  the same :class:`GauntletConfig` produce the same schedule, seeds included.
  Seeds come from :func:`derive_seed`, a SHA-256 of a canonical JSON payload,
  rather than from Python's ``hash()``, which is randomized per interpreter.
  The deck, agent, and fallback streams are derived under separate purposes, so
  an agent's seed can never be a function of the deal it is playing, and a
  fallback draw cannot disturb the deck.
* **Complete.** Every scheduled match keeps its records and its status.
  :func:`summarize` reports finished, truncated, and failed matches separately;
  a match that failed or truncated is never quietly dropped and never counted as
  an ordinary loss. Win rates carry the denominator they were measured against
  in a :class:`Rate`, so "two wins" is never printed without "of how many".

Rotations are *cyclic*: for a lineup of ``n`` participants, deal ``d`` is played
``n`` times, with participant ``p`` sitting in seat ``(p + rotation) % n``. That
gives every participant every seat on every deal, which is what makes the seat
breakdown fair. It does **not** enumerate every seating permutation for three or
more participants -- the participants keep their cyclic order relative to each
other -- so it controls for seat advantage but not for who sits to whose left.
An all-permutations schedule is future work.

Nothing here decodes external input. The command-line script validates what a
user typed, builds a :class:`GauntletConfig` and the participant lineup from it,
and hands typed objects to this module.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import shed
from shed.agents import AgentSpec
from shed.engine import DEFAULT_DEALER, DEFAULT_RULES, PlayerId, RulesConfig
from shed.match import MatchConfig, MatchResult, MatchRunner, MatchStatus
from shed.replay import JsonObject, encode_spec, match_document

__all__ = [
    "SEED_SPACE",
    "Distribution",
    "GauntletConfig",
    "GauntletReport",
    "GauntletRun",
    "MatchFailure",
    "MatchRecord",
    "ParticipantReport",
    "Rate",
    "RunnerFactory",
    "ScheduledMatch",
    "SelectionStats",
    "StatusCounts",
    "build_schedule",
    "derive_seed",
    "gauntlet_document",
    "run_gauntlet",
    "summarize",
    "write_gauntlet",
]

SEED_SPACE = 2**32
"""Range derived seeds are reduced into, matching the runner's own seed space."""

GAUNTLET_SCHEMA_VERSION = 1
"""Version of the gauntlet report document written by :func:`gauntlet_document`.

It is separate from the replay schema: this document *embeds* replay documents,
each carrying their own version, and the two can move independently.
"""


def derive_seed(experiment_seed: int, purpose: str, *fields: int) -> int:
    """Derive one independent seed from the experiment seed.

    The payload is canonical JSON -- fixed field order, no incidental whitespace
    -- hashed with SHA-256 and reduced into :data:`SEED_SPACE`. Python's built-in
    ``hash()`` is deliberately not used: it is randomized per interpreter, so a
    schedule built with it would not reproduce across runs.

    Streams are separated by ``purpose`` rather than by arithmetic on one
    counter, so adding a stream cannot shift an existing one, and a deck seed
    and an agent seed for the same match are unrelated values.

    Args:
        experiment_seed: The gauntlet's root seed.
        purpose: Name of the stream, such as ``"deck"``, ``"agent"``, or
            ``"fallback"``.
        *fields: Coordinates within the stream, such as a deal index or a match
            index.

    Returns:
        A non-negative seed below :data:`SEED_SPACE`.
    """
    payload = json.dumps([experiment_seed, purpose, *fields], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % SEED_SPACE


@dataclass(frozen=True, slots=True)
class GauntletConfig:
    """What one gauntlet run is configured with.

    Everything a match needs beyond its seeds is here, and the seeds themselves
    are derived from :attr:`seed`, so a run reproduces its whole schedule from
    this object and the participant lineup.

    Attributes:
        deals: Deals in the bank. Each is played once per seat rotation, so a
            lineup of ``n`` participants plays ``deals * n`` matches.
        seed: Root seed every derived stream hangs off.
        dealer: Dealing seat, held fixed across the rotations of a deal so the
            rotations differ only in who sits where.
        seconds_per_turn: Acceptance budget per decision.
        max_play_decisions: Bound on play decisions before a match truncates.
        strict_failures: Whether an agent failure aborts a match instead of
            playing on from a fallback.
    """

    deals: int = 10
    seed: int = 0
    dealer: PlayerId = DEFAULT_DEALER
    seconds_per_turn: float = 2.0
    max_play_decisions: int = MatchConfig().max_play_decisions
    strict_failures: bool = False

    def __post_init__(self) -> None:
        """Check the deal bank is usable and the match settings are valid.

        The match settings are validated by building a :class:`MatchConfig` from
        them, so the budget and action limit obey exactly one set of rules.

        Raises:
            ValueError: If the deal count is not positive, or the timing
                settings do not make a runnable match.
        """
        if self.deals < 1:
            raise ValueError(f"deals must be positive, got {self.deals}")
        self.match_config(agent_seed=0, fallback_seed=0)

    def match_config(self, *, agent_seed: int, fallback_seed: int) -> MatchConfig:
        """Build the runner configuration for one scheduled match.

        Args:
            agent_seed: Seed of that match's agent-seed stream.
            fallback_seed: Seed of that match's fallback stream.

        Returns:
            The match configuration, sharing this run's timing and failure
            policy and carrying only the two seeds that differ per match.

        Raises:
            ValueError: If the timing settings do not make a runnable match.
        """
        return MatchConfig(
            seconds_per_turn=self.seconds_per_turn,
            max_play_decisions=self.max_play_decisions,
            fallback_seed=fallback_seed,
            agent_seed=agent_seed,
            strict_failures=self.strict_failures,
        )


@dataclass(frozen=True, slots=True)
class ScheduledMatch:
    """One entry of a schedule: a deal, a rotation, and the seeds to play it on.

    Attributes:
        index: Position in the schedule, and the coordinate the agent and
            fallback streams are derived at.
        deal_index: Which deal in the bank. Every rotation of one deal shares
            this index, this deck seed, and this dealer.
        rotation: Seat offset applied to the participants: participant ``p``
            sits in seat ``(p + rotation) % n``.
        deal_seed: Deck seed for the deal. Trusted metadata; it never reaches an
            agent.
        dealer: Dealing seat.
        seats: Participant index per seat, so ``seats[s]`` is the participant
            sitting in seat ``s``.
        config: Timing, limits, failure policy, and this match's two seeds.
    """

    index: int
    deal_index: int
    rotation: int
    deal_seed: int
    dealer: PlayerId
    seats: tuple[int, ...]
    config: MatchConfig

    def lineup(self, participants: Sequence[AgentSpec]) -> dict[PlayerId, AgentSpec]:
        """Seat the participants for this match.

        Args:
            participants: The lineup in participant order, the same sequence the
                schedule was built from.

        Returns:
            The specification per seat, ready for a :class:`MatchRunner`.
        """
        return {PlayerId(seat): participants[index] for seat, index in enumerate(self.seats)}


def build_schedule(
    participants: Sequence[AgentSpec],
    config: GauntletConfig,
) -> tuple[ScheduledMatch, ...]:
    """Build the complete match schedule for a lineup.

    Pure and total: no clock, no randomness beyond the derived seeds, and no
    match is played. The same arguments always produce the same schedule, which
    is what lets a run be described before it is started and compared after it.

    Deals are the outer loop and rotations the inner one, so the ``n`` rotations
    of a deal are adjacent and share a deck seed and a dealer. Only the seating
    and the per-match seeds differ between them.

    Args:
        participants: The lineup, in participant order. Two to five, matching
            the profile, with distinct labels.
        config: The run's configuration.

    Returns:
        ``config.deals * len(participants)`` entries, in play order.

    Raises:
        ValueError: If the lineup is outside the profile's supported table size
            or two participants share a label.
    """
    count = len(participants)
    if not DEFAULT_RULES.min_players <= count <= DEFAULT_RULES.max_players:
        raise ValueError(
            f"{DEFAULT_RULES.id} supports {DEFAULT_RULES.min_players}-"
            f"{DEFAULT_RULES.max_players} players, got {count}"
        )
    labels = [spec.name for spec in participants]
    if len(set(labels)) != count:
        raise ValueError(f"Participant labels must be distinct, got {labels}")

    schedule: list[ScheduledMatch] = []
    for deal_index in range(config.deals):
        deal_seed = derive_seed(config.seed, "deck", deal_index)
        for rotation in range(count):
            index = len(schedule)
            schedule.append(
                ScheduledMatch(
                    index=index,
                    deal_index=deal_index,
                    rotation=rotation,
                    deal_seed=deal_seed,
                    dealer=config.dealer,
                    # seats[s] is the participant in seat s, the inverse of
                    # "participant p sits in seat (p + rotation) % n".
                    seats=tuple((seat - rotation) % count for seat in range(count)),
                    config=config.match_config(
                        agent_seed=derive_seed(config.seed, "agent", index),
                        fallback_seed=derive_seed(config.seed, "fallback", index),
                    ),
                )
            )
    return tuple(schedule)


@dataclass(frozen=True, slots=True)
class MatchRecord:
    """One scheduled match together with what playing it produced.

    Attributes:
        scheduled: The entry that was played, including its seeds and seating.
        result: Everything the runner recorded, whatever ended the match.
    """

    scheduled: ScheduledMatch
    result: MatchResult

    @property
    def winner(self) -> int | None:
        """Return the participant index that won, or ``None`` if nobody did.

        Returns:
            The winning participant, or ``None`` for a truncated or aborted
            match. A match without an outcome never attributes a win.
        """
        outcome = self.result.outcome
        return None if outcome is None else self.scheduled.seats[outcome.winner]

    def participant(self, seat: PlayerId) -> int:
        """Return the participant index sitting in one seat of this match.

        Args:
            seat: The seat to resolve.

        Returns:
            The participant index.
        """
        return self.scheduled.seats[seat]


@dataclass(frozen=True, slots=True)
class Rate:
    """A count and the denominator it was measured against.

    Pairing the two is the point: a win count is meaningless without the number
    of matches it came from, and a gauntlet reports several denominators that
    are easy to confuse -- matches played, matches finished, decisions taken.

    Attributes:
        count: How many times the thing happened.
        total: How many opportunities there were.
    """

    count: int
    total: int

    @property
    def value(self) -> float | None:
        """Return the ratio, or ``None`` when nothing was measured.

        Returns:
            ``count / total``, or ``None`` for an empty denominator rather than
            a fabricated zero.
        """
        return None if self.total == 0 else self.count / self.total


@dataclass(frozen=True, slots=True)
class Distribution:
    """A small summary of a sample: how many, and where its middle is.

    Mean and median are both reported because selection times are skewed -- a
    decision that reaches the deadline sits well above one an agent finalized
    immediately -- and either alone would be misleading.

    Attributes:
        count: Sample size.
        mean: Arithmetic mean, or ``None`` for an empty sample.
        median: Median, or ``None`` for an empty sample.
    """

    count: int
    mean: float | None
    median: float | None

    @classmethod
    def of(cls, values: Sequence[float]) -> Distribution:
        """Summarize a sample.

        Args:
            values: The observations, in any order.

        Returns:
            The summary; an empty sample yields ``None`` for both statistics.
        """
        if not values:
            return cls(count=0, mean=None, median=None)
        return cls(
            count=len(values),
            mean=statistics.fmean(values),
            median=float(statistics.median(values)),
        )


@dataclass(frozen=True, slots=True)
class SelectionStats:
    """How the decisions in a set of matches went.

    Every rate here shares one denominator, :attr:`decisions`, so the three
    diagnostics can be compared with each other directly.

    Attributes:
        decisions: Selections made, applied or not.
        fallbacks: Decisions that accepted no candidate and fell back to a
            seeded legal move.
        rejections: Submissions refused as malformed or illegal, over the same
            decision denominator. One decision can contribute several, so this
            ratio may exceed one.
        crashes: Decisions whose worker reported a failure.
        selection_seconds: Wall time per decision, from starting the worker to
            closing the decision.
    """

    decisions: int
    fallbacks: Rate
    rejections: Rate
    crashes: Rate
    selection_seconds: Distribution

    @classmethod
    def of(cls, records: Sequence[MatchRecord], participant: int | None = None) -> SelectionStats:
        """Summarize the decisions of some matches, optionally for one participant.

        Args:
            records: The matches to read. Failed and truncated matches count
                here exactly like finished ones: their decisions happened.
            participant: Restrict to decisions taken by this participant, or
                ``None`` for every decision in the matches.

        Returns:
            The diagnostics over the selected decisions.
        """
        fallbacks = rejections = crashes = 0
        elapsed: list[float] = []
        for record in records:
            for turn in record.result.turns:
                if participant is not None and record.participant(turn.player) != participant:
                    continue
                fallbacks += turn.used_fallback
                rejections += turn.rejected
                crashes += turn.failure is not None
                elapsed.append(turn.selection_seconds)
        decisions = len(elapsed)
        return cls(
            decisions=decisions,
            fallbacks=Rate(fallbacks, decisions),
            rejections=Rate(rejections, decisions),
            crashes=Rate(crashes, decisions),
            selection_seconds=Distribution.of(elapsed),
        )


@dataclass(frozen=True, slots=True)
class StatusCounts:
    """How many scheduled matches ended each way.

    The four statuses are reported separately and always add up to
    :attr:`scheduled`. A truncated match is not a loss, and an aborted one is
    not a match that merely went badly for somebody; folding either into the
    win/loss accounting would overstate how much was actually played.

    Attributes:
        scheduled: Matches in the schedule.
        finished: Matches the rules ended, the only ones with a winner.
        truncated: Matches the action limit stopped.
        agent_failed: Matches strict mode aborted on an agent failure.
        engine_failed: Matches the engine or the runner's infrastructure ended.
    """

    scheduled: int
    finished: int
    truncated: int
    agent_failed: int
    engine_failed: int

    def __post_init__(self) -> None:
        """Check every scheduled match is accounted for exactly once.

        Raises:
            ValueError: If the per-status counts do not add up to the number
                scheduled, which would mean a match went missing from the
                accounting.
        """
        if self.played != self.scheduled:
            raise ValueError(
                f"Status counts sum to {self.played} but {self.scheduled} matches were scheduled"
            )

    @property
    def played(self) -> int:
        """Return how many matches the per-status counts account for."""
        return self.finished + self.truncated + self.agent_failed + self.engine_failed

    @property
    def failed(self) -> int:
        """Return how many matches ended in a failure of either kind."""
        return self.agent_failed + self.engine_failed

    @classmethod
    def of(cls, records: Sequence[MatchRecord], scheduled: int | None = None) -> StatusCounts:
        """Count how a set of matches ended.

        Args:
            records: The matches to count.
            scheduled: How many were scheduled, when that is known separately.
                Defaults to the number of records.

        Returns:
            The counts.

        Raises:
            ValueError: If the records do not account for every scheduled match.
        """
        tally = {status: 0 for status in MatchStatus}
        for record in records:
            tally[record.result.status] += 1
        return cls(
            scheduled=len(records) if scheduled is None else scheduled,
            finished=tally[MatchStatus.FINISHED],
            truncated=tally[MatchStatus.TRUNCATED],
            agent_failed=tally[MatchStatus.AGENT_FAILED],
            engine_failed=tally[MatchStatus.ENGINE_FAILED],
        )


@dataclass(frozen=True, slots=True)
class MatchFailure:
    """A scheduled match that did not finish, named so it cannot be overlooked.

    Attributes:
        index: Position in the schedule.
        deal_index: Which deal it came from.
        rotation: Which rotation of that deal.
        status: How it ended; never :attr:`~shed.match.MatchStatus.FINISHED`.
        detail: The runner's recorded explanation, when it left one.
    """

    index: int
    deal_index: int
    rotation: int
    status: MatchStatus
    detail: str | None


@dataclass(frozen=True, slots=True)
class ParticipantReport:
    """What one participant did across the whole gauntlet.

    Attributes:
        index: Position in the lineup.
        spec: The participant's kind and label.
        wins: Wins over the finished matches this participant played. Truncated
            and aborted matches are in neither the numerator nor the
            denominator, so they are never silently counted as losses.
        by_seat: Wins per seat, indexed by seat. With cyclic rotations every
            participant plays every seat the same number of times, so these
            denominators are equal in a complete run.
        by_opponents: Wins per set of opponents, keyed by their labels joined
            with ``" vs "``. In a two-participant gauntlet this is the head to
            head; with one fixed lineup of three or more it collapses to a
            single key, because the opposition never changes.
        selection: Decision diagnostics for this participant's own decisions.
    """

    index: int
    spec: AgentSpec
    wins: Rate
    by_seat: tuple[Rate, ...]
    by_opponents: tuple[tuple[str, Rate], ...]
    selection: SelectionStats


@dataclass(frozen=True, slots=True)
class GauntletReport:
    """The aggregate of a whole gauntlet run.

    Attributes:
        participants: The lineup, in participant order.
        config: What the run was configured with.
        status: How the scheduled matches ended.
        agents: One report per participant, in lineup order.
        selection: Decision diagnostics over every match.
        play_decisions: Play decisions per match, over every match.
        failures: Every match that did not finish, in schedule order.
    """

    participants: tuple[AgentSpec, ...]
    config: GauntletConfig
    status: StatusCounts
    agents: tuple[ParticipantReport, ...]
    selection: SelectionStats
    play_decisions: Distribution
    failures: tuple[MatchFailure, ...]


def _opponents_key(participants: Sequence[AgentSpec], index: int) -> str:
    """Name the opposition one participant faced in a lineup.

    Args:
        participants: The lineup, in participant order.
        index: The participant whose opposition to name.

    Returns:
        The other participants' labels in lineup order, joined with ``" vs "``.
    """
    return " vs ".join(spec.name for position, spec in enumerate(participants) if position != index)


def _participant_report(
    participants: Sequence[AgentSpec],
    records: Sequence[MatchRecord],
    index: int,
) -> ParticipantReport:
    """Aggregate one participant's results.

    Only finished matches enter the win accounting, and each such match
    contributes one to exactly one seat denominator, so the seat breakdown adds
    up to the overall record.

    Args:
        participants: The lineup, in participant order.
        records: Every match record of the run.
        index: The participant to report on.

    Returns:
        The participant's report.
    """
    seats = len(participants)
    played = 0
    wins = 0
    seat_played = [0] * seats
    seat_wins = [0] * seats
    opponents: dict[str, list[int]] = {}

    for record in records:
        if record.result.status is not MatchStatus.FINISHED:
            continue
        seat = record.scheduled.seats.index(index)
        won = record.winner == index
        played += 1
        wins += won
        seat_played[seat] += 1
        seat_wins[seat] += won
        # The lineup is fixed for a run, so a participant faces one opposition;
        # the key is computed per match anyway, so a schedule that ever mixes
        # lineups would split these rows rather than silently pool them.
        key = _opponents_key(participants, index)
        tally = opponents.setdefault(key, [0, 0])
        tally[0] += won
        tally[1] += 1

    return ParticipantReport(
        index=index,
        spec=participants[index],
        wins=Rate(wins, played),
        by_seat=tuple(Rate(seat_wins[seat], seat_played[seat]) for seat in range(seats)),
        by_opponents=tuple(
            (key, Rate(tally[0], tally[1])) for key, tally in sorted(opponents.items())
        ),
        selection=SelectionStats.of(records, participant=index),
    )


def summarize(
    participants: Sequence[AgentSpec],
    config: GauntletConfig,
    records: Sequence[MatchRecord],
    *,
    scheduled: int | None = None,
) -> GauntletReport:
    """Aggregate the records of a gauntlet run.

    Pure: it reads records and computes counts, so a caller can re-aggregate a
    run read back from disk, or aggregate records produced by something other
    than :func:`run_gauntlet`.

    Args:
        participants: The lineup, in participant order.
        config: What the run was configured with.
        records: Every match that was played, in schedule order.
        scheduled: How many matches were scheduled, when a run stopped short of
            playing them all. Defaults to the number of records.

    Returns:
        The aggregate report.

    Raises:
        ValueError: If the records do not account for every scheduled match.
    """
    return GauntletReport(
        participants=tuple(participants),
        config=config,
        status=StatusCounts.of(records, scheduled),
        agents=tuple(
            _participant_report(participants, records, index) for index in range(len(participants))
        ),
        selection=SelectionStats.of(records),
        play_decisions=Distribution.of([record.result.play_decisions for record in records]),
        failures=tuple(
            MatchFailure(
                index=record.scheduled.index,
                deal_index=record.scheduled.deal_index,
                rotation=record.scheduled.rotation,
                status=record.result.status,
                detail=record.result.failure,
            )
            for record in records
            if record.result.status is not MatchStatus.FINISHED
        ),
    )


@dataclass(frozen=True, slots=True)
class GauntletRun:
    """A gauntlet that has been played, with everything it produced.

    Attributes:
        participants: The lineup, in participant order.
        config: What the run was configured with.
        rules: The profile the matches ran under.
        schedule: Every match that was scheduled, in play order.
        records: Every match that was played, in the same order.
    """

    participants: tuple[AgentSpec, ...]
    config: GauntletConfig
    rules: RulesConfig
    schedule: tuple[ScheduledMatch, ...]
    records: tuple[MatchRecord, ...]

    @property
    def report(self) -> GauntletReport:
        """Return the aggregate of this run.

        Returns:
            The report, computed over the records against the scheduled count.
        """
        return summarize(self.participants, self.config, self.records, scheduled=len(self.schedule))


class RunnerFactory(Protocol):
    """How :func:`run_gauntlet` obtains a runner for one scheduled match."""

    def __call__(
        self,
        agents: dict[PlayerId, AgentSpec],
        config: MatchConfig,
        *,
        rules: RulesConfig,
    ) -> MatchRunner:
        """Build the runner for one match.

        Args:
            agents: Participant per seat for this match.
            config: Timing, limits, failure policy, and this match's seeds.
            rules: The profile to play under.

        Returns:
            A runner ready to play exactly this match.
        """


def run_gauntlet(
    participants: Sequence[AgentSpec],
    config: GauntletConfig,
    *,
    rules: RulesConfig = DEFAULT_RULES,
    runner_factory: RunnerFactory = MatchRunner,
    on_match: Callable[[MatchRecord], None] | None = None,
) -> GauntletRun:
    """Play a whole schedule, one match at a time.

    Matches run sequentially and each one gets a fresh runner, so no two timed
    agents ever compete for the same cores and no state survives from one match
    into the next. A failed or truncated match is recorded and the schedule
    continues: the run reports what happened rather than stopping at the first
    disappointment.

    Args:
        participants: The lineup, in participant order.
        config: The run's configuration.
        rules: The profile to play under.
        runner_factory: How to build the runner for a match. The default builds
            a real :class:`~shed.match.MatchRunner`; a test may substitute a
            synchronous stand-in to exercise scheduling without starting
            processes.
        on_match: Called with each record as it completes, for progress
            reporting during a long run.

    Returns:
        The completed run, with the schedule and every record.

    Raises:
        ValueError: If the lineup cannot be scheduled.
    """
    schedule = build_schedule(participants, config)
    records: list[MatchRecord] = []
    for scheduled in schedule:
        runner = runner_factory(scheduled.lineup(participants), scheduled.config, rules=rules)
        result = runner.run(deal_seed=scheduled.deal_seed, dealer=scheduled.dealer)
        record = MatchRecord(scheduled=scheduled, result=result)
        records.append(record)
        if on_match is not None:
            on_match(record)
    return GauntletRun(
        participants=tuple(participants),
        config=config,
        rules=rules,
        schedule=schedule,
        records=tuple(records),
    )


# --------------------------------------------------------------------------- #
# Serialization: the report document, embedding one replay per match.
# --------------------------------------------------------------------------- #


def _encode_rate(rate: Rate) -> JsonObject:
    """Encode a count with its denominator.

    Args:
        rate: The measurement.

    Returns:
        The count, the denominator, and the ratio -- ``null`` when nothing was
        measured, never a fabricated zero.
    """
    return {"count": rate.count, "total": rate.total, "rate": rate.value}


def _encode_distribution(distribution: Distribution) -> JsonObject:
    """Encode a sample summary.

    Args:
        distribution: The summary.

    Returns:
        The sample size, the mean, and the median.
    """
    return {
        "count": distribution.count,
        "mean": distribution.mean,
        "median": distribution.median,
    }


def _encode_selection(stats: SelectionStats) -> JsonObject:
    """Encode decision diagnostics.

    Args:
        stats: The diagnostics.

    Returns:
        The decision count, each rate with its denominator, and the timing
        summary.
    """
    return {
        "decisions": stats.decisions,
        "fallbacks": _encode_rate(stats.fallbacks),
        "rejections": _encode_rate(stats.rejections),
        "crashes": _encode_rate(stats.crashes),
        "selection_seconds": _encode_distribution(stats.selection_seconds),
    }


def _encode_config(config: GauntletConfig) -> JsonObject:
    """Encode the run configuration.

    Args:
        config: What the run was configured with.

    Returns:
        Every field, so the schedule can be rebuilt from the document.
    """
    return {
        "deals": config.deals,
        "seed": config.seed,
        "dealer": int(config.dealer),
        "seconds_per_turn": config.seconds_per_turn,
        "max_play_decisions": config.max_play_decisions,
        "strict_failures": config.strict_failures,
    }


def _encode_report(report: GauntletReport) -> JsonObject:
    """Encode the aggregate report.

    Args:
        report: The aggregate.

    Returns:
        The status accounting, one entry per participant, the overall
        diagnostics, and every match that did not finish.
    """
    status = report.status
    return {
        "status": {
            "scheduled": status.scheduled,
            "finished": status.finished,
            "truncated": status.truncated,
            "agent_failed": status.agent_failed,
            "engine_failed": status.engine_failed,
        },
        "agents": [
            {
                "index": agent.index,
                "agent": encode_spec(agent.spec),
                "wins": _encode_rate(agent.wins),
                "by_seat": [_encode_rate(rate) for rate in agent.by_seat],
                "by_opponents": [
                    {"opponents": key, "wins": _encode_rate(rate)}
                    for key, rate in agent.by_opponents
                ],
                "selection": _encode_selection(agent.selection),
            }
            for agent in report.agents
        ],
        "selection": _encode_selection(report.selection),
        "play_decisions": _encode_distribution(report.play_decisions),
        "failures": [
            {
                "index": failure.index,
                "deal_index": failure.deal_index,
                "rotation": failure.rotation,
                "status": failure.status.value,
                "detail": failure.detail,
            }
            for failure in report.failures
        ],
    }


def _encode_record(record: MatchRecord, *, replays: bool) -> JsonObject:
    """Encode one played match.

    The match itself is encoded by :func:`~shed.replay.match_document`, the
    project's one match format, so a gauntlet file holds real replays rather
    than a second, weaker record of the same games.

    Args:
        record: The match to encode.
        replays: Whether to embed the full replay document. With it left out the
            entry keeps its schedule, its status, and its winner, but the match
            can no longer be replayed from this file.

    Returns:
        The schedule entry, how the match ended, and optionally the replay.
    """
    scheduled = record.scheduled
    entry: JsonObject = {
        "index": scheduled.index,
        "deal_index": scheduled.deal_index,
        "rotation": scheduled.rotation,
        "deal_seed": scheduled.deal_seed,
        "dealer": int(scheduled.dealer),
        "seats": list(scheduled.seats),
        "agent_seed": scheduled.config.agent_seed,
        "fallback_seed": scheduled.config.fallback_seed,
        "status": record.result.status.value,
        "winner": record.winner,
        "play_decisions": record.result.play_decisions,
        "failure": record.result.failure,
    }
    if replays:
        entry["replay"] = match_document(record.result)
    return entry


def gauntlet_document(
    run: GauntletRun,
    *,
    source_revision: str | None = None,
    replays: bool = True,
) -> JsonObject:
    """Build the complete report document for one gauntlet run.

    Args:
        run: The completed run.
        source_revision: Commit the run was played at, when it is known. Pass
            :func:`~shed.replay.detect_source_revision` for the ordinary case.
        replays: Whether to embed each match's replay document. Keeping them
            makes the file self-contained -- every finished match in it can be
            verified with :func:`~shed.replay.verify_replay` -- at the cost of
            its size.

    Returns:
        A JSON-ready document: provenance, the configuration, the lineup, the
        aggregate report, and one entry per played match.
    """
    return {
        "schema": GAUNTLET_SCHEMA_VERSION,
        "package_version": shed.__version__,
        "python_version": platform.python_version(),
        "source_revision": source_revision,
        "rules": run.rules.id,
        "config": _encode_config(run.config),
        "participants": [encode_spec(spec) for spec in run.participants],
        "report": _encode_report(run.report),
        "matches": [_encode_record(record, replays=replays) for record in run.records],
    }


def write_gauntlet(
    run: GauntletRun,
    path: Path,
    *,
    source_revision: str | None = None,
    replays: bool = True,
) -> Path:
    """Write a gauntlet run to a UTF-8 JSON file.

    Missing parent directories are created, so a caller can name
    ``results/gauntlet.json`` in a fresh checkout.

    Args:
        run: The completed run.
        path: Destination file, overwritten if it exists.
        source_revision: Commit the run was played at, when it is known.
        replays: Whether to embed each match's replay document.

    Returns:
        The path written, for the caller to report.

    Raises:
        ValueError: If a measurement is not finite. JSON has no ``NaN``, and
            writing one would produce a file no strict reader accepts.
    """
    document = gauntlet_document(run, source_revision=source_revision, replays=replays)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(f"{text}\n", encoding="utf-8")
    return path
