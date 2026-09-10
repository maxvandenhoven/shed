"""Schedules, seed streams, accounting, and the gauntlet command.

The module is layered the way the gauntlet is. :func:`build_schedule` and
:func:`derive_seed` are pure, so they are tested directly and exhaustively.
:func:`run_gauntlet` is tested with a synchronous runner injected in place of the
real one, which exercises the scheduling, the seeding, and the record keeping
without starting a worker for every decision. Aggregation is tested on records
built by hand, because that is the only way to put a finished, a truncated, and
both kinds of failed match in the same run and check that all four are still
accounted for.

Two tests do run the real thing. One plays a small baseline gauntlet through the
documented command and verifies every finished match from the file it wrote,
which is the milestone's acceptance criterion; the other checks that a schedule
is identical in a fresh interpreter, which is what rules out Python's randomized
``hash()`` sneaking into a seed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from shed.agents import AgentSpec
from shed.cli import build_participants, gauntlet_progress, gauntlet_summary
from shed.engine import DEFAULT_RULES, Outcome, Phase, PickUp, PlayerId, RulesConfig, Unrestricted
from shed.gauntlet import (
    SEED_SPACE,
    Distribution,
    GauntletConfig,
    GauntletRun,
    MatchRecord,
    Rate,
    RunnerFactory,
    ScheduledMatch,
    StatusCounts,
    build_schedule,
    derive_seed,
    gauntlet_document,
    run_gauntlet,
    summarize,
)
from shed.match import (
    CloseReason,
    FinalPosition,
    MatchConfig,
    MatchMetadata,
    MatchResult,
    MatchRunner,
    MatchStatus,
    TurnRecord,
    WorkerFailed,
)
from shed.replay import decode_replay, verify_replay
from tests.test_replay import SyncRunner

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
"""The directory the documented command is run from."""

PAIR = ("random", "greedy")
"""The two-agent lineup most tests here schedule."""


def participants(*kinds: str) -> tuple[AgentSpec, ...]:
    """Build a lineup with the labels the commands would give it.

    Args:
        *kinds: Agent kinds in lineup order.

    Returns:
        The participants, in the same order.
    """
    return build_participants(kinds)


def sync_factory(*, seed: int = 0, failing: Sequence[int] = ()) -> RunnerFactory:
    """Build a runner factory that decides synchronously.

    The gauntlet's only requirement of a runner is the two calls it makes, so a
    synchronous stand-in exercises the whole scheduling and accounting path in
    milliseconds instead of starting one process per decision.

    Args:
        seed: Seed for each runner's move-selection generator.
        failing: Decision identifiers each runner reports as agent failures.

    Returns:
        A factory the gauntlet can call for every scheduled match.
    """

    def factory(
        agents: dict[PlayerId, AgentSpec],
        config: MatchConfig,
        *,
        rules: RulesConfig = DEFAULT_RULES,
    ) -> MatchRunner:
        """Build one synchronous runner.

        Args:
            agents: Participant per seat for this match.
            config: Timing, limits, failure policy, and the match's seeds.
            rules: Accepted and ignored; the stand-in always plays the default
                profile.

        Returns:
            The runner for that match.
        """
        return SyncRunner(agents, config, seed=seed, failing=failing)

    return factory


def empty_position() -> FinalPosition:
    """Build a position digest for a synthetic result that describes no game.

    Aggregation never reads the position, so a synthetic record can carry an
    empty one rather than a crafted state.

    Returns:
        A digest with no cards in it.
    """
    return FinalPosition(
        phase=Phase.PLAY,
        current_player=None,
        current_ply=0,
        constraint=Unrestricted(),
        draw_pile=(),
        discard_pile=(),
        burned_cards=(),
        players=(),
    )


def turn(
    decision_id: int,
    player: int,
    *,
    fallback: bool = False,
    rejected: int = 0,
    crashed: bool = False,
    seconds: float = 0.01,
) -> TurnRecord:
    """Build one synthetic selection record.

    Args:
        decision_id: Position of the decision in its match.
        player: The seat that decided.
        fallback: Whether the decision accepted no candidate.
        rejected: Submissions refused during the decision.
        crashed: Whether the worker reported a failure.
        seconds: Wall time the selection took.

    Returns:
        The record, carrying the diagnostics a gauntlet aggregates.
    """
    return TurnRecord(
        decision_id=decision_id,
        player=PlayerId(player),
        phase=Phase.PLAY,
        move=PickUp(),
        reason=CloseReason.DEADLINE if fallback else CloseReason.FINAL,
        used_fallback=fallback,
        accepted=0 if fallback else 1,
        rejected=rejected,
        failure=WorkerFailed(exception="ValueError", message="synthetic") if crashed else None,
        budget_seconds=1.0,
        selection_seconds=seconds,
        cleanup_seconds=0.001,
        agent_seed=decision_id,
    )


def synthetic(
    scheduled: ScheduledMatch,
    lineup: Sequence[AgentSpec],
    status: MatchStatus,
    *,
    winner_seat: int | None = None,
    turns: Sequence[TurnRecord] = (),
    play_decisions: int = 0,
) -> MatchRecord:
    """Build a match record without playing a match.

    Args:
        scheduled: The schedule entry the record belongs to.
        lineup: The participants, in lineup order.
        status: How the match ended.
        winner_seat: The winning seat, for a finished match.
        turns: Selection records to attach.
        play_decisions: Applied play decisions to report.

    Returns:
        The record.
    """
    return MatchRecord(
        scheduled=scheduled,
        result=MatchResult(
            status=status,
            outcome=None if winner_seat is None else Outcome(PlayerId(winner_seat)),
            turns=tuple(turns),
            initial_events=(),
            decisions=(),
            play_decisions=play_decisions,
            failure=None if status is MatchStatus.FINISHED else f"synthetic {status.value}",
            metadata=MatchMetadata(
                rules=DEFAULT_RULES,
                player_count=len(scheduled.seats),
                dealer=scheduled.dealer,
                deal_seed=scheduled.deal_seed,
                agents=tuple(lineup[index] for index in scheduled.seats),
                config=scheduled.config,
            ),
            final_position=empty_position(),
        ),
    )


class TestDerivedSeeds:
    """The stream derivation every schedule hangs off."""

    def test_streams_with_the_same_coordinates_do_not_collide(self) -> None:
        """The purpose separates the streams, so a deck and an agent differ."""
        seeds = {derive_seed(42, purpose, 0) for purpose in ("deck", "agent", "fallback")}

        assert len(seeds) == 3

    def test_coordinates_within_a_stream_differ(self) -> None:
        """Consecutive matches draw unrelated seeds from the same stream."""
        seeds = {derive_seed(42, "agent", index) for index in range(64)}

        assert len(seeds) == 64

    def test_the_root_seed_changes_every_stream(self) -> None:
        """A different experiment is a different set of deals and decisions."""
        assert derive_seed(42, "deck", 0) != derive_seed(43, "deck", 0)

    def test_seeds_stay_inside_the_runners_seed_space(self) -> None:
        """A derived seed is usable wherever a drawn one is."""
        seeds = [derive_seed(7, "deck", index) for index in range(100)]

        assert all(0 <= seed < SEED_SPACE for seed in seeds)

    def test_derivation_is_stable_across_interpreters(self) -> None:
        """Hash randomization cannot reach a seed, so a run reproduces tomorrow.

        Two fresh interpreters with different ``PYTHONHASHSEED`` values must
        derive the same seed. That is what Python's own ``hash()`` would fail.
        """
        program = "from shed.gauntlet import derive_seed; print(derive_seed(42, 'deck', 3))"
        values = {
            subprocess.run(
                [sys.executable, "-c", program],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"},
            ).stdout.strip()
            for hash_seed in ("0", "1", "12345")
        }

        assert values == {str(derive_seed(42, "deck", 3))}


class TestSchedule:
    """What :func:`build_schedule` lays out before anything is played."""

    @pytest.mark.parametrize("deals", [1, 3, 10])
    def test_two_agents_and_n_deals_produce_two_n_matches(self, deals: int) -> None:
        """The documented rotation count: every deal is played by both seatings."""
        schedule = build_schedule(participants(*PAIR), GauntletConfig(deals=deals))

        assert len(schedule) == 2 * deals

    @pytest.mark.parametrize("agents", [2, 3, 4, 5])
    def test_a_lineup_is_rotated_once_per_participant(self, agents: int) -> None:
        """A lineup of ``n`` plays each deal ``n`` times, once per rotation."""
        lineup = participants(*(["random"] * agents))
        schedule = build_schedule(lineup, GauntletConfig(deals=4))

        assert len(schedule) == 4 * agents
        assert [entry.rotation for entry in schedule[:agents]] == list(range(agents))

    def test_the_rotations_of_a_deal_share_deck_and_dealer(self) -> None:
        """Rotations differ in seating and in nothing else about the deal."""
        schedule = build_schedule(participants(*PAIR), GauntletConfig(deals=3, seed=5))

        for start in range(0, len(schedule), 2):
            first, second = schedule[start], schedule[start + 1]
            assert first.deal_index == second.deal_index
            assert first.deal_seed == second.deal_seed
            assert first.dealer == second.dealer
            assert first.seats != second.seats

    def test_separate_deals_get_separate_decks(self) -> None:
        """A deal bank is a bank of distinct deals."""
        schedule = build_schedule(participants(*PAIR), GauntletConfig(deals=25))
        seeds = {entry.deal_index: entry.deal_seed for entry in schedule}

        assert len(set(seeds.values())) == 25

    @pytest.mark.parametrize("agents", [2, 3, 5])
    def test_every_participant_plays_every_seat_equally_often(self, agents: int) -> None:
        """Cyclic rotation is what makes the seat breakdown a fair comparison."""
        lineup = participants(*(["greedy"] * agents))
        schedule = build_schedule(lineup, GauntletConfig(deals=6))

        for index in range(agents):
            seats = [entry.seats.index(index) for entry in schedule]
            assert sorted(seats) == sorted(list(range(agents)) * 6)

    def test_a_rotation_seats_participants_cyclically(self) -> None:
        """Participant ``p`` sits in seat ``(p + rotation) % n``."""
        lineup = participants("random", "greedy", "greedy")
        schedule = build_schedule(lineup, GauntletConfig(deals=1))

        for entry in schedule:
            seated = entry.lineup(lineup)
            for index, spec in enumerate(lineup):
                assert seated[PlayerId((index + entry.rotation) % 3)] == spec

    def test_the_three_seed_streams_stay_apart(self) -> None:
        """No match reuses one seed for the deck, the agents, and the fallback."""
        schedule = build_schedule(participants(*PAIR), GauntletConfig(deals=20))

        for entry in schedule:
            seeds = {entry.deal_seed, entry.config.agent_seed, entry.config.fallback_seed}
            assert len(seeds) == 3

    def test_agent_seeds_are_per_match_and_deck_seeds_are_per_deal(self) -> None:
        """Two rotations share a deal but never share an agent stream."""
        schedule = build_schedule(participants(*PAIR), GauntletConfig(deals=8))
        agent_seeds = [entry.config.agent_seed for entry in schedule]
        fallback_seeds = [entry.config.fallback_seed for entry in schedule]

        assert len(set(agent_seeds)) == len(schedule)
        assert len(set(fallback_seeds)) == len(schedule)
        assert len({entry.deal_seed for entry in schedule}) == 8

    def test_timing_settings_do_not_disturb_the_deals(self) -> None:
        """Changing a budget re-runs the same deals, which is the point of them."""
        slow = build_schedule(participants(*PAIR), GauntletConfig(deals=4, seconds_per_turn=5.0))
        fast = build_schedule(participants(*PAIR), GauntletConfig(deals=4, seconds_per_turn=0.5))

        assert [entry.deal_seed for entry in slow] == [entry.deal_seed for entry in fast]
        assert [entry.config.agent_seed for entry in slow] == [
            entry.config.agent_seed for entry in fast
        ]

    def test_generation_is_stable(self) -> None:
        """The same arguments always describe the same experiment."""
        config = GauntletConfig(deals=5, seed=99, seconds_per_turn=0.25)

        assert build_schedule(participants(*PAIR), config) == build_schedule(
            participants(*PAIR), config
        )

    def test_repeated_kinds_are_distinguishable_participants(self) -> None:
        """A mirror match still has two names to report results under."""
        lineup = participants("greedy", "greedy")

        assert [spec.name for spec in lineup] == ["greedy-0", "greedy-1"]
        assert len(build_schedule(lineup, GauntletConfig(deals=2))) == 4

    def test_duplicate_labels_are_refused(self) -> None:
        """Two participants sharing a label would pool their results silently."""
        clash = (AgentSpec(kind="greedy", name="same"), AgentSpec(kind="random", name="same"))

        with pytest.raises(ValueError, match="labels must be distinct"):
            build_schedule(clash, GauntletConfig(deals=1))

    @pytest.mark.parametrize("agents", [1, 6])
    def test_unsupported_table_sizes_are_refused(self, agents: int) -> None:
        """The profile's range is checked before a match is ever built."""
        lineup = tuple(AgentSpec(kind="random", name=f"random-{index}") for index in range(agents))

        with pytest.raises(ValueError, match="supports 2-5 players"):
            build_schedule(lineup, GauntletConfig(deals=1))


class TestConfiguration:
    """What a run refuses to be configured with."""

    @pytest.mark.parametrize("deals", [0, -1])
    def test_a_deal_bank_must_hold_something(self, deals: int) -> None:
        """A gauntlet of no deals is a mistake, not an empty result."""
        with pytest.raises(ValueError, match="deals must be positive"):
            GauntletConfig(deals=deals)

    @pytest.mark.parametrize("budget", [0.0, -1.0, float("inf"), float("nan")])
    def test_the_budget_must_be_positive_and_finite(self, budget: float) -> None:
        """The match runner's rule, enforced before the first match is built."""
        with pytest.raises(ValueError, match="seconds_per_turn"):
            GauntletConfig(deals=1, seconds_per_turn=budget)

    def test_the_action_limit_must_be_positive(self) -> None:
        """A limit of zero would truncate every match before its first decision."""
        with pytest.raises(ValueError, match="max_play_decisions"):
            GauntletConfig(deals=1, max_play_decisions=0)


class TestRunning:
    """Playing a schedule, with a synchronous runner in place of the real one."""

    def test_every_scheduled_match_is_played_in_order(self) -> None:
        """The run's records line up with the schedule, one for one."""
        lineup = participants(*PAIR)
        config = GauntletConfig(deals=3, seed=11, seconds_per_turn=0.5)

        run = run_gauntlet(lineup, config, runner_factory=sync_factory(seed=4))

        assert run.schedule == build_schedule(lineup, config)
        assert [record.scheduled for record in run.records] == list(run.schedule)
        assert all(
            record.result.metadata.deal_seed == record.scheduled.deal_seed for record in run.records
        )

    def test_rotations_of_one_deal_deal_the_same_cards(self) -> None:
        """The shared deck is what makes a rotated comparison paired."""
        lineup = participants(*PAIR)
        run = run_gauntlet(
            lineup, GauntletConfig(deals=1, seed=3), runner_factory=sync_factory(seed=2)
        )
        first, second = run.records

        assert first.result.initial_events == second.result.initial_events
        assert first.scheduled.seats != second.scheduled.seats

    def test_each_match_gets_its_own_agent_and_fallback_streams(self) -> None:
        """A runner is configured with this match's seeds and nobody else's."""
        run = run_gauntlet(
            participants(*PAIR), GauntletConfig(deals=4), runner_factory=sync_factory()
        )

        for record in run.records:
            assert record.result.metadata.config == record.scheduled.config

    def test_progress_is_reported_once_per_match_in_order(self) -> None:
        """A long run can say where it is without the gauntlet knowing about a console."""
        seen: list[int] = []
        run = run_gauntlet(
            participants(*PAIR),
            GauntletConfig(deals=2),
            runner_factory=sync_factory(),
            on_match=lambda record: seen.append(record.scheduled.index),
        )

        assert seen == [0, 1, 2, 3]
        assert len(run.records) == 4

    def test_truncated_matches_are_kept_rather_than_dropped(self) -> None:
        """An action limit stops a match; it does not remove it from the run."""
        run = run_gauntlet(
            participants(*PAIR),
            GauntletConfig(deals=2, max_play_decisions=3),
            runner_factory=sync_factory(),
        )
        report = run.report

        assert report.status.truncated == 4
        assert report.status.finished == 0
        assert all(record.result.outcome is None for record in run.records)
        assert all(record.winner is None for record in run.records)

    def test_a_strict_abort_is_recorded_as_a_failure(self) -> None:
        """A match nobody could play is reported, never counted as a loss."""
        run = run_gauntlet(
            participants(*PAIR),
            GauntletConfig(deals=1, strict_failures=True),
            runner_factory=sync_factory(failing=[0]),
        )
        report = run.report

        assert report.status.agent_failed == 2
        assert report.status.finished == 0
        assert all(agent.wins == Rate(0, 0) for agent in report.agents)
        assert [failure.status for failure in report.failures] == [MatchStatus.AGENT_FAILED] * 2


class TestAccounting:
    """Aggregation over records built by hand, so every status can appear."""

    def setup_method(self) -> None:
        """Lay out a two-agent schedule the accounting tests share."""
        self.lineup = participants(*PAIR)
        self.config = GauntletConfig(deals=3, seed=1)
        self.schedule = build_schedule(self.lineup, self.config)

    def mixed_records(self) -> list[MatchRecord]:
        """Build one run holding every status a match can end in.

        Returns:
            Six records: two finished with different winners, one truncated,
            one aborted on an agent failure, one stopped by the engine, and one
            more finished match, so no status is a special case of position.
        """
        entries = self.schedule
        return [
            synthetic(entries[0], self.lineup, MatchStatus.FINISHED, winner_seat=0),
            synthetic(entries[1], self.lineup, MatchStatus.FINISHED, winner_seat=0),
            synthetic(entries[2], self.lineup, MatchStatus.TRUNCATED),
            synthetic(entries[3], self.lineup, MatchStatus.AGENT_FAILED),
            synthetic(entries[4], self.lineup, MatchStatus.ENGINE_FAILED),
            synthetic(entries[5], self.lineup, MatchStatus.FINISHED, winner_seat=1),
        ]

    def test_every_status_is_counted_and_the_totals_close(self) -> None:
        """The four statuses add up to the number scheduled, always."""
        report = summarize(self.lineup, self.config, self.mixed_records())
        status = report.status

        assert (status.finished, status.truncated) == (3, 1)
        assert (status.agent_failed, status.engine_failed) == (1, 1)
        assert status.failed == 2
        assert status.played == status.scheduled == 6

    def test_a_run_that_lost_a_match_is_refused(self) -> None:
        """Accounting that does not close is a bug, not a rounding difference."""
        with pytest.raises(ValueError, match="Status counts sum to 5"):
            summarize(self.lineup, self.config, self.mixed_records()[:5], scheduled=6)

    def test_wins_are_measured_against_finished_matches_only(self) -> None:
        """Truncated and aborted matches are in neither the top nor the bottom."""
        report = summarize(self.lineup, self.config, self.mixed_records())
        by_name = {agent.spec.name: agent for agent in report.agents}

        # Rotation 0 seats random-0 in seat 0 and rotation 1 seats greedy-1
        # there, so the two seat-0 wins go one each; the third finished match
        # was won from seat 1 of a rotation 1, which is random-0 again.
        assert by_name["random-0"].wins == Rate(2, 3)
        assert by_name["greedy-1"].wins == Rate(1, 3)
        assert sum(agent.wins.count for agent in report.agents) == report.status.finished

    def test_seat_breakdowns_add_up_to_the_overall_record(self) -> None:
        """The per-seat denominators partition a participant's finished matches."""
        report = summarize(self.lineup, self.config, self.mixed_records())

        for agent in report.agents:
            assert sum(rate.count for rate in agent.by_seat) == agent.wins.count
            assert sum(rate.total for rate in agent.by_seat) == agent.wins.total

    def test_a_participant_without_finished_matches_has_no_win_rate(self) -> None:
        """An empty denominator reports as unmeasured rather than as zero."""
        records = [synthetic(entry, self.lineup, MatchStatus.TRUNCATED) for entry in self.schedule]
        report = summarize(self.lineup, self.config, records)

        assert all(agent.wins == Rate(0, 0) for agent in report.agents)
        assert all(agent.wins.value is None for agent in report.agents)

    def test_head_to_head_rows_carry_the_same_denominator(self) -> None:
        """The opponent breakdown is the overall record for one fixed lineup."""
        report = summarize(self.lineup, self.config, self.mixed_records())

        for agent in report.agents:
            assert len(agent.by_opponents) == 1
            opponents, rate = agent.by_opponents[0]
            assert agent.spec.name not in opponents
            assert rate == agent.wins

    def test_failures_and_truncations_are_named_individually(self) -> None:
        """Every match that did not finish can be looked up, not just counted."""
        report = summarize(self.lineup, self.config, self.mixed_records())

        assert [failure.index for failure in report.failures] == [2, 3, 4]
        assert [failure.status for failure in report.failures] == [
            MatchStatus.TRUNCATED,
            MatchStatus.AGENT_FAILED,
            MatchStatus.ENGINE_FAILED,
        ]
        assert all(failure.detail for failure in report.failures)

    def test_decision_diagnostics_count_every_match(self) -> None:
        """A decision taken in a match that later failed still happened."""
        records = [
            synthetic(
                self.schedule[0],
                self.lineup,
                MatchStatus.FINISHED,
                winner_seat=0,
                turns=[turn(0, 0, seconds=0.2), turn(1, 1, fallback=True, seconds=0.4)],
            ),
            synthetic(
                self.schedule[1],
                self.lineup,
                MatchStatus.ENGINE_FAILED,
                turns=[turn(0, 0, rejected=2, crashed=True, seconds=0.6)],
            ),
        ]
        report = summarize(self.lineup, self.config, records, scheduled=2)
        stats = report.selection

        assert stats.decisions == 3
        assert stats.fallbacks == Rate(1, 3)
        assert stats.rejections == Rate(2, 3)
        assert stats.crashes == Rate(1, 3)
        assert stats.selection_seconds.median == pytest.approx(0.4)
        assert stats.selection_seconds.mean == pytest.approx(0.4)

    def test_diagnostics_are_attributed_to_the_participant_that_decided(self) -> None:
        """Rotation means a seat is not an agent; the record follows the agent."""
        records = [
            synthetic(
                self.schedule[0],  # rotation 0: seat 0 is random-0.
                self.lineup,
                MatchStatus.FINISHED,
                winner_seat=0,
                turns=[turn(0, 0, fallback=True), turn(1, 1)],
            ),
            synthetic(
                self.schedule[1],  # rotation 1: seat 0 is greedy-1.
                self.lineup,
                MatchStatus.FINISHED,
                winner_seat=0,
                turns=[turn(0, 0, fallback=True), turn(1, 1)],
            ),
        ]
        report = summarize(self.lineup, self.config, records, scheduled=2)
        by_name = {agent.spec.name: agent for agent in report.agents}

        assert by_name["random-0"].selection.fallbacks == Rate(1, 2)
        assert by_name["greedy-1"].selection.fallbacks == Rate(1, 2)

    def test_play_decisions_are_summarized_over_every_match(self) -> None:
        """Match length is reported for the whole run, failures included."""
        records = [
            synthetic(entry, self.lineup, MatchStatus.FINISHED, winner_seat=0, play_decisions=size)
            for entry, size in zip(self.schedule, [10, 20, 30, 40, 50, 60], strict=True)
        ]
        report = summarize(self.lineup, self.config, records)

        assert report.play_decisions.count == 6
        assert report.play_decisions.mean == pytest.approx(35.0)
        assert report.play_decisions.median == pytest.approx(35.0)


class TestSmallValues:
    """The two value types the report is built out of."""

    def test_an_empty_denominator_has_no_rate(self) -> None:
        """Nothing measured is reported as nothing, never as zero."""
        assert Rate(0, 0).value is None
        assert Rate(1, 4).value == pytest.approx(0.25)

    def test_an_empty_sample_has_no_statistics(self) -> None:
        """A distribution over nothing reports its emptiness."""
        empty = Distribution.of([])

        assert (empty.count, empty.mean, empty.median) == (0, None, None)

    def test_a_status_tally_must_close(self) -> None:
        """Counts that do not add up are rejected where they are built."""
        with pytest.raises(ValueError, match="Status counts sum to 3"):
            StatusCounts(scheduled=4, finished=2, truncated=1, agent_failed=0, engine_failed=0)


class TestConsole:
    """The compact table the command prints."""

    def test_the_summary_reports_every_status_and_denominator(self) -> None:
        """A reader sees what was scheduled, what finished, and out of how many."""
        lineup = participants(*PAIR)
        config = GauntletConfig(deals=1, seed=2)
        schedule = build_schedule(lineup, config)
        records = [
            synthetic(schedule[0], lineup, MatchStatus.FINISHED, winner_seat=1, turns=[turn(0, 0)]),
            synthetic(schedule[1], lineup, MatchStatus.TRUNCATED, turns=[turn(0, 1)]),
        ]

        text = "\n".join(gauntlet_summary(summarize(lineup, config, records)))

        assert "1 deals x 2 rotations = 2 matches" in text
        assert (
            "1 finished | 1 truncated | 0 agent failed | 0 engine failed (of 2 scheduled)" in text
        )
        assert "0/1 (0.000)" in text  # random-0 lost its one finished match.
        assert "1/1 (1.000)" in text  # greedy-1 won it.
        assert "matches that did not finish (1):" in text
        assert "truncated" in text

    def test_progress_names_the_seating_and_the_ending(self) -> None:
        """The line a long run prints says who sat where and what happened."""
        lineup = participants(*PAIR)
        schedule = build_schedule(lineup, GauntletConfig(deals=1))
        record = synthetic(schedule[1], lineup, MatchStatus.FINISHED, winner_seat=0)

        line = gauntlet_progress(record, len(schedule), lineup)

        assert line.startswith("match 2/2 (deal 0, rotation 1)")
        assert "seat 0: greedy-1" in line
        assert "greedy-1 wins" in line

    def test_progress_of_an_unfinished_match_names_no_winner(self) -> None:
        """A truncated match reports its status where a winner would be."""
        lineup = participants(*PAIR)
        schedule = build_schedule(lineup, GauntletConfig(deals=1))
        record = synthetic(schedule[0], lineup, MatchStatus.TRUNCATED)

        assert "truncated" in gauntlet_progress(record, len(schedule), lineup)


class TestDocument:
    """The JSON report, and the replays it embeds."""

    def run(self) -> GauntletRun:
        """Play a tiny synchronous gauntlet to have a document to encode.

        Returns:
            The completed run: one deal, two rotations, decided without workers.
        """
        config = GauntletConfig(deals=1, seed=8)
        return run_gauntlet(participants(*PAIR), config, runner_factory=sync_factory(seed=1))

    def test_the_document_carries_the_report_and_one_entry_per_match(self) -> None:
        """The file says what was run, how it went, and what each match did."""
        document: Any = gauntlet_document(self.run())

        assert document["schema"] == 1
        assert document["rules"] == DEFAULT_RULES.id
        assert [spec["name"] for spec in document["participants"]] == ["random-0", "greedy-1"]
        assert len(document["matches"]) == 2
        assert document["report"]["status"]["scheduled"] == 2

    def test_each_match_embeds_a_real_replay(self) -> None:
        """The gauntlet reuses the match format rather than inventing a weaker one."""
        document: Any = gauntlet_document(self.run())

        for entry in document["matches"]:
            replay = decode_replay(entry["replay"])
            assert verify_replay(replay).ok

    def test_replays_can_be_left_out(self) -> None:
        """A caller that only wants the aggregate does not pay for the games."""
        document: Any = gauntlet_document(self.run(), replays=False)

        assert all("replay" not in entry for entry in document["matches"])
        assert all(entry["status"] for entry in document["matches"])

    def test_rates_keep_their_denominators_in_json(self) -> None:
        """A machine reader gets the same explicit denominators the console does."""
        document: Any = gauntlet_document(self.run(), replays=False)
        wins = document["report"]["agents"][0]["wins"]

        assert set(wins) == {"count", "total", "rate"}
        assert wins["total"] == document["report"]["status"]["finished"]


def run_command(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the gauntlet command from the repository root.

    Args:
        *arguments: Command-line arguments to pass it.

    Returns:
        The finished process, with its output captured as text.
    """
    # A fixed argument vector, no shell, and the project's own script.
    return subprocess.run(
        [sys.executable, "scripts/gauntlet.py", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Any]:
    """Run one small real gauntlet through the documented command.

    Real workers, real timed decisions, and the file the command writes. Every
    test that needs an actually-played gauntlet shares this one run, because it
    is the only thing in this module that costs seconds.

    Args:
        tmp_path_factory: Where to put the report file.

    Returns:
        The command's standard output and the decoded report document.
    """
    output = tmp_path_factory.mktemp("results") / "gauntlet.json"
    finished = run_command(
        "--agents",
        "random",
        "greedy",
        "--deals",
        "1",
        "--seed",
        "42",
        "--seconds-per-turn",
        "0.5",
        "--output",
        str(output),
    )
    assert finished.returncode == 0, finished.stderr
    return finished.stdout, json.loads(output.read_text(encoding="utf-8"))


class TestCommand:
    """``scripts/gauntlet.py``, run as documented."""

    def test_a_baseline_gauntlet_runs_and_reports(self, baseline: tuple[str, Any]) -> None:
        """One deal, two rotations, and a table that accounts for both matches."""
        stdout, document = baseline

        assert "1 deals x 2 rotations = 2 matches" in stdout
        assert "wins among finished matches:" in stdout
        assert document["report"]["status"]["scheduled"] == 2
        assert len(document["matches"]) == 2

    def test_every_finished_match_replays_from_the_saved_file(
        self, baseline: tuple[str, Any]
    ) -> None:
        """The acceptance criterion: the file is enough to check the games.

        Verification deals the recorded deck and applies the recorded moves
        through the engine. No agent is built and no worker is started, so it
        reproduces exactly even though the matches themselves were timed.
        """
        _, document = baseline
        finished = [
            entry for entry in document["matches"] if entry["status"] == MatchStatus.FINISHED.value
        ]

        assert finished, "the baseline gauntlet finished no match to verify"
        for entry in finished:
            check = verify_replay(decode_replay(entry["replay"]))
            assert check.ok, check.problems
            assert check.outcome is not None
            assert entry["winner"] == entry["seats"][check.outcome.winner]

    def test_the_accounting_matches_the_recorded_matches(self, baseline: tuple[str, Any]) -> None:
        """The report's status counts are the statuses in the file, not a claim."""
        _, document = baseline
        status = document["report"]["status"]
        recorded = [entry["status"] for entry in document["matches"]]

        assert status["finished"] == recorded.count("finished")
        assert status["truncated"] == recorded.count("truncated")
        assert (
            status["finished"]
            + status["truncated"]
            + status["agent_failed"]
            + status["engine_failed"]
            == status["scheduled"]
            == len(recorded)
        )

    def test_progress_goes_to_standard_error(self) -> None:
        """The table stays on standard output while a run narrates itself."""
        finished = run_command(
            "--agents", "random", "random", "--deals", "1", "--max-play-decisions", "2"
        )

        assert finished.returncode == 0, finished.stderr
        assert "match 1/2" in finished.stderr
        assert "match 1/2" not in finished.stdout
        assert "2 truncated" in finished.stdout

    def test_quiet_silences_the_progress_only(self) -> None:
        """``--quiet`` removes the narration and nothing else."""
        finished = run_command(
            "--agents", "random", "random", "--deals", "1", "--max-play-decisions", "2", "--quiet"
        )

        assert finished.returncode == 0, finished.stderr
        assert finished.stderr == ""
        assert "2 matches" in finished.stdout

    @pytest.mark.parametrize(
        "arguments",
        [
            ("--agents", "random"),
            ("--agents", "random", "greedy", "--dealer", "5"),
        ],
        ids=["one-agent", "dealer-off-the-table"],
    )
    def test_arguments_describing_no_runnable_gauntlet_exit_two(
        self, arguments: tuple[str, ...]
    ) -> None:
        """Decoded input is validated before any match is played."""
        finished = run_command(*arguments)

        assert finished.returncode == 2
        assert finished.stderr.startswith("error:")

    @pytest.mark.parametrize(
        "arguments",
        [
            ("--agents", "random", "greedy", "--deals", "0"),
            ("--agents", "random", "greedy", "--seconds-per-turn", "inf"),
            ("--agents", "random", "greedy", "--seconds-per-turn", "0"),
            ("--agents", "random", "psychic"),
        ],
        ids=["no-deals", "infinite-budget", "zero-budget", "unknown-kind"],
    )
    def test_argparse_refuses_malformed_options(self, arguments: tuple[str, ...]) -> None:
        """Counts, budgets, and kinds are checked by the parser itself."""
        assert run_command(*arguments).returncode == 2
