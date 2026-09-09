"""Complete synchronous baseline matches driven straight through the engine.

The loop lives in ``tests.agents.conftest``: it creates a state, filters the
opening events per seat, observes the actor, builds a fresh agent from its
specification and a fresh seed, lets it think against an in-memory turn, applies
what it finalized, and appends the filtered transition events to every seat's
history. No timing, process, or runner code is involved, so these tests exercise
the agent interface itself rather than a scheduler.
"""

import pytest

from shed.agents import AgentSpec
from shed.engine import (
    Arrange,
    CardsDrawn,
    GameState,
    HandDealt,
    PickUp,
    Play,
    PlayerView,
    Reveal,
    Zone,
)
from tests.agents.conftest import MatchLog, play_baseline_match
from tests.conftest import assert_cards_conserved

RANDOM = AgentSpec(kind="random", name="random-1")
"""One uniform random participant."""

GREEDY = AgentSpec(kind="greedy", name="greedy-1")
"""One greedy participant."""

LINEUPS = {
    "random": (RANDOM,),
    "greedy": (GREEDY,),
    "mixed": (RANDOM, GREEDY),
}
"""Specification pools; a lineup repeats a pool until every seat is filled."""


def _lineup(pool: tuple[AgentSpec, ...], player_count: int) -> list[AgentSpec]:
    """Fill every seat from a pool of participants.

    Repeated kinds get distinct labels, as a real lineup would.

    Args:
        pool: Specifications to cycle through.
        player_count: Seats to fill.

    Returns:
        One specification per seat, in seat order.
    """
    return [
        AgentSpec(kind=pool[seat % len(pool)].kind, name=f"{pool[seat % len(pool)].kind}-{seat}")
        for seat in range(player_count)
    ]


def _assert_only_public_cards(state: GameState, view: PlayerView) -> None:
    """Assert the observation carries no card the viewer may not know.

    Args:
        state: The live authoritative state at this decision.
        view: The observation the actor was handed.

    Raises:
        AssertionError: If a hidden card, or the size of the draw pile's
            contents, leaked into the view.
    """
    hidden = {card.id for card in state.draw_pile}
    for seat in state.seat_order:
        player = state.players[seat]
        hidden.update(card.id for card in player.face_down.values())
        if seat != view.viewer:
            hidden.update(card.id for card in player.hand)

    visible = {card.id for card in (*view.hand, *view.discard_pile, *view.burned_cards)}
    for public in view.players:
        visible.update(card.id for card in public.face_up)

    assert not visible & hidden, "an observation exposed cards the viewer cannot know"
    assert view.draw_count == len(state.draw_pile)


def _play(
    pool: tuple[AgentSpec, ...],
    player_count: int,
    *,
    deal_seed: int,
    agent_seed: int = 0,
) -> MatchLog:
    """Play one match of the given lineup.

    Args:
        pool: Specifications to fill seats from.
        player_count: Seats at the table.
        deal_seed: Deck seed.
        agent_seed: Seed of the per-decision agent-seed stream.

    Returns:
        The match log.
    """
    return play_baseline_match(
        _lineup(pool, player_count),
        deal_seed=deal_seed,
        agent_seed=agent_seed,
        inspect=_assert_only_public_cards,
    )


def _assert_finished(log: MatchLog, player_count: int) -> None:
    """Assert a match ended properly and left the engine's promises intact.

    Args:
        log: The finished match.
        player_count: Seats at the table.

    Raises:
        AssertionError: If the match truncated, lost a card, or ended without a
            winner who shed everything.
    """
    assert not log.truncated, "a baseline match should finish inside the action bound"
    outcome = log.outcome
    assert outcome is not None
    assert log.state.players[outcome.winner].remaining_count == 0
    assert log.state.current_player is None
    assert_cards_conserved(log.state)

    setup_decisions = [entry for entry in log.decisions if isinstance(entry.move, Arrange)]
    assert len(setup_decisions) == player_count
    for entry in log.decisions:
        assert entry.move in entry.view.legal_moves
        assert [submission.final for submission in entry.submissions] == [True]


@pytest.mark.parametrize("pool", list(LINEUPS), ids=list(LINEUPS))
@pytest.mark.parametrize("player_count", [2, 3, 4, 5])
def test_baselines_finish_matches_at_every_table_size(pool: str, player_count: int) -> None:
    """Both baselines play legal, complete games for 2-5 players."""
    log = _play(LINEUPS[pool], player_count, deal_seed=2024 + player_count)

    _assert_finished(log, player_count)


@pytest.mark.parametrize("pool", list(LINEUPS), ids=list(LINEUPS))
def test_matches_replay_identically_from_the_same_seeds(pool: str) -> None:
    """A deal seed and an agent seed reproduce the whole decision sequence."""
    first = _play(LINEUPS[pool], 3, deal_seed=77, agent_seed=5)
    second = _play(LINEUPS[pool], 3, deal_seed=77, agent_seed=5)

    assert [entry.move for entry in first.decisions] == [entry.move for entry in second.decisions]
    assert [entry.seed for entry in first.decisions] == [entry.seed for entry in second.decisions]
    assert first.outcome == second.outcome


def test_a_different_agent_seed_changes_the_choices() -> None:
    """Agent randomness is seeded independently of the deal, and it matters."""
    first = _play(LINEUPS["random"], 3, deal_seed=77, agent_seed=5)
    second = _play(LINEUPS["random"], 3, deal_seed=77, agent_seed=6)

    assert [entry.move for entry in first.decisions] != [entry.move for entry in second.decisions]


def test_every_decision_is_built_on_its_own_seed() -> None:
    """Fresh per-decision seeds are what stop one stream repeating all match."""
    log = _play(LINEUPS["mixed"], 3, deal_seed=31)

    seeds = [entry.seed for entry in log.decisions]

    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("pool", list(LINEUPS), ids=list(LINEUPS))
def test_baselines_cover_every_kind_of_decision(pool: str) -> None:
    """Across a handful of deals the baselines meet every decision the engine asks.

    Arrangements, hand batches, face-up batches, blind reveals, and forced
    pickups are all reached, which is the coverage the interface has to support.
    """
    seen: set[str] = set()
    for deal_seed in range(12):
        log = _play(LINEUPS[pool], 3, deal_seed=deal_seed)
        _assert_finished(log, 3)
        for entry in log.decisions:
            match entry.move:
                case Arrange():
                    seen.add("arrange")
                case Play(source=Zone.HAND):
                    seen.add("hand")
                case Play(source=Zone.FACE_UP):
                    seen.add("face_up")
                case Reveal():
                    seen.add("reveal")
                case PickUp():
                    seen.add("pickup")

    assert seen == {"arrange", "hand", "face_up", "reveal", "pickup"}


def test_history_reaches_agents_with_private_identities_stripped() -> None:
    """Each seat's history keeps its own deals and draws, and only counts of others."""
    log = _play(LINEUPS["mixed"], 3, deal_seed=404)

    for seat, history in log.histories.items():
        private = [event for event in history if isinstance(event, HandDealt | CardsDrawn)]
        assert private, "every match deals and draws cards"
        for event in private:
            if event.player == seat:
                assert event.cards is not None
                assert len(event.cards) == event.count
            else:
                assert event.cards is None
                assert event.count > 0


def test_agents_receive_views_only_for_their_own_seat() -> None:
    """The actor decides on their own observation, never on somebody else's."""
    log = _play(LINEUPS["mixed"], 4, deal_seed=808)

    for entry in log.decisions:
        assert entry.view.viewer == entry.player
        assert entry.view.current_player == entry.player
