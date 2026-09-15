"""Tests that every agent the package ships can advise on an observed game.

The companion hands an agent a :class:`~shed.engine.PlayerView` it built from
observations rather than from a ``GameState``, so the question these tests exist to
answer is whether that view is enough. Two of them answer it directly: one runs
every kind in :data:`~shed.agents.AGENT_KINDS` over positions covering all three
zones and a forced pickup, and one traces which view fields the agents actually
read and fails if any of them reaches outside
:data:`~shed.companion.advice.FAITHFUL_VIEW_FIELDS`.

That second test is the one that matters for a *future* agent. A companion-built
view is thinner than the engine's in ways no amount of care can fix -- an observed
game does not know the pile cards nobody saw -- so an agent that starts reading
``discard_pile`` or ``history`` will still run, but it will be reading a weaker
truth than its author assumed. The test turns that from a surprise into a failure
with a message.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from shed.agents import AGENT_KINDS, AgentSpec, build_agent
from shed.companion.advice import (
    AGENT_PROFILES,
    DEFAULT_AGENT,
    DEFAULT_CHOICE,
    FAITHFUL_VIEW_FIELDS,
    AgentChoice,
    agent_catalogue,
    build_player_view,
    profile_for,
    recommend,
)
from shed.companion.observed import (
    ME,
    AtLeast,
    ObservationError,
    ObservedState,
    PickUp,
    Rank,
    observed_legal_moves,
)
from shed.engine import Move, PlayerView
from tests.companion.conftest import craft


def _positions() -> list[ObservedState]:
    """Build one position per kind of decision an agent can be asked for.

    Returns:
        A hand batch, a face-up batch, a blind reveal, and a forced pickup. Setup
        arrangements are deliberately absent: the companion joins after the
        physical hand/table swap, so no agent is ever asked to arrange.
    """
    return [
        craft(
            my_hand=(Rank.THREE, Rank.SEVEN, Rank.TEN),
            my_face_down=2,
            opponent_hand_unknown=3,
            opponent_face_down=2,
            deck_count=10,
        ),
        craft(
            my_hand=(Rank.FIVE, Rank.FIVE, Rank.ACE),
            my_face_down=1,
            opponent_hand_unknown=2,
            deck_count=6,
        ),
        craft(
            my_face_up=(Rank.FOUR, Rank.NINE),
            my_face_down=1,
            opponent_hand_unknown=2,
            deck_count=0,
        ),
        craft(my_face_down=3, opponent_hand_unknown=2, deck_count=0),
        craft(
            my_hand=(Rank.THREE,),
            my_face_down=1,
            opponent_hand_unknown=2,
            pile=(Rank.KING,),
            constraint=AtLeast(Rank.KING),
            deck_count=0,
        ),
    ]


@pytest.mark.parametrize("kind", AGENT_KINDS)
def test_every_shipped_agent_advises_legally_on_every_kind_of_decision(kind: str) -> None:
    """A companion-built view is enough for every agent the package builds."""
    choice = AgentChoice(spec=AgentSpec(kind=kind, name=kind))
    for index, state in enumerate(_positions()):
        recommendation = recommend(state, choice=choice, seed=index)
        assert recommendation.move in observed_legal_moves(state, ME)
        assert recommendation.agent.spec.kind == kind
        assert recommendation.caveat


class _Collector:
    """A turn context that keeps whatever an agent submits.

    Written here rather than reusing the companion's own recorder so this test
    probes the agent through the published protocol only.

    Attributes:
        chosen: The last move submitted, if any.
    """

    def __init__(self) -> None:
        """Start with nothing submitted."""
        self.chosen: Move | None = None

    def remaining_seconds(self) -> float:
        """Return a fixed budget hint.

        Returns:
            One second. Nothing here enforces a deadline.
        """
        return 1.0

    def submit(self, move: Move, *, final: bool = False) -> None:
        """Keep the submitted move.

        Args:
            move: The candidate.
            final: Whether it closes the decision; unused, the last one wins.
        """
        self.chosen = move


@pytest.mark.parametrize("kind", AGENT_KINDS)
def test_no_shipped_agent_reads_a_view_field_the_companion_cannot_fill(kind: str) -> None:
    """The compatibility claim, checked rather than asserted.

    A companion-built view is honest about the fields in ``FAITHFUL_VIEW_FIELDS``
    and thinner than the engine's everywhere else. An agent that reads outside that
    set is not broken, but it is reading an observed game as though it were an
    omniscient one, and whoever adds it should decide that deliberately.
    """
    seen: set[str] = set()

    class _Tracing(PlayerView):
        """A view that records which of its fields were read."""

        def __getattribute__(self, name: str) -> Any:
            """Record a public attribute read, then serve it.

            Args:
                name: The attribute being read.

            Returns:
                The attribute's value, unchanged.
            """
            if not name.startswith("_"):
                seen.add(name)
            return object.__getattribute__(self, name)

    for index, state in enumerate(_positions()):
        honest = build_player_view(state, ME)
        traced = _Tracing(**{item.name: getattr(honest, item.name) for item in fields(PlayerView)})
        collector = _Collector()
        build_agent(AgentSpec(kind=kind, name=kind), seed=index).think(traced, collector)
        assert collector.chosen in traced.legal_moves

    # `me` is a property over `players` and `viewer`, both of which are faithful,
    # and `legal_moves` is read by the assertion above as well as by the agent.
    unfaithful = seen - FAITHFUL_VIEW_FIELDS - {"me"}
    assert not unfaithful, (
        f"the {kind} agent reads {sorted(unfaithful)}, which a companion-built view "
        "cannot fill as faithfully as the engine does; see FAITHFUL_VIEW_FIELDS"
    )


def test_the_catalogue_covers_every_kind_the_package_builds() -> None:
    """The picker is driven by the package, so it cannot fall behind it."""
    assert tuple(profile.kind for profile in agent_catalogue()) == AGENT_KINDS
    assert DEFAULT_AGENT in AGENT_KINDS


def test_an_agent_without_a_written_profile_still_gets_one() -> None:
    """A new kind works immediately; only its description has to be written."""
    described = set(AGENT_PROFILES)
    undescribed = [kind for kind in AGENT_KINDS if kind not in described]
    for kind in undescribed:  # Empty today; this is the guard for when it is not.
        profile = profile_for(kind)
        assert profile.label
        assert "no description" in profile.summary.lower()
    assert described <= set(AGENT_KINDS)


def test_a_kind_the_package_does_not_build_is_refused() -> None:
    """The picker's contents come from ``AGENT_KINDS``; anything else is an error."""
    with pytest.raises(ObservationError, match="not an agent this release ships"):
        profile_for("minimax")


def test_each_strategy_is_explained_in_its_own_terms() -> None:
    """A suggestion with no reasoning behind it must not read like one that has some."""
    state = craft(
        my_hand=(Rank.FIVE, Rank.FIVE, Rank.ACE),
        my_face_down=1,
        opponent_hand_unknown=2,
        deck_count=6,
    )
    greedy = recommend(state, choice=AgentChoice(AgentSpec("greedy", "greedy")), seed=1)
    random = recommend(state, choice=AgentChoice(AgentSpec("random", "random")), seed=1)
    assert "retention score" in greedy.reasoning or "only rank" in greedy.reasoning
    assert "sampled uniformly" in random.reasoning
    assert "compared nothing" in random.reasoning
    assert "not optimal play" in greedy.caveat
    assert "not advice" in random.caveat


def test_a_position_explanation_is_shared_by_every_strategy() -> None:
    """A forced pickup is explained by the rules, not by whoever chose it."""
    state = craft(
        my_hand=(Rank.THREE,),
        my_face_down=1,
        opponent_hand_unknown=2,
        pile=(Rank.KING,),
        constraint=AtLeast(Rank.KING),
        deck_count=0,
    )
    for kind in AGENT_KINDS:
        recommendation = recommend(state, choice=AgentChoice(AgentSpec(kind, kind)), seed=2)
        assert recommendation.move == PickUp()
        assert "only legal action" in recommendation.reasoning


def test_the_default_choice_is_the_shedding_baseline() -> None:
    """An operator who never opens the picker gets the useful suggestion."""
    assert DEFAULT_CHOICE.spec.kind == DEFAULT_AGENT == "greedy"
    state = craft(my_hand=(Rank.THREE,), opponent_hand_unknown=2, deck_count=5)
    assert recommend(state, seed=1).agent.spec.kind == "greedy"


def test_a_seed_salt_changes_the_tie_break_without_changing_the_position() -> None:
    """The salt is a knob on an arbitrary choice, never on what is legal."""
    state = craft(my_face_down=3, opponent_hand_unknown=2, deck_count=0)
    plain = AgentChoice(AgentSpec("random", "random"))
    salted = AgentChoice(AgentSpec("random", "random"), seed=99)
    assert plain.seed_for(1234) == 1234
    assert salted.seed_for(1234) != 1234
    for choice in (plain, salted):
        assert recommend(state, choice=choice, seed=1234).move in observed_legal_moves(state, ME)


def test_a_seeded_choice_is_still_reproducible() -> None:
    """Two asks about one position agree, salt or no salt."""
    state = craft(
        my_hand=(Rank.THREE, Rank.SEVEN, Rank.TEN),
        my_face_down=1,
        opponent_hand_unknown=2,
        deck_count=5,
    )
    choice = AgentChoice(AgentSpec("random", "random"), seed=7)
    first = recommend(state, choice=choice, seed=555)
    second = recommend(state, choice=choice, seed=555)
    assert first.move == second.move
