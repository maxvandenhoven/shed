"""Agent interface, turn protocol, agent specification, and the built-in factory.

An agent sees exactly two things: one immutable :class:`~shed.engine.PlayerView`
and a turn-scoped capability. It never receives the authoritative ``GameState``,
the deck seed, the fallback seed, the match runner, or another seat's view, so a
strategy can only depend on the observation, the public rules profile that
observation carries, and its own independent seed.

Legal choices come from ``view.legal_moves``. The engine is the single legality
authority: an agent never recomputes legality, never needs a rules object of its
own, and works with typed :class:`~shed.engine.Move` objects that it neither
encodes nor decodes. The turn is only a channel -- submit a candidate, ask how
much time is left -- and the runner stays free to reject anything illegal.

Agents are built fresh for every decision from a serializable
:class:`AgentSpec` and an explicit seed, so no live object and no generator
state ever crosses a process boundary. That is what keeps repeated construction
from replaying the same random stream, and it is why the specification carries
no seed of its own.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from shed.engine import Move, PlayerView

__all__ = [
    "AGENT_KINDS",
    "Agent",
    "AgentSpec",
    "TurnContext",
    "build_agent",
    "legal_choices",
]

AGENT_KINDS: tuple[str, ...] = ("greedy", "random")
"""Kinds :func:`build_agent` can build, sorted for stable error messages."""


class TurnContext(Protocol):
    """The whole capability an agent holds during one decision.

    The protocol is deliberately tiny: a way to submit candidates and a way to
    ask how much time is left. It exposes no legal moves -- those live on the
    view -- no game state, and no way to observe what the runner selected.

    Submissions are fire-and-forget: nothing is acknowledged, and the runner
    remains the legality authority, so a submitted move is a proposal rather
    than a decision. The latest legal candidate wins unless a final submission
    closes the turn first.
    """

    def remaining_seconds(self) -> float:
        """Return the time remaining in this decision's budget.

        Returns:
            Seconds until the decision's deadline, clamped to zero. This value
            is a cooperative hint; the match runner enforces the deadline.
        """

    def submit(self, move: Move, *, final: bool = False) -> None:
        """Offer one candidate move to the runner.

        Args:
            move: The candidate. It should be one of ``view.legal_moves``; an
                illegal candidate is rejected and leaves the previous one
                standing.
            final: Whether this candidate closes the turn. A final submission
                is the last one the runner accepts, so submit it only when the
                agent is done improving.
        """


class Agent(ABC):
    """Base class for every strategy.

    A subclass implements :meth:`think` and nothing else is required of it. The
    base class owns only the agent's dedicated generator, because every strategy
    needs seeded randomness somewhere -- tie-breaking at minimum -- and the
    factory always supplies a seed for it.

    Attributes:
        _rng: This agent's generator. It is independent of the deck and fallback
            streams and is never seeded from module-global randomness.
    """

    def __init__(self, *, seed: int) -> None:
        """Construct an agent with its own independent generator.

        Args:
            seed: Seed for this agent's generator. The caller draws a fresh one
                per decision; it is never the deck seed or the fallback seed.
        """
        self._rng = random.Random(seed)

    @abstractmethod
    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Decide, submitting candidates until one of them is final.

        An implementation should submit a cheap legal baseline immediately and
        then improve on it, submitting each improvement, because the runner
        keeps the latest legal candidate and may close the turn at any moment.
        A strategy that has nothing to improve simply submits once with
        ``final=True``.

        Args:
            view: This seat's immutable observation, including the legal moves
                the engine generated for it.
            turn: The submission channel for this decision.
        """


def legal_choices(view: PlayerView) -> tuple[Move, ...]:
    """Return the moves a view offers, refusing an empty tuple.

    Agents read legality from the observation rather than deriving it, so this
    is the one place the shared "there must be something to choose" check lives.

    Args:
        view: The observation handed to :meth:`Agent.think`.

    Returns:
        ``view.legal_moves``, already ordered deterministically by the engine.

    Raises:
        ValueError: If the view offers nothing, which means the agent was asked
            to decide for a seat that is not the current actor.
    """
    if not view.legal_moves:
        raise ValueError(f"Player {view.viewer} was asked to decide with no legal moves")
    return view.legal_moves


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """A serializable description of one participant.

    Specifications travel to workers and into results; live agents never do. The
    spec deliberately carries no seed: seeds are drawn per decision and recorded
    by the runner, so replaying a spec cannot resurrect a stale generator, and a
    deck seed can never reach an agent by riding along in its configuration.

    Typed configuration fields belong here only once an implemented agent needs
    them.

    Attributes:
        kind: Which built-in strategy to build; one of :data:`AGENT_KINDS`.
        name: Stable label used in results and console output. Repeated kinds
            get distinct labels so a lineup can hold two of the same strategy.
    """

    kind: str
    name: str

    def __post_init__(self) -> None:
        """Check the kind is buildable and the label is usable.

        Validating here rather than in the factory means a mistyped lineup fails
        where it is written, not inside a worker process at decision time.

        Raises:
            ValueError: If the kind is unknown or the name is empty.
        """
        if self.kind not in AGENT_KINDS:
            raise ValueError(
                f"Unknown agent kind {self.kind!r}; expected one of {', '.join(AGENT_KINDS)}"
            )
        if not self.name:
            raise ValueError("AgentSpec.name must be a non-empty label")


def build_agent(spec: AgentSpec, *, seed: int) -> Agent:
    """Build a fresh agent for one decision.

    Construction is explicit rather than discovered: the built-in kinds are
    listed here, and adding a strategy means adding a branch and a kind.

    Args:
        spec: The participant to build. Its kind was already validated when the
            specification was created.
        seed: Seed for the new agent's generator. Pass a fresh value per
            decision -- reusing one replays the same random stream -- and never
            pass the deck or fallback seed.

    Returns:
        A newly constructed agent that shares no state with any previous one.

    Raises:
        ValueError: If the kind is not one this factory builds, which can only
            happen if :data:`AGENT_KINDS` grew without a branch here.
    """
    # Imported inside the factory so the baselines can import this module for
    # their base class without an import cycle. This is the only caller.
    from shed.agents.greedy import GreedyAgent
    from shed.agents.random import RandomAgent

    match spec.kind:
        case "greedy":
            return GreedyAgent(seed=seed)
        case "random":
            return RandomAgent(seed=seed)
        case _:
            raise ValueError(f"No builder for agent kind {spec.kind!r}")
