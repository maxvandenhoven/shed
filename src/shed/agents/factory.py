"""The serializable participant specification and the built-in agent factory.

Agents are built fresh for every decision from an :class:`AgentSpec` and an
explicit seed, so no live object and no generator state ever crosses a process
boundary. That is what keeps repeated construction from replaying the same
random stream, and it is why the specification carries no seed of its own.

The factory sits above the strategies: it imports every built-in agent at module
scope, which is exactly why it is not part of :mod:`shed.agents.base`. A
strategy importing its base class therefore never reaches the factory, and no
import has to be deferred into a function body to break a cycle.
"""

from dataclasses import dataclass

from shed.agents.base import Agent
from shed.agents.greedy import GreedyAgent
from shed.agents.random import RandomAgent

__all__ = ["AGENT_KINDS", "AgentSpec", "build_agent"]

AGENT_KINDS: tuple[str, ...] = ("greedy", "random")
"""Kinds :func:`build_agent` can build, sorted for stable error messages."""


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

        Validating here rather than in :func:`build_agent` means a mistyped
        lineup fails where it is written, not inside a worker process at
        decision time.

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
    match spec.kind:
        case "greedy":
            return GreedyAgent(seed=seed)
        case "random":
            return RandomAgent(seed=seed)
        case _:
            raise ValueError(f"No builder for agent kind {spec.kind!r}")
