"""Public interface of the Shed agents package.

Agents depend on the engine's value types and on nothing else: no timing, no
processes, no serialization, and no authoritative state. Each strategy is
constructed fresh for a single decision from a serializable
:class:`AgentSpec` and an explicit seed, reads its options from the observation
it is given, and submits candidates through the turn it is handed.

Two baselines ship: :class:`RandomAgent` samples uniformly among the legal
actions, and :class:`GreedyAgent` sheds as much as it can as cheaply as it can.
Both cover every decision the engine can ask for -- arrangement, hand, face-up,
blind reveal, and forced pickup.
"""

from shed.agents.base import AGENT_KINDS, Agent, AgentSpec, TurnContext, build_agent
from shed.agents.greedy import RETENTION_SCORE, GreedyAgent
from shed.agents.random import RandomAgent

__all__ = [
    "AGENT_KINDS",
    "RETENTION_SCORE",
    "Agent",
    "AgentSpec",
    "GreedyAgent",
    "RandomAgent",
    "TurnContext",
    "build_agent",
]
