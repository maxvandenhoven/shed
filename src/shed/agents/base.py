"""The interface every strategy implements, and the channel it talks through.

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

This module is the root of the package: it imports engine types and nothing from
``shed.agents`` itself. Strategies import their base class from here, and
:mod:`shed.agents.factory` imports both this module and the strategies, so the
dependencies inside the package run one way and no import is deferred into a
function body.

``TurnContext`` is defined beside its *consumer* rather than beside an
implementation. The concrete pipe-backed context belongs to the match runner,
which already imports this package to build agents; declaring the protocol there
would make the two packages import each other.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Protocol

from shed.engine import Move, PlayerView

__all__ = ["Agent", "TurnContext"]


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
