"""The uniform random baseline.

The agent exists to give the gauntlet a floor to measure against and to exercise
every branch of the engine: sampling uniformly from the engine's own legal-move
tuple eventually reaches arrangements, hand and face-up batches, blind reveals,
and forced pickups.
"""

from shed.agents.base import Agent, TurnContext
from shed.engine import PlayerView

__all__ = ["RandomAgent"]


class RandomAgent(Agent):
    """Samples one legal move uniformly and finalizes immediately.

    The sample is uniform over the *rank/count* actions the engine offers, not
    over physical card subsets, so equivalent suit permutations cannot
    overweight a rank the way enumerating subsets would.

    There is nothing to improve after the sample, so the single submission is
    final and the decision closes without consulting the clock at all.
    """

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Sample one legal move uniformly and submit it as final.

        Args:
            view: This seat's observation; its ``legal_moves`` is the sample
                space, and the agent derives no legality of its own.
            turn: The submission channel for this decision.

        Raises:
            ValueError: If the view offers no legal moves, which means this seat
                is not the current actor.
        """
        if not view.legal_moves:
            raise ValueError(f"Player {view.viewer} was asked to decide with no legal moves")
        turn.submit(self._rng.choice(view.legal_moves), final=True)
