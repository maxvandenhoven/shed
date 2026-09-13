"""The research agent: an evolving heuristic measured against the greedy baseline.

The strategy here is developed by experiment. Each version is benchmarked head to
head against :class:`~shed.agents.greedy.GreedyAgent` over a fixed bank of deals,
and only a version that scores better replaces the one before it, so the file
always holds the best measured heuristic rather than the most recent idea.

Arrangements are still scored with greedy's retention table. Plays are not:
greedy sorts them by ``(-count, retention)``, so batch size decides first and the
agent dumps three kings when one would do. This agent sorts them by
``(retention, -count)`` instead -- it picks the *cheapest rank* it can spend and
then spends every copy of it -- and settles ties with the seeded generator over
the engine's deterministically ordered move tuple.

The retention table is what makes that ordering sensible rather than merely
frugal. A high ordinary rank answers more constraints than a low one, since
``AtLeast(r)`` accepts everything from ``r`` up, and the four always-playable
ranks answer every constraint there is. Spending the cheapest rank therefore
keeps the most flexible cards in hand, which is what stops a turn from ending in
a forced pickup.

The agent decides in one pass and submits once, as final. That is not a
concession to the clock: the heuristic costs microseconds, so there is nothing to
improve with the rest of the budget, and a single final submission is what the
shared agent tests require of every strategy in the factory.
"""

from collections.abc import Mapping

from shed.agents.base import Agent, TurnContext
from shed.agents.greedy import RETENTION_SCORE
from shed.engine import Arrange, CardId, Move, PickUp, Play, PlayerView, Rank, Reveal

__all__ = ["ResearchAgent"]


def _own_rank_by_id(view: PlayerView) -> dict[CardId, Rank]:
    """Map the viewer's own visible cards to their ranks.

    Arrangements name card identifiers, so scoring one needs the rank behind
    each identifier. During setup the viewer's hand and face-up collection are
    exactly the six cards an arrangement may choose from.

    Args:
        view: The viewer's observation.

    Returns:
        Rank per card identifier, over the viewer's hand and face-up cards.
    """
    return {card.id: card.rank for card in (*view.hand, *view.me.face_up)}


def _score(move: Move, view: PlayerView, ranks: Mapping[CardId, Rank]) -> tuple[int, ...]:
    """Score one candidate move; smaller sorts better.

    Arrangements sort by the negated retention sum of the cards they leave face
    up, so the largest sum wins. Plays sort by the retention score of the rank
    spent first -- spend what is least useful to keep -- and then by negated
    batch size, so the whole of the chosen rank goes at once rather than one card
    at a time. Blind reveals and a forced pickup offer nothing to compare: every
    reveal hides the same unknown, and a pickup is the only action when it
    appears at all.

    Args:
        move: A candidate from the engine's legal-move tuple.
        view: The observation being decided on. Unused by this version, and
            named so a heuristic that reads the pile, the constraint, or the
            opponents has it without changing every call site.
        ranks: Rank per identifier for the viewer's own visible cards, needed
            only by arrangements.

    Returns:
        A sort key. Keys are only ever compared within one decision, where every
        candidate is the same kind of move and therefore has the same shape.

    Raises:
        ValueError: If the move is not a known move type.
    """
    del view
    match move:
        case Arrange(face_up_cards=card_ids):
            return (-sum(RETENTION_SCORE[ranks[card_id]] for card_id in card_ids),)
        case Play(rank=rank, count=count):
            return (RETENTION_SCORE[rank], -count)
        case Reveal() | PickUp():
            return ()
    raise ValueError(f"Unknown move {move!r}")


class ResearchAgent(Agent):
    """The experimental strategy, scored move by move over the legal tuple.

    The agent reads its options from ``view.legal_moves`` and derives no
    legality of its own, so a narrowed view narrows the agent. It decides in one
    pass and submits that decision as final.
    """

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Score every legal move, break ties with the seeded generator, submit.

        Args:
            view: This seat's observation; its ``legal_moves`` is the candidate
                set.
            turn: The submission channel for this decision.

        Raises:
            ValueError: If the view offers no legal moves, which means this seat
                is not the current actor.
        """
        turn.submit(self._choose(view), final=True)

    def _choose(self, view: PlayerView) -> Move:
        """Pick the best-scoring legal move, breaking ties with the generator.

        Args:
            view: This seat's observation.

        Returns:
            One of ``view.legal_moves``: the best-scoring candidate, or a
            uniform choice among the candidates tied for best.

        Raises:
            ValueError: If the view offers no legal moves.
        """
        moves = view.legal_moves
        if not moves:
            raise ValueError(f"Player {view.viewer} was asked to decide with no legal moves")
        ranks = _own_rank_by_id(view)
        scored = [(_score(move, view, ranks), move) for move in moves]
        best = min(key for key, _ in scored)
        # The tied list keeps the engine's move order, so the seeded choice
        # among equals is reproducible rather than dependent on iteration luck.
        tied = [move for key, move in scored if key == best]
        return self._rng.choice(tied)
