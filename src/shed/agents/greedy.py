"""The deterministic shedding heuristic, with seeded tie-breaking.

The heuristic is an explicit baseline, not a claim of good play: it keeps the
cards that are hardest to shed for last and spends the cheapest cards first.

Two numbers drive every decision, both from the fixed retention table below:
during setup a candidate face-up set is worth the sum of its retention scores,
and during play a batch is preferred by size first and then by how little the
cards it spends are worth keeping. Whatever is still tied afterwards is settled
by the agent's seeded generator over the engine's deterministically ordered
move tuple, so the choice is reproducible from the seed alone.
"""

from collections.abc import Mapping

from shed.agents.base import Agent, TurnContext, legal_choices
from shed.engine import Arrange, CardId, Move, PickUp, Play, PlayerView, Rank, Reveal

__all__ = ["RETENTION_SCORE", "GreedyAgent"]

RETENTION_SCORE: Mapping[Rank, int] = {
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 17,
    Rank.EIGHT: 8,
    Rank.NINE: 20,
    Rank.TEN: 23,
    Rank.JACK: 11,
    Rank.QUEEN: 12,
    Rank.KING: 13,
    Rank.ACE: 14,
    Rank.TWO: 21,
    Rank.JOKER: 22,
}
"""How much this baseline wants to keep a card of each rank.

Ordinary ranks score their numeric value. The cards with special powers score
above every ordinary rank because they can be played against any pile: seven 17,
nine 20, two 21, joker 22, and ten 23. The table is the heuristic's only
parameter; it deliberately ignores the ``Rank`` enum's own ordering, in which a
joker would otherwise outrank a ten.
"""


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


def _score(move: Move, ranks: Mapping[CardId, Rank]) -> tuple[int, ...]:
    """Score one candidate move; smaller sorts better.

    Arrangements sort by the negated retention sum of the cards they leave face
    up, so the largest sum wins. Plays sort by negated batch size first -- shed
    as many cards as possible -- and then by the retention score of the rank
    spent, so among equally sized plays the cheapest cards go first. Blind
    reveals and a forced pickup offer nothing to compare: every reveal hides the
    same unknown, and a pickup is the only action when it appears at all.

    Args:
        move: A candidate from the engine's legal-move tuple.
        ranks: Rank per identifier for the viewer's own visible cards, needed
            only by arrangements.

    Returns:
        A sort key. Keys are only ever compared within one decision, where every
        candidate is the same kind of move and therefore has the same shape.

    Raises:
        ValueError: If the move is not a known move type.
    """
    match move:
        case Arrange(face_up_cards=card_ids):
            return (-sum(RETENTION_SCORE[ranks[card_id]] for card_id in card_ids),)
        case Play(rank=rank, count=count):
            return (-count, RETENTION_SCORE[rank])
        case Reveal() | PickUp():
            return ()
    raise ValueError(f"Unknown move {move!r}")


class GreedyAgent(Agent):
    """Sheds as much as it can, as cheaply as it can, and keeps its powers.

    The agent decides in one pass and submits that decision as final: it has no
    iterative improvement to spend the remaining budget on, so it never consults
    the clock.
    """

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Score every legal move, break ties with the seeded generator, submit.

        Args:
            view: This seat's observation; its ``legal_moves`` is the candidate
                set, and the agent derives no legality of its own.
            turn: The submission channel for this decision.

        Raises:
            ValueError: If the view offers no legal moves.
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
        moves = legal_choices(view)
        ranks = _own_rank_by_id(view)
        scored = [(_score(move, ranks), move) for move in moves]
        best = min(key for key, _ in scored)
        # The tied list keeps the engine's move order, so the seeded choice
        # among equals is reproducible rather than dependent on iteration luck.
        tied = [move for key, move in scored if key == best]
        return self._rng.choice(tied)
