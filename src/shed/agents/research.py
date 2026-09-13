"""The research agent: an evolving heuristic measured against the greedy baseline.

The strategy here is developed by experiment. Each version is benchmarked head to
head against :class:`~shed.agents.greedy.GreedyAgent` over a fixed bank of deals,
and only a version that scores better replaces the one before it, so the file
always holds the best measured heuristic rather than the most recent idea.

Four ideas are in it, in the order they were measured.

**Spend the cheapest rank, then every copy of it.** Greedy sorts plays by
``(-count, retention)``, so batch size decides first and it dumps three kings
when one would do. This agent puts retention first. A high ordinary rank answers
more constraints than a low one, since ``AtLeast(r)`` accepts everything from
``r`` up, and the four always-playable ranks answer every constraint there is, so
spending the cheapest rank keeps the most flexible cards in hand -- which is what
stops a turn from ending in a forced pickup.

**Price a seven as an ordinary card.** :data:`RETENTION` is this agent's own
table, and it differs from greedy's in that one entry.

**Value a play by what it does to the opponent.** The observation implies exactly
which cards nobody has seen. The next seat's hand is that many unseen cards, so
the chance that none of them answers the constraint a candidate would leave is a
hypergeometric draw over the unseen ranks, and a blocked opponent takes the whole
pile. A play that probably blocks is therefore worth more than the cards it
spends, and the pile measures how much more.

**Price that pressure by who is winning the race.** Holding more cards than the
next seat means losing on the current trajectory, so a card spent to bury them is
worth more than one kept; holding fewer means the cheap, safe play is already
winning. The bonus is scaled accordingly.

The agent decides in one pass and submits once, as final. That is not a
concession to the clock: the heuristic costs microseconds, so there is nothing to
improve with the rest of the budget, and a single final submission is what the
shared agent tests require of every strategy in the factory.

Nothing here decides legality. The candidate set is always ``view.legal_moves``,
and :func:`shed.engine.can_play_rank` -- the engine's own predicate -- is what
the block estimate asks about a hypothetical card. The one piece of rules
knowledge reimplemented is :func:`_constraint_after`, which predicts the
*consequence* of a move the engine has already declared legal.
"""

from collections import Counter
from collections.abc import Mapping
from math import comb

from shed.agents.base import Agent, TurnContext
from shed.engine import (
    Arrange,
    AtLeast,
    AtMost,
    CardId,
    Move,
    PickUp,
    Play,
    PlayConstraint,
    PlayerView,
    PublicPlayerState,
    Rank,
    Reveal,
    Unrestricted,
    build_deck,
    can_play_rank,
)

__all__ = ["BLOCK_WEIGHT", "RACE_WEIGHT", "RETENTION", "ResearchAgent"]

RETENTION: Mapping[Rank, int] = {
    Rank.THREE: 3,
    Rank.FOUR: 4,
    Rank.FIVE: 5,
    Rank.SIX: 6,
    Rank.SEVEN: 10,
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
"""How much this agent wants to keep a card of each rank.

It starts from greedy's table -- ordinary ranks score their numeric value, and
the ranks that play against any pile score above every ordinary one -- and
differs in exactly one entry, which is a measured change rather than a taste.

A **seven scores 10 rather than 17**, which drops it below the jack and makes it
an early spend instead of a hoarded special. A seven is the weakest of the
"special" ranks to hold: it is not always playable, it answers only the
constraints a low card answers, and the ``AtMost(SEVEN)`` it leaves behind
restricts *this* agent again on its next turn as much as it restricts anybody.
Holding one is a liability the table now prices in.
"""

BLOCK_WEIGHT = 1.0
"""Retention points one expected pile card is worth when a play blocks.

The two quantities a play trades off are in different units -- cards kept in hand
against cards pushed onto the opponent -- and this is the exchange rate between
them. At ``1.0`` a play that certainly blocks an opponent holding a ten-card pile
is worth ten retention points, which buys a king over a three but not a ten over
one. It was measured over six independent deal banks; the score is flat between
about 0.75 and 1.5 and falls away outside that range.
"""

RACE_WEIGHT = 0.5
"""How far the card race moves the block bonus, either side of even.

The exchange rate above is the price of pressure in an even position. It should
not be the price in every position: a seat holding twelve cards against three is
losing on the present trajectory and should pay more for a chance to reverse it,
while a seat three cards from winning gains little by spending its best card.
This weight scales the bonus by the normalized card difference, so at ``0.5`` the
bonus runs from half again as valuable when hopelessly behind to half as valuable
when hopelessly ahead. Reversing its sign costs about four points of win rate,
which is what says the direction is real rather than fitted.
"""

_DECK_RANKS: Mapping[Rank, int] = Counter(card.rank for card in build_deck())
"""How many cards of each rank the canonical deck holds; the unseen baseline."""


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


def _unseen_ranks(view: PlayerView) -> Counter[Rank]:
    """Count the cards whose identity this seat has never been shown.

    Everything the viewer has seen is in the observation: its own hand, every
    seat's public face-up cards, the pile in play order, and the burned
    collection. What is left is the draw pile, every face-down slot -- the
    viewer's own included, since owners do not know their own blind cards -- and
    the other seats' hands.

    Args:
        view: The viewer's observation.

    Returns:
        Remaining count per rank. No hidden information is used to build it, and
        a card that was seen and then hidden again, such as a pile taken into a
        hand, correctly returns to the unseen pool.
    """
    seen: Counter[Rank] = Counter(card.rank for card in view.hand)
    for public in view.players:
        seen.update(card.rank for card in public.face_up)
    seen.update(card.rank for card in view.discard_pile)
    seen.update(card.rank for card in view.burned_cards)
    return Counter(_DECK_RANKS) - seen


def _constraint_after(rank: Rank, count: int, current: PlayConstraint) -> PlayConstraint | None:
    """Predict the constraint a play would leave the next seat.

    This is the profile's resolution rule, not a legality decision: the move was
    already offered by the engine, and this only says what the pile would look
    like afterwards. A burn is reported as ``None`` because it leaves no
    constraint for an opponent to answer -- the same seat decides again.

    Args:
        rank: Rank of the batch.
        count: Cards in the batch; four of one rank burn when played at once.
        current: The constraint standing now.

    Returns:
        The constraint the next ordinary rank would have to satisfy, or ``None``
        when the play burns the pile.
    """
    if rank is Rank.TEN or count == 4:
        return None
    if rank is Rank.NINE:
        return current
    if rank is Rank.JOKER:
        return Unrestricted()
    if rank is Rank.SEVEN:
        return AtMost(Rank.SEVEN)
    return AtLeast(rank)


def _following_seat(view: PlayerView) -> PublicPlayerState:
    """Return the public state of the seat that decides after this one.

    A play's pressure lands on exactly one seat, the next one clockwise, because
    that is who must answer the constraint it leaves. With more than two seats
    the ones after it are not modelled; they will face a position this decision
    cannot predict.

    Args:
        view: The viewer's observation.

    Returns:
        The following seat's public state.
    """
    seats = view.seat_order
    following = seats[(seats.index(view.viewer) + 1) % len(seats)]
    return next(entry for entry in view.players if entry.player == following)


def _remaining(public: PublicPlayerState) -> int:
    """Count the cards one seat still has to shed, across every zone.

    Args:
        public: A seat's public state.

    Returns:
        Hand size plus face-up cards plus face-down slots. Every term is public,
        and none of them names a card.
    """
    return public.hand_count + len(public.face_up) + len(public.face_down_slots)


def _race_multiplier(view: PlayerView) -> float:
    """Scale the block bonus by how the card race stands.

    Args:
        view: The viewer's observation.

    Returns:
        ``1.0`` in an even race, rising towards ``1 + RACE_WEIGHT`` as this seat
        falls behind and falling towards ``1 - RACE_WEIGHT`` as it pulls ahead.
    """
    mine = _remaining(view.me)
    theirs = _remaining(_following_seat(view))
    return 1.0 + RACE_WEIGHT * (mine - theirs) / max(mine + theirs, 1)


def _block_chance(view: PlayerView, unseen: Counter[Rank], constraint: PlayConstraint) -> float:
    """Estimate the chance the next seat has no answer to ``constraint``.

    The next seat's hand is modelled as an unordered draw of ``hand_count`` cards
    from the unseen pool, so the chance every one of them is dead against the
    constraint is a hypergeometric probability. Two cases are not estimates at
    all: a seat with an empty hand plays from its *public* face-up cards, which
    settles the question exactly, and a seat with nothing but face-down slots
    turns over one unknown card, which is the same draw with one card.

    The pool is slightly pessimistic on purpose -- it includes the viewer's own
    face-down cards, which the opponent cannot hold -- because correcting for
    three cards out of twenty or more would not change which move wins.

    Args:
        view: The viewer's observation.
        unseen: Remaining count per rank, from :func:`_unseen_ranks`.
        constraint: The constraint the play under consideration would leave.

    Returns:
        A probability in ``[0, 1]``. It is ``0.0`` whenever the pool cannot
        support the estimate, which keeps an impossible draw from inventing a
        bonus.
    """
    public = _following_seat(view)
    if public.hand_count == 0 and public.face_up:
        playable = any(can_play_rank(card.rank, constraint) for card in public.face_up)
        return 0.0 if playable else 1.0

    total = sum(unseen.values())
    dead = sum(count for rank, count in unseen.items() if not can_play_rank(rank, constraint))
    # An empty hand with no face-up cards means a single blind reveal, which is
    # one card drawn from the same pool.
    held = public.hand_count or 1
    if held > dead or held > total:
        return 0.0
    return comb(dead, held) / comb(total, held)


def _play_value(move: Play, view: PlayerView, unseen: Counter[Rank]) -> float:
    """Score one play in retention points; smaller is better.

    The cost of a play is what it gives up, the retention of the rank it spends.
    Against that stands what it does to the next seat: a play that leaves a
    constraint they probably cannot answer hands them the whole pile, which is
    worth :data:`BLOCK_WEIGHT` retention points per card they would take, scaled
    by :func:`_race_multiplier` for how badly this seat needs the swing.

    A burn earns no such bonus. It clears the pile rather than handing it over
    and leaves the opponent unrestricted, so its only merits -- removing cards
    from the game and keeping the turn -- are ones the retention table already
    prices into the ten it spends.

    Args:
        move: The candidate play.
        view: The observation being decided on.
        unseen: Remaining count per rank, from :func:`_unseen_ranks`.

    Returns:
        The play's cost net of the pressure it applies.
    """
    value = float(RETENTION[move.rank])
    constraint = _constraint_after(move.rank, move.count, view.constraint)
    if constraint is None:
        return value
    taken = len(view.discard_pile) + move.count
    bonus = BLOCK_WEIGHT * _race_multiplier(view) * _block_chance(view, unseen, constraint)
    return value - bonus * taken


def _score(
    move: Move,
    view: PlayerView,
    ranks: Mapping[CardId, Rank],
    unseen: Counter[Rank],
) -> tuple[float, ...]:
    """Score one candidate move; smaller sorts better.

    Arrangements sort by the negated retention sum of the cards they leave face
    up, so the largest sum wins: face-up cards are played last, against a late and
    hostile pile, which is where the flexible ranks earn their keep. Plays sort by
    :func:`_play_value` and then by negated batch size, so the whole of the chosen
    rank goes at once rather than one card at a time. Blind reveals and a forced
    pickup offer nothing to compare: every reveal hides the same unknown, and a
    pickup is the only action when it appears at all.

    Args:
        move: A candidate from the engine's legal-move tuple.
        view: The observation being decided on.
        ranks: Rank per identifier for the viewer's own visible cards, needed
            only by arrangements.
        unseen: Remaining count per rank, from :func:`_unseen_ranks`.

    Returns:
        A sort key. Keys are only ever compared within one decision, where every
        candidate is the same kind of move and therefore has the same shape.

    Raises:
        ValueError: If the move is not a known move type.
    """
    match move:
        case Arrange(face_up_cards=card_ids):
            return (-float(sum(RETENTION[ranks[card_id]] for card_id in card_ids)),)
        case Play(count=count):
            return (_play_value(move, view, unseen), float(-count))
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
        unseen = _unseen_ranks(view)
        scored = [(_score(move, view, ranks, unseen), move) for move in moves]
        best = min(key for key, _ in scored)
        # The tied list keeps the engine's move order, so the seeded choice
        # among equals is reproducible rather than dependent on iteration luck.
        tied = [move for key, move in scored if key == best]
        return self._rng.choice(tied)
