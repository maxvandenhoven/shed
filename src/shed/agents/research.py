"""The research agent: an evolving heuristic measured against the greedy baseline.

The strategy here is developed by experiment. Each version is benchmarked head to
head against :class:`~shed.agents.greedy.GreedyAgent` over a fixed bank of deals,
and only a version that scores better replaces the one before it, so the file
always holds the best measured heuristic rather than the most recent idea.

Five scoring ideas are in it, in the order they were measured, and a search
sits on top of them.

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

**Pay extra to block a seat that is about to win.** A pile measures what a block
costs an opponent in cards, which is the wrong measure when the cards they still
hold are one turn from ending the game. Against a seat down to its last couple of
cards the block is worth a fixed premium on top.

**Search the endgame, and only the endgame.** Once the draw pile is empty the
position is small, a rollout reaches a real winner in a few milliseconds instead
of being cut off at a horizon, and the outcome it reports is the game's own
rather than an evaluation function's guess. So when the deck is gone and the
scores are close, the shortlisted moves are played out on sampled worlds and the
one that wins most often is taken. Everywhere else the static score decides
alone: the same search run over mid-game positions measured *worse* than no
search at all, because a rollout truncated at a horizon is mostly noise.

The agent decides in one pass and submits once, as final. A single final
submission is what the shared agent tests require of every strategy in the
factory, so the search is bounded by a sample count rather than by the clock,
with the clock only as a backstop, and any failure inside it falls back to the
static choice rather than costing the decision.

Nothing here decides legality. The candidate set is always ``view.legal_moves``,
and :func:`shed.engine.can_play_rank` -- the engine's own predicate -- is what
the block estimate asks about a hypothetical card. The one piece of rules
knowledge reimplemented is :func:`_constraint_after`, which predicts the
*consequence* of a move the engine has already declared legal.
"""

import random
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from math import comb

from shed.agents.base import Agent, TurnContext
from shed.agents.greedy import GreedyAgent
from shed.engine import (
    Arrange,
    AtLeast,
    AtMost,
    Card,
    CardId,
    CardRevealed,
    CardsPlayed,
    GameState,
    Move,
    Phase,
    PickUp,
    PilePickedUp,
    Play,
    PlayConstraint,
    PlayerId,
    PlayerState,
    PlayerView,
    PublicPlayerState,
    Rank,
    Reveal,
    SlotId,
    Unrestricted,
    build_deck,
    can_play_rank,
    validate_decision_boundary,
)

__all__ = [
    "BLOCK_WEIGHT",
    "ENDGAME_MARGIN",
    "ENDGAME_SAMPLES",
    "ENDGAME_WIDTH",
    "MATCH_POINT",
    "MATCH_POINT_VALUE",
    "RACE_WEIGHT",
    "RETENTION",
    "ResearchAgent",
]

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

MATCH_POINT = 2
"""Cards left in a seat's every zone at which it counts as about to win."""

MATCH_POINT_VALUE = 10.0
"""Retention points a certain block is worth against a seat at match point.

The ordinary bonus is proportional to the pile, because a pile is what a blocked
opponent has to pick up and then shed again. That reasoning fails at the end: a
seat two cards from winning is barely inconvenienced by a three-card pile in
absolute terms, and enormously inconvenienced by having to take it at all. This
premium is added on top of the pile term whenever the next seat holds
:data:`MATCH_POINT` cards or fewer, and it is deliberately blunt -- the score is
flat anywhere between roughly 6 and 15, so nothing here balances on the value.
"""

ENDGAME_SAMPLES = 8
"""Worlds sampled per searched decision.

Eight is where the score stopped moving: sixteen measured the same within noise
and cost twice as much. Each sample is a complete game rather than a truncated
one, so the estimate it contributes is an outcome, not a guess.
"""

ENDGAME_WIDTH = 3
"""Most candidates a search will compare.

The static score is a good ranker even when it is not a good decider, so the
shortlist is the moves it already likes. Three is enough to cover a genuine
disagreement without paying for the tail.
"""

ENDGAME_MARGIN = 3.0
"""Retention points within the best static score that put a move on the shortlist.

A move the static score dislikes by more than this is not close, and searching it
would spend samples separating options that are not in contention.
"""

ENDGAME_OVERRIDE = 0.125
"""Share of samples by which search must beat the static pick to replace it.

The static choice is the incumbent. Requiring a clear margin -- an eighth of the
samples -- keeps sampling noise from unseating a decision the heuristic got right,
which is what separates this from the mid-game search that measured worse.
"""

ENDGAME_TIME_FLOOR = 0.5
"""Seconds left at which the search stops sampling and takes what it has.

The sample count is the real bound -- a searched decision costs about 50 ms on
average and 0.6 s at the 99th percentile -- and this is the backstop that keeps an
unusually large endgame from running past its deadline, which would cost the
whole match under strict accounting. It is checked before every sample *and*
between candidates within one, so the longest a decision can overrun the check is
a single rollout; a decision that starts with nothing left submits the static
choice instead. Measured over 4575 decisions against a real countdown, the
slowest searched decision took under a second of the 2.5 available.
"""

_SEED_SPACE = 2**32
"""Range the rollout policies' per-decision seeds are drawn from."""

_ROLLOUT_LIMIT = 120
"""Decisions a single rollout may take before it is abandoned as unfinished.

An endgame settles in roughly twenty decisions, so this is generous for the
games that end and short for the ones that do not. The bound is what keeps the
*worst* case affordable rather than the typical one: two agents can recirculate
the same cards indefinitely, and at the old bound of 400 a single decision's
samples could run past the whole turn budget. One did, in one decision out of
15698, and it cost that match.
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

    The pool deliberately includes the viewer's own face-down cards. That is not
    an approximation: nothing distinguishes one unseen card from another here, so
    every hand-sized subset of the pool is equally likely and the draw is the
    correct marginal. What the estimate does ignore is inference from *behaviour*
    -- which cards a seat has picked up, and what its choices imply about the
    rest -- and tracking the pickups was measured and did not pay for itself.

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
    by :func:`_race_multiplier` for how badly this seat needs the swing, plus
    :data:`MATCH_POINT_VALUE` when that seat is about to win.

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
    following = _following_seat(view)
    chance = _block_chance(view, unseen, constraint)
    taken = len(view.discard_pile) + move.count
    worth = BLOCK_WEIGHT * _race_multiplier(view) * taken
    if _remaining(following) <= MATCH_POINT:
        worth += MATCH_POINT_VALUE
    return value - chance * worth


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


class _Silent:
    """A turn context that records a simulated agent's move and nothing else.

    Rollouts drive real agents, and a real agent needs somewhere to submit. This
    is that somewhere: it has no clock to speak of and no deadline, because a
    simulated decision is not timed.
    """

    def __init__(self) -> None:
        """Start with nothing submitted."""
        self.move: Move | None = None

    def remaining_seconds(self) -> float:
        """Report a fixed budget; a simulated decision is never timed.

        Returns:
            A constant, large enough that no policy tries to economize.
        """
        return 1.0

    def submit(self, move: Move, *, final: bool = False) -> None:
        """Keep the latest candidate.

        Args:
            move: The submitted move.
            final: Ignored; the last submission is the one that counts.
        """
        self.move = move


def _unseen_cards(view: PlayerView) -> list[Card]:
    """Return the canonical cards this seat has never been shown.

    Args:
        view: The viewer's observation.

    Returns:
        The cards in deck order. They are exactly the ones sitting where the
        viewer cannot see: the draw pile, every face-down slot, and the other
        seats' hands.
    """
    seen: set[CardId] = {card.id for card in view.hand}
    for public in view.players:
        seen.update(card.id for card in public.face_up)
    seen.update(card.id for card in view.discard_pile)
    seen.update(card.id for card in view.burned_cards)
    return [card for card in build_deck() if card.id not in seen]


def _known_cards(view: PlayerView) -> dict[PlayerId, set[CardId]]:
    """Track the cards each seat is publicly known to still hold.

    A pickup names every card it transfers, so those cards are known to be in
    that seat's hand from then until it plays them; a play and a reveal both name
    what left. Draws are private, so what this recovers is a *slice* of a hand
    rather than the whole of it.

    Nothing here is inference: every event read is one the filtered history shows
    this viewer, carrying identities the engine made public.

    Args:
        view: The viewer's observation, carrying its filtered history.

    Returns:
        Known card identifiers per seat; seats with nothing known map to an
        empty set.
    """
    held: dict[PlayerId, set[CardId]] = {seat: set() for seat in view.seat_order}
    for event in view.history:
        match event:
            case PilePickedUp(player=player, cards=cards):
                held[player].update(card.id for card in cards)
            case CardsPlayed(player=player, cards=cards):
                held[player].difference_update(card.id for card in cards)
            case CardRevealed(player=player, card=card):
                held[player].discard(card.id)
            case _:
                pass
    return held


def _determinize(view: PlayerView, rng: random.Random) -> GameState:
    """Sample one full position the observation could have come from.

    The unseen cards are dealt back to the places the view says are hidden: every
    face-down slot including the viewer's own, the other seats' hands at their
    public counts, and whatever is left to the draw pile. Nothing but the
    observation and the generator goes in, so a sampled world is a guess this
    seat is entitled to make.

    The deal is not uniform, because the observation is not silent. Cards a
    pickup put in a hand are placed back in that hand rather than shuffled among
    the unknowns, so every sampled world agrees with what the history already
    showed. Only the genuinely unknown remainder is dealt at random.

    Args:
        view: The viewer's observation.
        rng: Generator for the deal.

    Returns:
        A state that passes the engine's decision-boundary invariants and offers
        the viewer exactly the moves the observation offered.

    Raises:
        StateInvariantError: If the sample is not a valid position, which would
            mean the observation and the deck disagree.
    """
    unseen = _unseen_cards(view)
    known = _known_cards(view)
    # A slice larger than the public hand count is stale rather than usable, so
    # that seat falls back to an ordinary random hand.
    counts = {public.player: public.hand_count for public in view.players}
    placed: dict[PlayerId, list[Card]] = {}
    spoken: set[CardId] = set()
    for seat, ids in known.items():
        if seat == view.viewer or not ids or len(ids) > counts[seat]:
            continue
        cards = [card for card in unseen if card.id in ids]
        if len(cards) != len(ids):  # a known card is no longer unseen; distrust it
            continue
        placed[seat] = cards
        spoken.update(ids)

    pool = [card for card in unseen if card.id not in spoken]
    rng.shuffle(pool)
    cursor = 0
    players: dict[PlayerId, PlayerState] = {}
    for public in view.players:
        face_down: dict[SlotId, Card] = {}
        for slot in public.face_down_slots:
            face_down[slot] = pool[cursor]
            cursor += 1
        if public.player == view.viewer:
            hand = list(view.hand)
        else:
            hand = list(placed.get(public.player, ()))
            wanted = public.hand_count - len(hand)
            hand.extend(pool[cursor : cursor + wanted])
            cursor += wanted
        players[public.player] = PlayerState(
            hand=hand, face_up=list(public.face_up), face_down=face_down
        )
    state = GameState(
        rules=view.rules,
        players=players,
        seat_order=view.seat_order,
        dealer=view.dealer,
        phase=view.phase,
        current_player=view.current_player,
        current_ply=view.current_ply,
        draw_pile=pool[cursor:],
        discard_pile=list(view.discard_pile),
        burned_cards=list(view.burned_cards),
        constraint=view.constraint,
        setup=None,
        outcome=view.outcome,
    )
    validate_decision_boundary(state)
    return state


def _rollout(state: GameState, viewer: PlayerId, rng: random.Random) -> float:
    """Play a sampled world out and report whether the viewer won it.

    Each simulated seat decides from ``state.observe`` of its own seat, so a
    policy inside a rollout sees exactly what that seat would see and never the
    cards this sample happens to have dealt elsewhere. The viewer plays the
    *static* heuristic rather than this class, because a searching agent inside
    its own search would recurse; the other seats play the greedy baseline, which
    is an explicit model of the opponent rather than anything read off the state.

    Args:
        state: A sampled world, mutated to the end of the game.
        viewer: The seat whose result is wanted.
        rng: Generator for the simulated agents' seeds.

    Returns:
        ``1.0`` if the viewer won, ``0.0`` otherwise. A game that somehow fails
        to end inside the bound counts as a loss, which is what a game the runner
        would truncate is worth.
    """
    for _ in range(_ROLLOUT_LIMIT):
        if state.is_finished:
            break
        actor = state.current_player
        if actor is None:
            break
        seat_view = state.observe(actor)
        seed = rng.randrange(_SEED_SPACE)
        if actor == viewer:
            move = _static_choice(seat_view, random.Random(seed))
        else:
            turn = _Silent()
            GreedyAgent(seed=seed).think(seat_view, turn)
            if turn.move is None:
                break
            move = turn.move
        state.apply_move(move)
    outcome = state.outcome
    return 1.0 if outcome is not None and outcome.winner == viewer else 0.0


def _static_choice(view: PlayerView, rng: random.Random) -> Move:
    """Pick a move by the static score alone, with no search.

    This is both the agent's answer outside the endgame and the viewer's policy
    inside a rollout, which is what keeps the search from recursing into itself.

    Args:
        view: The observation to decide on.
        rng: Generator that settles ties.

    Returns:
        One of ``view.legal_moves``.

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
    # The tied list keeps the engine's move order, so the seeded choice among
    # equals is reproducible rather than dependent on iteration luck.
    tied = [move for key, move in scored if key == best]
    return rng.choice(tied)


def _leader(wins: list[float]) -> int:
    """Return the index of the best-scoring candidate so far.

    Args:
        wins: Accumulated wins per shortlisted move.

    Returns:
        The leading index; ties go to the earliest, which is the candidate the
        static score ranked highest.
    """
    return max(range(len(wins)), key=lambda index: wins[index])


def _shortlist(view: PlayerView, fallback: Move) -> list[Play]:
    """Return the plays worth searching, best static score first.

    Args:
        view: The observation to decide on.
        fallback: The static pick, which always makes the list.

    Returns:
        Up to :data:`ENDGAME_WIDTH` plays within :data:`ENDGAME_MARGIN` of the
        best static score, or an empty list when there is nothing to compare.
    """
    plays = [move for move in view.legal_moves if isinstance(move, Play)]
    if len(plays) < 2:
        return []
    unseen = _unseen_ranks(view)
    ranked = sorted(plays, key=lambda move: (_play_value(move, view, unseen), -move.count))
    limit = _play_value(ranked[0], view, unseen) + ENDGAME_MARGIN
    short = [move for move in ranked if _play_value(move, view, unseen) <= limit][:ENDGAME_WIDTH]
    if isinstance(fallback, Play) and fallback in plays and fallback not in short:
        short = [fallback, *short[: ENDGAME_WIDTH - 1]]
    return short if len(short) > 1 else []


class ResearchAgent(Agent):
    """The experimental strategy, scored move by move over the legal tuple.

    The agent reads its options from ``view.legal_moves`` and derives no
    legality of its own, so a narrowed view narrows the agent. It decides in one
    pass and submits that decision as final.
    """

    def think(self, view: PlayerView, turn: TurnContext) -> None:
        """Score the legal moves, search the close endgame ones, submit once.

        Args:
            view: This seat's observation; its ``legal_moves`` is the candidate
                set.
            turn: The submission channel for this decision.

        Raises:
            ValueError: If the view offers no legal moves, which means this seat
                is not the current actor.
        """
        choice = _static_choice(view, self._rng)
        if self._searchable(view):
            choice = self._search(view, choice, turn)
        turn.submit(choice, final=True)

    def _searchable(self, view: PlayerView) -> bool:
        """Report whether this decision is one worth spending rollouts on.

        Args:
            view: This seat's observation.

        Returns:
            ``True`` only in the endgame. While the deck lasts a rollout cannot
            reach the end of the game cheaply, and a truncated one measured worse
            than no search at all.
        """
        return view.draw_count == 0 and view.phase is not Phase.SETUP

    def _search(self, view: PlayerView, fallback: Move, turn: TurnContext) -> Move:
        """Compare shortlisted moves on shared sampled worlds.

        Every candidate is played out on the *same* worlds with the same rollout
        seeds, so the comparison is paired and most of the sampling noise cancels
        rather than deciding the move. The static pick is the incumbent and is
        only replaced when a rival wins by :data:`ENDGAME_OVERRIDE` of the
        samples.

        Search is an optimization, never a risk to the decision: anything raised
        inside it -- an exhausted pool, an engine refusal, a sample that is not a
        legal world -- leaves the static choice standing.

        Args:
            view: This seat's observation.
            fallback: The static pick, returned unless search clearly beats it.
            turn: The submission channel, consulted only for its clock.

        Returns:
            One of ``view.legal_moves``.
        """
        try:
            short = _shortlist(view, fallback)
            if not short:
                return fallback
            wins = [0.0] * len(short)
            taken = 0
            for _ in range(ENDGAME_SAMPLES):
                if turn.remaining_seconds() < ENDGAME_TIME_FLOOR:
                    break
                world = _determinize(view, self._rng)
                seed = self._rng.randrange(_SEED_SPACE)
                for index, move in enumerate(short):
                    if index and turn.remaining_seconds() < ENDGAME_TIME_FLOOR:
                        # Abandon a half-finished sample rather than score the
                        # candidates on different numbers of rollouts.
                        return fallback if not taken else short[_leader(wins)]
                    trial = deepcopy(world)
                    trial.apply_move(move)
                    wins[index] += _rollout(trial, view.viewer, random.Random(seed))
                taken += 1
            if not taken:
                return fallback
            held = short.index(fallback) if isinstance(fallback, Play) and fallback in short else 0
            best = max(range(len(short)), key=lambda i: wins[i])
            if wins[best] <= wins[held] + ENDGAME_OVERRIDE * taken:
                return fallback
            return short[best]
        except Exception:
            return fallback
