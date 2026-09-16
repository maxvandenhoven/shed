from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from shed.agents import AGENT_KINDS, RETENTION_SCORE, AgentSpec, build_agent
from shed.companion.observed import (
    ME,
    RANK_TEXT,
    SEAT_NAMES,
    SEATS,
    ObservationError,
    ObservedState,
    describe_constraint,
    observed_legal_moves,
)
from shed.engine import (
    Arrange,
    Card,
    CardId,
    Move,
    PickUp,
    Play,
    PlayerId,
    PlayerView,
    PublicPlayerState,
    Rank,
    Reveal,
    SlotId,
    Suit,
    Zone,
    burn_reason,
    constraint_after,
)

__all__ = [
    "ADVICE_BUDGET_SECONDS",
    "AGENT_PROFILES",
    "BOOKKEEPING_SUIT",
    "DEFAULT_AGENT",
    "FAITHFUL_VIEW_FIELDS",
    "AgentChoice",
    "AgentProfile",
    "Recommendation",
    "agent_catalogue",
    "build_player_view",
    "describe_move",
    "profile_for",
    "recommend",
    "view_gaps",
]

FAITHFUL_VIEW_FIELDS: frozenset[str] = frozenset(
    {"rules", "viewer", "seat_order", "phase", "current_player", "hand", "players", "legal_moves"}
)
"""The ``PlayerView`` fields a companion-built view fills as truthfully as the engine.

``hand`` is exact because :func:`build_player_view` refuses to build a view whose
own hand has unrecorded ranks. ``players`` is exact in the counts it promises,
hand size, the public face-up set, one slot per remaining face-down card, and
``PublicPlayerState`` promises nothing more about an opponent's hand than its size.
``legal_moves`` comes from the engine's own generator over observed counts. The
remaining fields are either constant (``rules``, ``seat_order``, ``viewer``) or
tracked exactly (``phase``, ``current_player``).

Every other field is *thinner* than the engine's: see :func:`view_gaps` for the
ones the operator is told about, and the module docstring for the rest. An agent
that reads outside this set still runs, but it is reading an observed game rather
than an omniscient one, and it should say so.
"""

BOOKKEEPING_SUIT: Suit = Suit.CLUBS
"""The one suit every card in a companion-built view carries.

``Card`` requires a suit for an ordinary rank, and this profile never reads one.
Using a single value throughout, rather than spreading four around to look like a
deal, keeps the suits in these views obviously meaningless: they are the shape the
type demands, not an observation of the physical card.
"""

ADVICE_BUDGET_SECONDS: float = 1.0
"""What the agent is told is left of its budget; nothing enforces it.

The greedy baseline decides in one pass and never asks, but the protocol has the
question, so the answer is a fixed cooperative hint. A future agent that iterates
would be cut off by nothing here, add real enforcement before shipping one.
"""


class _Recorder:
    """A turn context that collects submissions instead of racing a clock.

    The match runner's context writes candidates down a pipe to a parent process
    that enforces a deadline. Here the agent is in the same process and there is no
    deadline, so the context is the whole channel: it keeps the latest legal
    candidate, exactly as the runner's selection policy would, and remembers
    whether the agent closed the decision.

    Attributes:
        _legal: The moves the view offered; anything else is dropped, keeping this
            context as strict about legality as the runner is.
        _chosen: The latest legal candidate, or ``None`` if none was legal.
        _closed: Whether a submission arrived marked final.
    """

    def __init__(self, legal: tuple[Move, ...]) -> None:
        """Start with no candidate and the decision open.

        Args:
            legal: The legal moves for this decision.
        """
        self._legal = legal
        self._chosen: Move | None = None
        self._closed = False

    @property
    def chosen(self) -> Move | None:
        """The candidate that would have been accepted, if any."""
        return self._chosen

    @property
    def closed(self) -> bool:
        """Whether the agent finalized its decision."""
        return self._closed

    def remaining_seconds(self) -> float:
        """Return the fixed budget hint.

        Returns:
            :data:`ADVICE_BUDGET_SECONDS`. Nothing enforces it; see that constant.
        """
        return ADVICE_BUDGET_SECONDS

    def submit(self, move: Move, *, final: bool = False) -> None:
        """Accept one candidate, keeping the latest legal one.

        Args:
            move: The candidate. An illegal one is dropped and leaves the previous
                candidate standing, as the runner would.
            final: Whether this submission closes the decision.
        """
        if move in self._legal:
            self._chosen = move
        if final:
            self._closed = True


class _Bookkeeper:
    """Hands out distinct bookkeeping card identifiers for one view.

    Identifiers only have to be distinct inside a single view: the greedy baseline
    uses them to look a rank back up when scoring an arrangement, and nothing
    compares them across views or against the canonical deck.

    Attributes:
        _next: The next identifier to hand out.
    """

    def __init__(self) -> None:
        """Start numbering from zero."""
        self._next = 0

    def card(self, rank: Rank) -> Card:
        """Build one card with the next bookkeeping identifier.

        Args:
            rank: The observed rank, which is the only real information here.

        Returns:
            A card carrying that rank, a fresh identifier, and
            :data:`BOOKKEEPING_SUIT` unless the rank is a joker, which has no suit.
        """
        card = Card(
            id=CardId(self._next),
            rank=rank,
            suit=None if rank is Rank.JOKER else BOOKKEEPING_SUIT,
        )
        self._next += 1
        return card


def build_player_view(state: ObservedState, viewer: PlayerId = ME) -> PlayerView:
    """Project a tracked physical game into the observation an agent expects.

    What reaches the view is exactly what was observed. Ranks that were seen become
    cards with bookkeeping identifiers; everything else stays a count:

    * The opponent's hand appears only as ``hand_count``. Their known ranks are
      deliberately dropped too, because ``PublicPlayerState`` has nowhere to put a
      partly-known hand and the count is the part its contract promises.
    * Face-down cards appear as slot identifiers ``0..n-1`` with no identities.
      The identifiers are positional bookkeeping: physical face-down cards are
      indistinguishable, so every reveal is the same action.
    * ``discard_pile`` and ``burned_cards`` carry the cards whose ranks were seen
      and omit the rest, so the tuples understate those piles' sizes whenever
      :func:`view_gaps` reports a gap. The alternative, padding them with
      invented ranks, would be exactly the fabrication this module exists to
      avoid. ``GreedyAgent`` reads neither field, and the full sizes are carried
      to the interface by the companion's own state instead.
    * ``history`` is empty. The companion keeps its own observation log, and the
      engine's event vocabulary cannot express "a card moved and I did not see it".

    Args:
        state: The tracked position.
        viewer: Whose view to build. Only :data:`ME` has a fully known hand, so
            only that view can honestly carry one.

    Returns:
        The view, with ``legal_moves`` filled in when the viewer is to act.

    Raises:
        ObservationError: If the viewer is not a seat, the game is over, the
            viewer's own hand has unrecorded ranks, or the position cannot
            generate moves, a missing deck count, most often.
    """
    if viewer not in SEATS:
        raise ObservationError(f"{viewer} is not a seat at this table")
    if state.is_finished:
        raise ObservationError("A finished game offers no decision to observe")
    seat = state.seat(viewer)
    if not seat.hand_fully_known:
        raise ObservationError(
            f"{seat.hand_unknown} of {SEAT_NAMES[viewer]}'s cards have no recorded "
            "rank; a view cannot be built without inventing them"
        )

    ids = _Bookkeeper()
    hand = tuple(ids.card(rank) for rank in seat.hand_known)
    publics: list[PublicPlayerState] = []
    for other in state.seats:
        publics.append(
            PublicPlayerState(
                player=other.player,
                hand_count=other.hand_count,
                face_up=tuple(ids.card(rank) for rank in other.face_up),
                face_down_slots=tuple(SlotId(index) for index in range(other.face_down)),
            )
        )
    pile = tuple(ids.card(rank) for rank in state.pile if rank is not None)
    burned = tuple(ids.card(rank) for rank in state.burned if rank is not None)
    legal = observed_legal_moves(state, viewer) if state.to_act == viewer else ()
    return PlayerView(
        rules=state.rules,
        viewer=viewer,
        seat_order=SEATS,
        dealer=SEATS[0],
        phase=state.phase,
        current_player=state.to_act,
        current_ply=0,
        hand=hand,
        players=tuple(publics),
        draw_count=state.deck_count if state.deck_count is not None else 0,
        discard_pile=pile,
        burned_cards=burned,
        constraint=state.constraint,
        outcome=None,
        history=(),
        legal_moves=legal,
    )


def view_gaps(state: ObservedState) -> tuple[str, ...]:
    """List where a built view is thinner than the position it came from.

    Every gap is a place the companion chose a count over an invented card. They
    are reported rather than hidden so the interface can say what the advice did
    not look at.

    Args:
        state: The tracked position.

    Returns:
        One sentence per gap, in reading order; empty when the view carries
        everything observed.
    """
    gaps: list[str] = []
    unseen_pile = sum(1 for rank in state.pile if rank is None)
    if unseen_pile:
        gaps.append(
            f"{unseen_pile} of the {state.pile_size} cards in the pile have no "
            "recorded rank, so they are left out of the pile the agent sees. "
            "Greedy never reads the pile."
        )
    unseen_burned = sum(1 for rank in state.burned if rank is None)
    if unseen_burned:
        plural = "cards have" if unseen_burned != 1 else "card has"
        gaps.append(
            f"{unseen_burned} burned {plural} no recorded rank and is left out of the "
            "burned cards the agent sees. Greedy never reads them."
        )
    opponent = state.seat(state.other(ME))
    if opponent.hand_known:
        gaps.append(
            f"You know {len(opponent.hand_known)} of your opponent's "
            f"{opponent.hand_count} hand cards, but a view carries only the count. "
            "Greedy never reads an opponent's hand."
        )
    return tuple(gaps)


def describe_move(move: Move, *, owner: str = "your") -> str:
    """Describe one move in the words the phone interface shows.

    Args:
        move: The move to describe.
        owner: Possessive naming whose cards move, so the sentence reads naturally
            for either seat: ``"your"`` or ``"their"``.

    Returns:
        A short imperative or descriptive phrase.

    Raises:
        ValueError: If the move is not a known move type.
    """
    match move:
        case Play(source=source, rank=rank, count=count):
            where = "hand" if source is Zone.HAND else "face-up cards"
            return f"Play {count} x {RANK_TEXT[rank]} from {owner} {where}"
        case Reveal():
            return f"Turn over one of {owner} face-down cards"
        case PickUp():
            return "Pick up the pile"
        case Arrange():
            return "Choose a face-up set"
    raise ValueError(f"Unknown move {move!r}")


def _effect_note(state: ObservedState, move: Move) -> str:
    """Say what the recommended move does to the pile, per the profile's rules.

    The sentence is derived, not written down twice: the burn rule and the
    constraint transition are the engine's own
    :func:`~shed.engine.burn_reason` and
    :func:`~shed.engine.constraint_after`, so this cannot describe an effect the
    engine does not implement.

    Args:
        state: The position the move is played into.
        move: The recommended move.

    Returns:
        One sentence, or an empty string for a move with no pile effect to state.
    """
    match move:
        case Play(rank=rank, count=count):
            reason = burn_reason(rank, count)
            if reason is not None:
                cause = "A ten" if rank is Rank.TEN else f"Four {RANK_TEXT[rank]}s in one action"
                return f"{cause} burns the pile out of the game, and you play again."
            after = constraint_after(rank, state.constraint)
            # A nine is checked before an unrestricted result: a nine played onto an
            # open pile also leaves Unrestricted, and that is transparency, not a joker.
            if rank is Rank.NINE:
                return (
                    "A nine is transparent: it leaves the restriction exactly as it "
                    f"is ({describe_constraint(after)})."
                )
            if rank is Rank.JOKER:
                return "A joker clears the pile's restriction for your opponent."
            return f"That leaves your opponent {describe_constraint(after)}."
        case Reveal():
            return (
                "A face-down card is judged against the restriction in force before "
                "it is turned; if it cannot go down you take the pile."
            )
        case PickUp():
            return "The pile goes into your hand and the restriction clears."
    return ""


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """What the interface says about one strategy, and how it explains a play.

    A profile is presentation, not behaviour: the agent's own code decides the
    move, and everything here only describes it. That separation is what keeps the
    explanation honest, :func:`_play_reasoning` reads the greedy retention table
    back out of the same module the agent scored with, and the random profile
    claims no reasoning at all, because there is none to claim.

    Attributes:
        kind: The :data:`~shed.agents.AGENT_KINDS` entry this describes.
        label: Short name for the heading above a suggestion.
        summary: One line for the agent picker, saying what the strategy does.
        caveat: The standing disclaimer shown under every suggestion it makes.
            It travels with the recommendation so no caller can present the
            strategy without what it is.
    """

    kind: str
    label: str
    summary: str
    caveat: str


GREEDY_CAVEAT: str = (
    "Greedy is a baseline heuristic, not optimal play. It looks only at your own "
    "cards and the current restriction: it does not count cards, read the pile, "
    "plan ahead, or model your opponent. No win probability is implied."
)
"""The standing caveat on every greedy recommendation.

It is a fixed string because it is a fact about ``GreedyAgent``'s implementation,
one pass over ``view.legal_moves`` scored by a fixed retention table, and not
something to soften per position.
"""

RANDOM_CAVEAT: str = (
    "Random is the floor the other agents are measured against, not advice. It "
    "samples uniformly among the legal actions and considers nothing at all. "
    "Pick it to see what the engine allows, not to play well."
)
"""The standing caveat on every random recommendation.

Blunter than the greedy one on purpose: a suggestion with no reasoning behind it
must not read like one that has some.
"""

UNKNOWN_AGENT_CAVEAT: str = (
    "The companion does not ship a description of this strategy, so it can say "
    "what the agent chose but not why. Treat the suggestion as unexplained."
)
"""The caveat for a kind that exists in the package but has no profile here.

A new agent works in the companion the moment it joins
:data:`~shed.agents.AGENT_KINDS`; what it does not get for free is an
explanation, and saying so is better than inventing one.
"""

AGENT_PROFILES: Mapping[str, AgentProfile] = {
    "greedy": AgentProfile(
        kind="greedy",
        label="Greedy",
        summary=(
            "Sheds as many cards as it can, then spends the rank it least wants to "
            "keep. The project's shedding baseline."
        ),
        caveat=GREEDY_CAVEAT,
    ),
    "random": AgentProfile(
        kind="random",
        label="Random",
        summary="Picks uniformly among the legal actions. A floor, not a strategy.",
        caveat=RANDOM_CAVEAT,
    ),
}
"""Presentation for each built-in strategy, keyed by kind.

Deliberately not exhaustive over :data:`~shed.agents.AGENT_KINDS`:
:func:`profile_for` falls back for a kind added without one, so the package stays
the single source of which agents exist.
"""

DEFAULT_AGENT = "greedy"
"""The kind a game uses when nothing else was chosen.

The shedding baseline rather than the random one: an operator who never opens the
agent picker should get the useful suggestion, and a document written before the
picker existed meant this one.
"""


def profile_for(kind: str) -> AgentProfile:
    """Return the presentation for one strategy.

    Args:
        kind: A :data:`~shed.agents.AGENT_KINDS` entry.

    Returns:
        Its profile, or a generic one naming the kind when the companion ships no
        description for it.

    Raises:
        ObservationError: If the kind is not one this package builds.
    """
    if kind not in AGENT_KINDS:
        raise ObservationError(
            f"{kind!r} is not an agent this release ships; choose one of {', '.join(AGENT_KINDS)}"
        )
    described = AGENT_PROFILES.get(kind)
    if described is not None:
        return described
    return AgentProfile(
        kind=kind,
        label=kind.replace("_", " ").title(),
        summary="No description ships with the companion for this strategy.",
        caveat=UNKNOWN_AGENT_CAVEAT,
    )


def agent_catalogue() -> tuple[AgentProfile, ...]:
    """List every strategy the operator may choose, in the package's own order.

    Returns:
        One profile per :data:`~shed.agents.AGENT_KINDS` entry. Driving the picker
        from the package rather than from a list here means a new agent appears on
        the phone as soon as it is registered, with a generic description until
        somebody writes it one.
    """
    return tuple(profile_for(kind) for kind in AGENT_KINDS)


@dataclass(frozen=True, slots=True)
class AgentChoice:
    """Which agent advises this game, and how its tie-breaking is seeded.

    The seed lives here rather than on :class:`~shed.agents.AgentSpec` on purpose.
    A spec deliberately carries no seed, the match runner draws a fresh one per
    decision and records it, so replaying a spec cannot resurrect a stale
    generator, and the companion has the opposite need: the same position should
    give the same suggestion every time the page asks. Keeping the salt out here
    respects both.

    Attributes:
        spec: The participant to build. Its kind is validated on construction.
        seed: Optional salt mixed into the per-position seed. Leave it ``None`` for
            the default, and set it to shake loose a different tie-break, worth
            something for the random baseline, and visible in greedy only among
            interchangeable face-down cards.
    """

    spec: AgentSpec
    seed: int | None = None

    @property
    def profile(self) -> AgentProfile:
        """The presentation for this choice's strategy."""
        return profile_for(self.spec.kind)

    def seed_for(self, position_seed: int) -> int:
        """Combine the position's own seed with this choice's salt.

        Args:
            position_seed: A seed derived from the session's content, so that
                asking twice about one position gives one answer.

        Returns:
            The seed to build the agent with: the position's seed unchanged when
            no salt was chosen, and a value that still depends on the position
            when one was.
        """
        if self.seed is None:
            return position_seed
        return position_seed ^ (self.seed & _SEED_MASK)


_SEED_MASK = (1 << 64) - 1
"""Bound on an operator-supplied salt, so an absurd number cannot bloat the seed."""

DEFAULT_CHOICE = AgentChoice(spec=AgentSpec(kind=DEFAULT_AGENT, name=DEFAULT_AGENT))
"""The choice a game gets when none was recorded; see :data:`DEFAULT_AGENT`."""


def _play_reasoning(kind: str, chosen: Play, options: tuple[Move, ...]) -> str:
    """Explain a recommended batch in the terms the chosen strategy actually used.

    Only a batch needs this. A blind reveal and a forced pickup are explained by the
    position rather than by the strategy, every remaining face-down card is the
    same decision, and a forced pickup is the only legal action, so every agent
    gets the same sentence for those.

    Args:
        kind: Which strategy chose.
        chosen: The recommended batch.
        options: Every legal move it chose from.

    Returns:
        A sentence grounded in that strategy's implementation, or one saying plainly
        that the companion cannot explain this strategy.
    """
    if kind == "greedy":
        return _greedy_play_reasoning(chosen, options)
    if kind == "random":
        return (
            f"The random baseline sampled uniformly from the {len(options)} legal "
            "actions here. It compared nothing: another ask on this same position "
            "would give the same answer only because the seed is fixed."
        )
    return (
        f"This strategy chose among {len(options)} legal actions. The companion "
        "ships no explanation for it, so this is what it picked, not why."
    )


def _greedy_play_reasoning(chosen: Play, options: tuple[Move, ...]) -> str:
    """Explain a recommended batch in terms of the greedy agent's own two keys.

    ``GreedyAgent`` sorts a play by ``(-count, RETENTION_SCORE[rank])``: shed as
    many cards as possible, then, among batches of that size, spend the rank it
    least wants to keep. Both halves are read back out of the same table the agent
    scored with, so the sentence cannot describe a different heuristic.

    Args:
        chosen: The recommended batch.
        options: Every legal move the agent chose from.

    Returns:
        Two clauses naming the batch size it maximized and the retention score it
        minimized.
    """
    plays = [move for move in options if isinstance(move, Play)]
    largest = max(play.count for play in plays)
    same_size = [play for play in plays if play.count == largest]
    ranks = ", ".join(RANK_TEXT[rank] for rank in sorted({play.rank for play in same_size}))
    score = RETENTION_SCORE[chosen.rank]
    size_clause = (
        f"{largest} card{'s' if largest != 1 else ''} is the biggest batch it can shed here"
    )
    if len(same_size) == 1:
        return f"{size_clause}, and {RANK_TEXT[chosen.rank]} is the only rank offering it."
    return (
        f"{size_clause}, and of the ranks that offer it ({ranks}) "
        f"{RANK_TEXT[chosen.rank]} is the one it least wants to keep "
        f"(retention score {score})."
    )


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One suggestion, with the reasoning that actually produced it.

    Attributes:
        agent: The choice that produced it, so the screen can name the strategy it
            is showing rather than implying there is only one.
        move: The move the agent submitted. It is one of the legal moves generated
            for the observed position, and nothing here applies it: recording a
            move is a separate observation the operator makes after playing it.
        headline: The action, phrased for a tap target.
        reasoning: Why this strategy picked it, in its own terms.
        effect: What the move does to the pile under this profile's rules.
        caveat: The chosen strategy's standing disclaimer, carried along so no
            caller can drop it.
        considered: How many legal moves it chose among.
        notes: Anything the built view was missing; see :func:`view_gaps`.
    """

    agent: AgentChoice
    move: Move
    headline: str
    reasoning: str
    effect: str
    caveat: str
    considered: int
    notes: tuple[str, ...]


def recommend(
    state: ObservedState,
    *,
    choice: AgentChoice = DEFAULT_CHOICE,
    seed: int = 0,
) -> Recommendation:
    """Ask one agent what to do in the observed position.

    Args:
        state: The tracked position. It is frozen and is not modified: a
            recommendation is a question, and the move is recorded only when the
            operator says they played it.
        choice: Which strategy to ask, and its tie-break salt.
        seed: Seed for the agent's generator, before the choice's salt is mixed in.
            Derive it from the session revision so the same position always produces
            the same suggestion.

    Returns:
        The suggestion, its reasoning, and what it leaves behind.

    Raises:
        ObservationError: If the position cannot be advised on, it is not my
            turn, ranks are outstanding, the deck has not been counted, or the game
            is over, or if the agent finished without a legal candidate. Call
            :func:`~shed.companion.observed.advice_blockers` first to tell the
            operator which.
    """
    if state.to_act != ME:
        raise ObservationError("The companion only advises on your own turn")
    if state.pending:
        raise ObservationError("Record the cards you are holding before asking for advice")
    view = build_player_view(state, ME)
    options = view.legal_moves
    if not options:
        raise ObservationError("The observed position offers no legal move")

    recorder = _Recorder(options)
    agent = build_agent(choice.spec, seed=choice.seed_for(seed))
    agent.think(view, recorder)
    move = recorder.chosen
    if move is None or not recorder.closed:
        raise ObservationError(
            f"The {choice.profile.label} agent finished without a legal suggestion"
        )

    match move:
        case Play():
            reasoning = _play_reasoning(choice.spec.kind, move, options)
        case Reveal():
            reasoning = (
                "Your hand and face-up cards are gone, so a face-down card is the "
                "only thing left to play. They are indistinguishable, so every one "
                "is the same decision."
            )
        case _:
            reasoning = (
                "Nothing you hold can go on this pile, so picking it up is the only "
                "legal action. There is no voluntary pickup."
            )
    return Recommendation(
        agent=choice,
        move=move,
        headline=describe_move(move),
        reasoning=reasoning,
        effect=_effect_note(state, move),
        caveat=choice.profile.caveat,
        considered=len(options),
        notes=view_gaps(state),
    )
