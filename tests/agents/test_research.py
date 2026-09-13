"""Tests for the research agent.

The agent is developed by experiment, so these tests pin the things that must
hold whatever the heuristic becomes -- it decides only from ``view.legal_moves``,
it closes every decision the engine can ask for with exactly one legal final
submission, and it is reproducible from its seed -- alongside the choices the
current version makes in crafted positions. A strategy change is expected to
rewrite the second group and to leave the first untouched.
"""

import random
from collections import Counter
from dataclasses import replace

import pytest

from shed.agents import AgentSpec, ResearchAgent, build_agent
from shed.agents.greedy import RETENTION_SCORE
from shed.agents.research import (
    ENDGAME_WIDTH,
    RETENTION,
    _block_chance,
    _determinize,
    _race_multiplier,
    _shortlist,
    _static_choice,
    _unseen_ranks,
)
from shed.engine import (
    Arrange,
    AtLeast,
    GameState,
    Move,
    Phase,
    PickUp,
    Play,
    PlayerId,
    PlayerView,
    Rank,
    Reveal,
    SlotId,
    Zone,
    validate_decision_boundary,
)
from tests.agents.conftest import FakeTurn, play_baseline_match
from tests.conftest import DeckPicker, build_play_state, build_setup_state

FIRST_SEAT: PlayerId = PlayerId(0)
"""Seat zero, the actor in every crafted position here."""

SECOND_SEAT: PlayerId = PlayerId(1)
"""The opponent in the crafted two-player positions here."""


def _agent(seed: int = 7) -> ResearchAgent:
    """Build a research agent through the factory.

    Going through the factory rather than the constructor keeps the tests on the
    path a match actually uses, so a missing registration fails here too.

    Args:
        seed: Seed for the agent's generator.

    Returns:
        A freshly built research agent.
    """
    built = build_agent(AgentSpec(kind="research", name="research-0"), seed=seed)
    assert isinstance(built, ResearchAgent)
    return built


def _decide(view: PlayerView, *, seed: int = 7) -> Move:
    """Run one decision and return the move the agent finalized.

    Args:
        view: The observation to decide on.
        seed: Seed for the agent's generator.

    Returns:
        The single move the agent submitted.

    Raises:
        AssertionError: If the agent did not close the turn with exactly one
            final submission.
    """
    turn = FakeTurn()
    _agent(seed).think(view, turn)
    assert [submission.final for submission in turn.submissions] == [True]
    assert turn.selected in view.legal_moves
    return turn.selected


def _actor_view(state: GameState) -> PlayerView:
    """Observe a state as its current actor.

    Args:
        state: A live state.

    Returns:
        The actor's view, carrying their legal moves.
    """
    actor = state.current_player
    assert actor is not None
    return state.observe(actor)


def test_the_factory_builds_the_research_kind() -> None:
    """The kind is registered, so a lineup can name it."""
    assert "research" in AgentSpec(kind="research", name="research-0").kind
    assert isinstance(_agent(), ResearchAgent)


def test_it_refuses_a_view_with_no_choices(picker: DeckPicker) -> None:
    """Observing as a non-actor yields no moves, and the agent invents none."""
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: picker.take(Rank.FIVE), SECOND_SEAT: picker.take(Rank.KING)},
    )
    waiting = state.observe(SECOND_SEAT)
    assert waiting.legal_moves == ()

    with pytest.raises(ValueError, match="no legal moves"):
        _agent().think(waiting, FakeTurn())


def test_it_decides_only_from_the_view(picker: DeckPicker) -> None:
    """A narrowed legal-move tuple narrows the agent.

    The view is cut down to one option the engine would not have offered alone,
    so an agent that recomputed legality from the cards it can see would ignore
    the narrowing.
    """
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: [*picker.take(Rank.FIVE, 2), *picker.take(Rank.NINE, 2)],
            SECOND_SEAT: picker.take(Rank.KING),
        },
    )
    view = _actor_view(state)
    assert len(view.legal_moves) > 1
    only = view.legal_moves[-1]

    assert _decide(replace(view, legal_moves=(only,))) == only


def test_setup_keeps_the_hardest_cards_face_up(picker: DeckPicker) -> None:
    """Arrangement is a choice among the six visible cards, and it is scored.

    The dealt split is deliberately wrong -- the cheap cards start face up --
    so keeping the deal would be a different move from the one scoring picks.
    """
    state = build_setup_state(
        picker,
        hands={
            FIRST_SEAT: picker.many([Rank.TEN, Rank.TWO, Rank.JOKER]),
            SECOND_SEAT: picker.any_cards(3),
        },
        face_up={
            FIRST_SEAT: picker.many([Rank.THREE, Rank.FOUR, Rank.FIVE]),
            SECOND_SEAT: picker.any_cards(3),
        },
    )
    # Seat 1 arranges first when seat 0 deals; keep its deal to reach seat 0.
    first = _actor_view(state)
    assert first.viewer == SECOND_SEAT
    state.apply_move(_decide(first))

    view = _actor_view(state)
    assert view.phase is Phase.SETUP
    chosen = _decide(view)

    assert isinstance(chosen, Arrange)
    kept = {card.rank for card in view.hand if card.id in chosen.face_up_cards}
    assert kept == {Rank.TEN, Rank.TWO, Rank.JOKER}


def test_it_plays_from_the_hand_while_the_hand_has_cards(picker: DeckPicker) -> None:
    """The engine picks the active zone, and the agent never names another."""
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: picker.take(Rank.FIVE, 2), SECOND_SEAT: picker.take(Rank.KING)},
        face_up={FIRST_SEAT: picker.take(Rank.ACE, 3)},
    )

    chosen = _decide(_actor_view(state))

    assert isinstance(chosen, Play)
    assert chosen.source is Zone.HAND


def test_it_plays_the_face_up_zone_when_hand_and_deck_are_empty(picker: DeckPicker) -> None:
    """Face-up play is an ordinary scored decision, not a special case."""
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: [], SECOND_SEAT: picker.take(Rank.KING)},
        face_up={FIRST_SEAT: picker.many([Rank.FOUR, Rank.QUEEN, Rank.TEN])},
        draw_count=0,
    )

    chosen = _decide(_actor_view(state))

    assert isinstance(chosen, Play)
    assert chosen.source is Zone.FACE_UP


def test_it_reveals_a_face_down_slot_when_that_is_all_that_is_left(picker: DeckPicker) -> None:
    """Blind reveals are indistinguishable, so any offered slot is acceptable."""
    hidden = {SlotId(0): picker.one(Rank.ACE), SlotId(2): picker.one(Rank.THREE)}
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: [], SECOND_SEAT: picker.take(Rank.KING)},
        face_down={FIRST_SEAT: hidden},
        draw_count=0,
    )
    view = _actor_view(state)
    assert view.hand == ()

    chosen = _decide(view)

    assert isinstance(chosen, Reveal)
    assert chosen.slot in hidden


def test_it_takes_the_forced_pickup_when_nothing_is_playable(picker: DeckPicker) -> None:
    """Pickup is never chosen for its merits; it is the only move on offer."""
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: picker.take(Rank.THREE), SECOND_SEAT: picker.take(Rank.KING)},
        constraint=AtLeast(Rank.KING),
    )
    view = _actor_view(state)
    assert view.legal_moves == (PickUp(),)

    assert _decide(view) == PickUp()


def test_it_spends_the_cheapest_rank_rather_than_the_biggest_batch(
    picker: DeckPicker,
) -> None:
    """Rank decides before batch size, which is what parts it from greedy.

    Three kings are a larger batch than one four, and the four is the card worth
    less in hand, so the two orderings disagree here by construction.
    """
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: [*picker.take(Rank.FOUR, 1), *picker.take(Rank.KING, 3)],
            SECOND_SEAT: picker.take(Rank.ACE),
        },
    )

    chosen = _decide(_actor_view(state))

    assert chosen == Play(Zone.HAND, Rank.FOUR, 1)


def test_it_spends_every_copy_of_the_rank_it_settles_on(picker: DeckPicker) -> None:
    """Batch size still decides, once the cheapest rank has been chosen."""
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: [*picker.take(Rank.FOUR, 2), *picker.take(Rank.KING, 2)],
            SECOND_SEAT: picker.take(Rank.ACE),
        },
    )

    chosen = _decide(_actor_view(state))

    assert chosen == Play(Zone.HAND, Rank.FOUR, 2)


def test_the_table_differs_from_greedy_only_in_the_seven() -> None:
    """The retention table is this agent's own, and its one edit is deliberate.

    Pinning the difference keeps an accidental divergence from the baseline
    visible: a later experiment that retunes the table is expected to rewrite
    this test along with the entry it changes.
    """
    differences = {
        rank: (RETENTION_SCORE[rank], RETENTION[rank])
        for rank in RETENTION
        if RETENTION_SCORE[rank] != RETENTION[rank]
    }

    assert differences == {Rank.SEVEN: (17, 10)}
    assert set(RETENTION) == set(RETENTION_SCORE)


def test_it_spends_a_seven_before_a_jack(picker: DeckPicker) -> None:
    """The cheaper seven is a behaviour change, not just a number.

    Greedy scores a seven 17 and a jack 11, so it would spend the jack; this
    agent scores the seven 10 and spends it.
    """
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: picker.many([Rank.SEVEN, Rank.JACK]),
            SECOND_SEAT: picker.take(Rank.ACE),
        },
    )

    chosen = _decide(_actor_view(state))

    assert chosen == Play(Zone.HAND, Rank.SEVEN, 1)


def test_it_keeps_its_specials_when_an_ordinary_rank_answers(picker: DeckPicker) -> None:
    """A ten is worth more held than spent on a pile an eight already answers."""
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: picker.many([Rank.EIGHT, Rank.TEN]),
            SECOND_SEAT: picker.take(Rank.KING),
        },
        constraint=AtLeast(Rank.SIX),
    )

    chosen = _decide(_actor_view(state))

    assert chosen == Play(Zone.HAND, Rank.EIGHT, 1)


def test_the_unseen_pool_is_the_deck_minus_what_this_seat_has_been_shown(
    picker: DeckPicker,
) -> None:
    """The block estimate's pool is derived, and it is derived correctly.

    The pool must be exactly the cards sitting where this seat cannot see them:
    the draw pile, every face-down slot -- its own included -- and the other
    seats' hands. It is checked against the authoritative state, which the agent
    never receives.
    """
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: picker.take(Rank.FIVE, 2),
            SECOND_SEAT: picker.take(Rank.KING, 2),
        },
        face_up={FIRST_SEAT: picker.take(Rank.ACE, 3)},
        face_down={FIRST_SEAT: {SlotId(0): picker.one(Rank.THREE)}},
        discard=picker.take(Rank.FOUR, 1),
        draw_count=9,
    )
    view = _actor_view(state)

    unseen = _unseen_ranks(view)

    hidden: Counter[Rank] = Counter(card.rank for card in state.draw_pile)
    for seat in state.seat_order:
        player = state.players[seat]
        hidden.update(card.rank for card in player.face_down.values())
        if seat != FIRST_SEAT:
            hidden.update(card.rank for card in player.hand)
    assert unseen == hidden
    assert sum(unseen.values()) == view.draw_count + 1 + 2


def test_a_public_face_up_zone_makes_the_block_estimate_exact(picker: DeckPicker) -> None:
    """An opponent playing off the table is read, not guessed.

    With an empty hand the next seat must play its face-up cards, which everyone
    can see, so the estimate is a certainty in both directions.
    """
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: picker.take(Rank.SIX), SECOND_SEAT: []},
        face_up={SECOND_SEAT: picker.take(Rank.FOUR, 2)},
        draw_count=0,
    )
    view = _actor_view(state)
    unseen = _unseen_ranks(view)

    assert _block_chance(view, unseen, AtLeast(Rank.KING)) == 1.0
    assert _block_chance(view, unseen, AtLeast(Rank.THREE)) == 0.0


def test_it_spends_a_dearer_card_to_bury_an_opponent_who_cannot_answer(
    picker: DeckPicker,
) -> None:
    """Pressure outscores frugality when the pile makes a block expensive.

    The opponent's only cards are public fours, so an ace blocks them and a three
    does not. A three is the cheaper card by eleven retention points, and the
    twelve-card pile a block would hand over is worth more than that. They hold
    three cards, clear of match point, so the pile term decides this alone.
    """
    hand = picker.many([Rank.THREE, Rank.ACE])
    theirs = picker.take(Rank.FOUR, 3)
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: hand, SECOND_SEAT: []},
        face_up={SECOND_SEAT: theirs},
        discard=picker.any_cards(12),
        draw_count=0,
    )

    chosen = _decide(_actor_view(state))

    assert chosen == Play(Zone.HAND, Rank.ACE, 1)


def test_a_pile_too_small_to_matter_leaves_the_cheap_play_standing(
    picker: DeckPicker,
) -> None:
    """The same position with a small pile is decided by retention alone."""
    hand = picker.many([Rank.THREE, Rank.ACE])
    theirs = picker.take(Rank.FOUR, 3)
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: hand, SECOND_SEAT: []},
        face_up={SECOND_SEAT: theirs},
        discard=picker.any_cards(2),
        draw_count=0,
    )

    chosen = _decide(_actor_view(state))

    assert chosen == Play(Zone.HAND, Rank.THREE, 1)


def test_it_stays_frugal_when_no_play_applies_pressure(picker: DeckPicker) -> None:
    """With nothing to gain, the cheapest rank still wins.

    The opponent's public twos answer every constraint there is, so each
    candidate earns a zero block bonus and the retention table decides alone.
    """
    hand = picker.many([Rank.THREE, Rank.ACE])
    theirs = picker.take(Rank.TWO, 2)
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: hand, SECOND_SEAT: []},
        face_up={SECOND_SEAT: theirs},
        discard=picker.any_cards(12),
        draw_count=0,
    )

    chosen = _decide(_actor_view(state))

    assert chosen == Play(Zone.HAND, Rank.THREE, 1)


def test_the_race_multiplier_follows_the_card_difference(picker: DeckPicker) -> None:
    """Falling behind raises the price this agent will pay for pressure.

    The multiplier is read directly, because it is the only thing that differs
    between a seat with nine cards against one and a seat with one against nine.
    """
    behind = build_play_state(
        picker,
        hands={
            FIRST_SEAT: [*picker.take(Rank.FIVE, 3), *picker.take(Rank.SIX, 3)],
            SECOND_SEAT: picker.take(Rank.KING),
        },
        draw_count=0,
    )
    spare = DeckPicker()
    ahead = build_play_state(
        spare,
        hands={
            FIRST_SEAT: spare.take(Rank.KING),
            SECOND_SEAT: [*spare.take(Rank.FIVE, 3), *spare.take(Rank.SIX, 3)],
        },
        draw_count=0,
    )
    level = DeckPicker()
    even = build_play_state(
        level,
        hands={FIRST_SEAT: level.take(Rank.KING), SECOND_SEAT: level.take(Rank.FIVE)},
        draw_count=0,
    )

    assert _race_multiplier(_actor_view(behind)) > 1.0
    assert _race_multiplier(_actor_view(ahead)) < 1.0
    assert _race_multiplier(_actor_view(even)) == pytest.approx(1.0)


def test_being_far_behind_buys_a_block_a_cheap_play_would_not() -> None:
    """The same pile and the same hand decide differently by race position.

    Both positions offer a three and an ace against an opponent whose only cards
    are public kings, with a nine-card pile. A king answers everything here
    except the ace, so the ace is the one play that certainly blocks, and three
    of them keeps that seat clear of match point. Level on cards the cheap three
    is right; six cards behind, the block is worth the eleven extra retention
    points the ace costs.

    This asks the static scorer rather than the agent. The position has an empty
    draw pile, so the agent would search it, and what a rollout concludes is a
    different claim from what the score prefers -- which is the claim under test.
    """

    def position(filler: int) -> GameState:
        cards = DeckPicker()
        hand = cards.many([Rank.THREE, Rank.ACE])
        theirs = cards.take(Rank.KING, 3)
        padding = [
            *cards.take(Rank.FIVE, min(filler, 3)),
            *cards.take(Rank.SIX, max(filler - 3, 0)),
        ]
        return build_play_state(
            cards,
            hands={FIRST_SEAT: [*hand, *padding], SECOND_SEAT: []},
            face_up={SECOND_SEAT: theirs},
            discard=cards.any_cards(9),
            draw_count=0,
        )

    level = _static_choice(_actor_view(position(0)), random.Random(7))
    behind = _static_choice(_actor_view(position(6)), random.Random(7))

    assert level == Play(Zone.HAND, Rank.THREE, 1)
    assert behind == Play(Zone.HAND, Rank.ACE, 1)


def test_a_seat_one_card_from_winning_is_worth_blocking_at_a_price(
    picker: DeckPicker,
) -> None:
    """A small pile is the wrong measure of a block against a seat about to win.

    The opponent holds one public king and nothing else, so a king answers and
    an ace does not. The pile is two cards, far too small for the ordinary bonus
    to buy an ace over a three; the match-point premium is what does.
    """
    hand = picker.many([Rank.THREE, Rank.ACE])
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: hand, SECOND_SEAT: []},
        face_up={SECOND_SEAT: picker.take(Rank.KING)},
        discard=picker.any_cards(2),
        draw_count=0,
    )
    view = _actor_view(state)
    assert sum(len(public.face_up) + public.hand_count for public in view.players[1:]) == 1

    assert _decide(view) == Play(Zone.HAND, Rank.ACE, 1)


def test_the_premium_does_not_apply_to_a_seat_with_cards_to_spare(
    picker: DeckPicker,
) -> None:
    """The same shape with the opponent further from home keeps the cheap play."""
    hand = picker.many([Rank.THREE, Rank.ACE])
    state = build_play_state(
        picker,
        hands={FIRST_SEAT: hand, SECOND_SEAT: []},
        face_up={SECOND_SEAT: picker.take(Rank.KING, 3)},
        discard=picker.any_cards(2),
        draw_count=0,
    )

    assert _decide(_actor_view(state)) == Play(Zone.HAND, Rank.THREE, 1)


def test_the_same_seed_decides_the_same_way(picker: DeckPicker) -> None:
    """Tie-breaking is seeded, so a decision is reproducible from the seed."""
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: picker.many([Rank.THREE, Rank.FOUR, Rank.FIVE]),
            SECOND_SEAT: picker.take(Rank.KING),
        },
    )
    view = _actor_view(state)

    assert _decide(view, seed=99) == _decide(view, seed=99)


def test_a_sampled_world_is_one_this_observation_could_have_come_from(
    picker: DeckPicker,
) -> None:
    """Determinization is checked against the engine, not just against itself.

    A sample must pass the engine's own decision-boundary invariants and must
    offer the viewer exactly the moves the real position offered; if it did not,
    the search would be reasoning about a game that could not be this one. The
    hidden zones must match the public counts, and the viewer's own hand must
    come back untouched.
    """
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: [*picker.take(Rank.FIVE, 2), *picker.take(Rank.NINE, 1)],
            SECOND_SEAT: picker.take(Rank.KING, 2),
        },
        face_up={FIRST_SEAT: picker.take(Rank.ACE, 2)},
        face_down={FIRST_SEAT: {SlotId(0): picker.one(Rank.THREE)}},
        discard=picker.take(Rank.FOUR, 1),
        draw_count=7,
    )
    view = _actor_view(state)
    rng = random.Random(3)

    for _ in range(25):
        world = _determinize(view, rng)

        validate_decision_boundary(world)
        assert set(world.get_legal_moves()) == set(view.legal_moves)
        assert [card.id for card in world.players[FIRST_SEAT].hand] == [c.id for c in view.hand]
        assert len(world.draw_pile) == view.draw_count
        for public in view.players:
            held = world.players[public.player]
            assert len(held.hand) == public.hand_count
            assert tuple(sorted(held.face_down)) == public.face_down_slots
            assert [c.id for c in held.face_up] == [c.id for c in public.face_up]


def test_the_search_runs_only_once_the_draw_pile_is_empty(picker: DeckPicker) -> None:
    """A shortlist is only built where a rollout can reach the end of the game."""
    agent = _agent()
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: [*picker.take(Rank.FIVE, 2), *picker.take(Rank.KING, 2)],
            SECOND_SEAT: picker.take(Rank.ACE, 2),
        },
        draw_count=8,
    )
    with_deck = _actor_view(state)

    assert not agent._searchable(with_deck)
    assert agent._searchable(replace(with_deck, draw_count=0))


def test_a_search_failure_leaves_the_static_choice_standing(picker: DeckPicker) -> None:
    """Search is an optimization; a fault in it must not cost the decision."""
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: [*picker.take(Rank.FIVE, 2), *picker.take(Rank.KING, 2)],
            SECOND_SEAT: picker.take(Rank.ACE, 2),
        },
        draw_count=0,
    )
    view = _actor_view(state)
    agent = _agent()
    expected = _static_choice(view, random.Random(7))

    class Exploding:
        """A turn whose clock raises, which the search must survive."""

        def remaining_seconds(self) -> float:
            raise RuntimeError("clock failed")

        def submit(self, move: Move, *, final: bool = False) -> None:
            raise AssertionError("the search must not submit")

    assert agent._search(view, expected, Exploding()) == expected


def test_the_shortlist_keeps_the_static_pick_and_stays_within_its_width(
    picker: DeckPicker,
) -> None:
    """Whatever else it compares, the incumbent is always on the list."""
    state = build_play_state(
        picker,
        hands={
            FIRST_SEAT: [
                *picker.take(Rank.THREE, 1),
                *picker.take(Rank.SIX, 1),
                *picker.take(Rank.JACK, 1),
                *picker.take(Rank.KING, 1),
            ],
            SECOND_SEAT: picker.take(Rank.ACE, 2),
        },
        draw_count=0,
    )
    view = _actor_view(state)
    fallback = _static_choice(view, random.Random(7))

    short = _shortlist(view, fallback)

    assert fallback in short
    assert len(short) <= ENDGAME_WIDTH
    assert all(move in view.legal_moves for move in short)


def test_it_plays_whole_matches_against_greedy_and_they_mostly_terminate() -> None:
    """Every decision of a full match is legal, and the games do end.

    This is the end-to-end check that the agent covers the decision shapes the
    engine asks for over a complete game rather than in a crafted position.

    Termination is asserted as a majority rather than for every deal. A runaway
    is a documented property of the profile -- two agents that never clear the
    pile can recirculate the same cards indefinitely, which is why the runner has
    a decision bound at all -- so one long deal is a fact about ``shed-v1`` and
    not a defect here. What would be a defect is a strategy that stops finishing
    games, since a truncated match wins nothing.
    """
    specs = (
        AgentSpec(kind="research", name="research-0"),
        AgentSpec(kind="greedy", name="greedy-1"),
    )
    shapes: set[type] = set()
    finished = 0

    for deal_seed in range(10):
        log = play_baseline_match(specs, deal_seed=deal_seed, action_limit=2_000)

        if not log.truncated:
            assert log.outcome is not None
            finished += 1
        for decision in log.decisions:
            assert decision.move in decision.view.legal_moves
            assert [submission.final for submission in decision.submissions] == [True]
            if decision.player == FIRST_SEAT:
                shapes.add(type(decision.move))

    assert finished >= 8, f"only {finished}/10 matches finished within the bound"
    assert Arrange in shapes
    assert Play in shapes
