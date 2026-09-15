"""The offline phone companion: track a physical Shed game, ask the agent for a move.

This package is for a game played with real cards across a table. It never deals,
never shuffles, and never knows a card nobody looked at. What it does is keep the
position the two of you can actually see -- my hand, both face-up sets, the pile,
the counts of everything else -- and hand that to the shipped greedy agent for a
suggestion.

The layers run one way and each does one thing:

* :mod:`shed.companion.observed` is the model and the rules. It imports the
  profile's legality, constraint transition, burn rule, batch generation, and
  active-zone ordering from :mod:`shed.engine` rather than restating them, so the
  companion cannot drift from the engine the agent was written against.
* :mod:`shed.companion.advice` builds a :class:`~shed.engine.PlayerView` from
  observed information and calls the chosen agent directly -- any kind in
  :data:`~shed.agents.AGENT_KINDS`, picked before the game starts and recorded in
  the document. No unobserved card becomes a card; the timed multiprocessing
  runner is not involved.
* :mod:`shed.companion.codec` decodes and encodes the JSON the browser sends, which
  is the one place untyped external data is validated.
* :mod:`shed.companion.session` is the recoverable document -- a versioned initial
  position and an ordered observation log -- and the pure function from it to the
  screen.
* :mod:`shed.companion.api` and :mod:`shed.companion.server` serve that function
  and the bundled assets over loopback, with no session state on the Python side.

Nothing outside the engine and the agents is imported: no match runner, no
multiprocessing, no replay codec, and no third-party package. The standard library
is the whole dependency, which is what makes ``pkg install python`` enough in
Termux.

Start it with ``python -m shed.companion`` and open ``http://127.0.0.1:8000``.
"""

from shed.companion.advice import (
    AGENT_PROFILES,
    DEFAULT_AGENT,
    FAITHFUL_VIEW_FIELDS,
    AgentChoice,
    AgentProfile,
    Recommendation,
    agent_catalogue,
    build_player_view,
    profile_for,
    recommend,
)
from shed.companion.codec import CompanionDataError, decode_event, decode_state
from shed.companion.observed import (
    ME,
    OPPONENT,
    CorrectState,
    ObservationError,
    ObservedState,
    PickUpPile,
    PlayCards,
    RecordCards,
    RevealFaceDown,
    SeatObservation,
    StatePatch,
    apply_event,
    join_game,
    new_game,
    validate_observed,
)
from shed.companion.server import main, serve
from shed.companion.session import (
    COMPANION_SCHEMA_VERSION,
    READABLE_SCHEMA_VERSIONS,
    Session,
    decode_session,
    derive,
    encode_session,
    render,
    revision,
)

__all__ = [
    "AGENT_PROFILES",
    "COMPANION_SCHEMA_VERSION",
    "DEFAULT_AGENT",
    "FAITHFUL_VIEW_FIELDS",
    "ME",
    "OPPONENT",
    "READABLE_SCHEMA_VERSIONS",
    "AgentChoice",
    "AgentProfile",
    "CompanionDataError",
    "CorrectState",
    "ObservationError",
    "ObservedState",
    "PickUpPile",
    "PlayCards",
    "Recommendation",
    "RecordCards",
    "RevealFaceDown",
    "SeatObservation",
    "Session",
    "StatePatch",
    "agent_catalogue",
    "apply_event",
    "build_player_view",
    "decode_event",
    "decode_session",
    "decode_state",
    "derive",
    "encode_session",
    "join_game",
    "main",
    "new_game",
    "profile_for",
    "recommend",
    "render",
    "revision",
    "serve",
    "validate_observed",
]
