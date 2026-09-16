# The phone companion

Track a physical two-player game of Shed on an Android phone, and ask the
shipped greedy agent what it would play. Everything runs on the phone, a
small Python server from this repository and a page Chrome loads from
`127.0.0.1`.

```bash
python -m shed.companion      # then open http://127.0.0.1:8000 in Chrome
```

- [What it is](#what-it-is)
- [Choosing the agent](#choosing-the-agent)
- [Set up Termux](#set-up-termux)
- [Install Python, Git, and this repository](#install-python-git-and-this-repository)
- [Start, stop, and restart](#start-stop-and-restart)
- [Keeping Termux alive on Android](#keeping-termux-alive-on-android)
- [Using it at the table](#using-it-at-the-table)
- [Recovery: undo, corrections, backup](#recovery-undo-corrections-backup)
- [Updating safely](#updating-safely)
- [Limitations](#limitations)
- [How it works](#how-it-works)
- [What was tested, and on what](#what-was-tested-and-on-what)

## What it is

A scorekeeper with an opinion. You play with real cards, you record what
happens, and it keeps the position and asks one of the project's agents,
the same ones the gauntlet measures, what it would do from there.

It never deals, never shuffles, and never invents a card. Your opponent's
hand is a number. Face-down cards are a count. A pile card you never saw
stays unrecorded. When something it needs is missing, it says which
observation to record instead of guessing, and withholds the recommendation
until you record it.

There is no camera recognition, no cloud, no account, and no APK. Nothing
you enter leaves the phone. After the one-time install, the whole thing
works in aeroplane mode.

## Choosing the agent

Both setup forms have an **Advice from** picker listing every agent the
installed package builds, with a line saying what each one does. The list is
read from `shed.agents.AGENT_KINDS` at runtime rather than written into the
page, so an agent added to the package appears in the picker without
touching the companion.

Today that is:

| Agent | What it does |
| --- | --- |
| Greedy (default) | Sheds as many cards as it can, then spends the rank it least wants to keep, from a fixed retention table. |
| Random | Samples uniformly among the legal actions. A floor to measure against. |

The recommendation panel is labelled with whichever you picked, *Greedy
recommendation* or *Random recommendation*, and both the reasoning and the
caveat are that agent's own. Greedy explains a batch in terms of the two
keys it actually sorts by, and random says plainly that it compared nothing.
An agent the companion ships no description for still works, and says so
instead of borrowing somebody else's explanation.

**Tie-break seed** is optional and almost never needed. A suggestion is
normally seeded from the position itself, so asking twice about one position
gives one answer and a screenshot can be reproduced from the exported
document. Setting a seed salts that, which shakes loose a different
arbitrary choice without changing anything about the position. It is worth
something for the random baseline. In greedy it is visible only between
interchangeable face-down cards.

The choice is made before the game starts and is recorded in the session
document, so it survives a reload, a server restart, and an export/import
round trip. A suggestion in the history always came from the strategy the
document names. To play the same game with a different agent, export it,
start a new game with the other agent, and import.

### Can every agent advise on an observed game?

Yes for both agents that exist, and that is checked rather than assumed. The
whole abstraction is `Agent.think(view, turn)`, and the shipped baselines
read exactly four things from the view. All four (`legal_moves`, `hand`,
`me.face_up`, and `viewer`) a companion-built view fills as truthfully as
the engine does.

The honest caveat is about agents that do not exist yet. A view built from
observations is thinner than one built from a `GameState`, and no amount of
care fixes that:

| View field | In an observed game |
| --- | --- |
| `discard_pile`, `burned_cards` | Carry only the cards whose ranks were seen, so they understate those piles whenever the table status reports unseen cards. |
| `players[…].hand_count` | Exact. But an opponent's known ranks are dropped, since `PublicPlayerState` has nowhere to put a partly-known hand. |
| `history` | Empty. The companion keeps its own observation log, and the engine's event vocabulary cannot express "a card moved and nobody saw it". |
| `current_ply`, `dealer` | Not observed. The companion joins after the deal. |
| `Arrange` moves | Never generated. The hand/table swap happens physically before tracking starts. |

`shed.companion.advice.FAITHFUL_VIEW_FIELDS` names the fields that are
honest, and `tests/companion/test_agents.py` traces every shipped agent's
field accesses and fails if one reads outside that set. A future agent that
starts consulting the pile will not silently get a thinner truth. The test
will say so, and whoever adds it can decide what the companion should do
about it. `view_gaps()` already reports the same gaps to the operator, under
the suggestion.

## Set up Termux

Termux is a terminal emulator and Linux environment for Android. Install it
from one of the sources the Termux project supports:

- F-Droid, <https://f-droid.org/en/packages/com.termux/>
- GitHub releases, <https://github.com/termux/termux-app/releases>

Two things the Termux project is explicit about, and both bite if ignored:

- Google Play carries an experimental branch. The project describes it as
  Android 11+ only, with missing functionality and bugs compared with the
  stable F-Droid build. Prefer F-Droid or GitHub.
- Never mix sources. APKs from F-Droid, GitHub, and Google Play are signed
  with different keys. Installing Termux from one source and a Termux
  add-on from another fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE` or
  `INSTALL_FAILED_SHARED_USER_INCOMPATIBLE`. Pick one source and take
  everything from it.

Termux supports Android 7 and above. On Android 12 and later see
[Keeping Termux alive on Android](#keeping-termux-alive-on-android) before a
long session. The OS kills background ("phantom") processes, which can end
the server mid-game with `[Process completed (signal 9)]`.

Check the project's own installation page for anything that has changed
since this was written:
<https://github.com/termux/termux-app#installation>.

## Install Python, Git, and this repository

Open Termux and run:

```bash
pkg upgrade                     # refresh the package lists and installed packages
pkg install python git          # Python 3 and Git
python --version                # must be 3.12 or newer
```

Then clone and install:

```bash
git clone https://github.com/maxvandenhoven/shed
cd shed
pip install .
```

`pip install .` downloads `hatchling` to build the wheel. That is the only
download, and it is the last one. The companion itself has no dependencies
at all. The package needs nothing outside the Python standard library, so
there is nothing to compile, no Rust toolchain, no Node, and no service to
reach. That is also why `pip install` is quick on a phone.

If `pip install .` gives you trouble, an old `pip`, no write access, or a
broken build, you can skip installing entirely and run from the checkout:

```bash
cd ~/shed
PYTHONPATH=src python -m shed.companion
```

Everything below works the same either way. Only the command changes.

## Start, stop, and restart

Start it:

```bash
python -m shed.companion
```

```
Shed companion running on http://127.0.0.1:8000
Open that address in Chrome on this phone. Press Ctrl-C to stop.
```

Now open Chrome on the same phone and go to `http://127.0.0.1:8000`. Type
the address in full, since Chrome will otherwise search for it. Add it to
your home screen (⋮, then *Add to Home screen*) and it opens in one tap next
time.

Stop it with `Ctrl-C` in Termux (the volume-down key is Termux's Ctrl). It
shuts the socket down cleanly, so starting it again on the same port works
immediately.

Restart it with the same command. Your game survives. The server keeps no
game state at all. The whole session lives in Chrome's local storage, and
the page sends it back whenever it reconnects. Stop the server mid-turn and
the page shows a reconnect banner, keeps the game and whatever you had
half-typed, and recovers by itself within a few seconds of the server coming
back. Closing Chrome, rebooting the phone, or reinstalling the package makes
no difference either.

Another port, if 8000 is taken:

```bash
python -m shed.companion --port 8181     # then open http://127.0.0.1:8181
```

If the port is in use you get one line saying so and another port to try,
not a traceback. `--log-requests` prints a line per request when something
is not behaving. `--host` exists but should stay at its default. `127.0.0.1`
means only this phone can reach the server, and anything else puts it on
whatever network the phone is on.

## Keeping Termux alive on Android

Android aggressively stops background work. The companion is unusual in that
being stopped costs you nothing, because the game is in the browser, but
reconnecting mid-game is a nuisance. Three settings help, in order of how
much they matter:

1. Acquire a wake lock. In the Termux notification, tap **Acquire
   wakelock**, or run `termux-wake-lock` before starting the server (and
   `termux-wake-unlock` when you are done). This is what keeps the device
   from suspending Termux while the screen is off.
2. Exempt Termux from battery optimization. Android *Settings, Apps, Termux,
   Battery, Unrestricted* (the exact path varies by manufacturer, and
   Samsung, Xiaomi, and OnePlus each hide it somewhere different). While you
   are there, allow Termux to run in the background.
3. Android 12 and later: phantom process trimming. Android kills background
   child processes beyond a limit, and kills processes it judges to be using
   too much CPU. On Android 12L and 13+ the switch is in *Developer options,
   Feature flags, `settings_enable_monitor_phantom_procs`*. Turn it off. On
   plain Android 12 there is no UI for it, and it takes `adb` from a
   computer:

   ```bash
   adb shell "/system/bin/device_config set_sync_disabled_for_tests persistent"
   adb shell "/system/bin/device_config put activity_manager max_phantom_processes 2147483647"
   adb shell settings put global settings_enable_monitor_phantom_procs false
   ```

   The companion runs as a single process and does not spawn workers, so it
   is much less exposed to this than a typical Termux workload, but a long
   session with the screen off can still be caught by the CPU-usage rule.

The simplest habit that avoids all of it is to keep the Termux notification
visible and switch between Chrome and Termux rather than closing Termux.

## Using it at the table

Starting a game. Deal as usual and make your hand/table swaps physically,
then enter the position that swap produced. That is your three hand cards,
your three face-up cards, your opponent's three face-up cards, and who
actually played first. Face-down cards are never entered, because nobody has
seen them.

Suits are never asked for. Suits affect neither legality nor strength, so
the rank is the whole identity worth recording.

Joining a game already in progress is the other setup tab. Enter what you
can see and leave the rest unknown. That is your hand, both face-up sets,
the counts of face-down cards, how many cards your opponent is holding, the
pile (with a count for cards you never saw), and the restriction currently
in force. If you have not counted the deck, tick the box. The companion will
then tell you that it needs the count before it can advise, rather than
assuming a number.

During play:

- **Greedy recommendation** shows the action and why that heuristic chose
  it. It records nothing until you tap **I played this**. Play the cards
  first, then tell it.
- **My cards** lists every legal alternative as a button, and your hand,
  face-up cards, and face-down count underneath. When only face-down cards
  are left, it asks for the rank that came up before resolving whether it
  went down.
- **Opponent action** is a rank, how many, and confirm, plus a prominent
  **They picked up the pile**. Everything else (their refills, whose turn it
  is, burns, retained turns) is worked out for you.
- **Table status** shows whose turn it is, the restriction, the pile, the
  deck, each player's remaining cards, and an explicit list of what is not
  known.
- Whenever you draw cards, the companion asks for their ranks before
  anything else can happen. The cards are already counted into your hand.
  Only their identities are outstanding.

## Recovery: undo, corrections, backup

The game is a list of observations, and the position on screen is always
that list folded up. Every recovery feature follows from that:

- **Undo** removes the last entry, whatever it was, a play, a pickup, a
  recorded draw, or a correction. It is always available.
- **Action history** is the list itself, newest first.
- **Correct state** records an explicit correction as its own entry. It
  shows in the history and Undo takes it back. Nothing is rewritten behind
  your back. Use it when you missed a move, mistyped a rank, or need to
  record a recount.
- A correction that could not describe a real table is refused with the
  reason, and everything you typed stays on screen. The commonest one is
  card accounting, since the recorded cards have to add up to 54.
- **Show export** prints the whole session as JSON. Copy it somewhere before
  anything drastic. **Restore** (on the setup screen) reads one back. An
  import that is not readable leaves the current game untouched.

The session is saved to the browser's local storage after every accepted
entry, so a refresh, a browser restart, or a phone reboot all pick up where
you were.

## Updating safely

```bash
cd ~/shed
git status                  # check for local edits first
git pull
pip install . --upgrade     # only needed if you installed rather than using PYTHONPATH
```

Then hard-reload the page in Chrome (pull down to refresh; the server sends
`Cache-Control: no-store`, so a plain reload is normally enough).

Your saved game is not touched by an update. It carries a schema version.
Older versions the current code still knows how to read are upgraded in
place (a document written before the agent picker existed reads as a greedy
game and is rewritten on the first reply), and a version it cannot read is
refused rather than half-understood. The page then offers the unreadable
data for you to copy before you start a new game. `GET /api/health` lists
the versions the running server accepts. Export a running game before
pulling if it matters to you.

To update Termux's own packages, `pkg upgrade`. Do that when you are not
mid-game, since it can restart the shell.

## Limitations

The advice is a baseline. `GreedyAgent`, the default, makes one pass over
the legal moves and scores them on two things, shedding as many cards as
possible and then spending the rank it least wants to keep, from a fixed
retention table. It does not count cards, read the pile, look ahead, or
model your opponent. It is the floor a stronger agent should beat.
[`docs/agent-baselines.md`](agent-baselines.md) records what the baselines
actually score. No win probability is shown, because there is nothing here
that could compute one.

The agent is fixed once a game starts. Switching mid-game would make the
history ambiguous about which strategy suggested what, so the picker is on
the setup screen only. Export and re-import to change it.

Observations are only as good as what you record. The companion cannot see
the table. If you forget to record a move, the position drifts and it may
start refusing entries that are perfectly legal at the table. That is the
correction workflow's job, and the refusal is the signal.

Unknown information stays unknown, and that has a price. Joining mid-game
without a deck count means no advice until you count it. A pile with unseen
cards stays partly unknown until it burns or is picked up. Your opponent's
hand is a count plus whatever a public transfer proved.

Two players only. The engine supports two to five. The companion tracks the
two-player game.

One phone, one game. There is no sync between devices. The game lives in the
browser you entered it on.

## How it works

- `src/shed/companion/observed.py` holds the observed position and the
  reducer that advances it. Rank legality, the constraint transition, the
  burn rule, batch generation, and the active-zone ordering are imported
  from `shed.engine` (`can_play_rank`, `constraint_after`, `burn_reason`,
  `legal_batches`, `active_zone_for_counts`) rather than restated, so the
  companion and the engine cannot drift apart.
- `src/shed/companion/advice.py` builds a `PlayerView` from observed
  information and calls the chosen agent directly, in this process. The
  timed multiprocessing runner is not involved. No unobserved card becomes a
  `Card`, and the identifiers and the single suit the view carries are
  bookkeeping the type demands, never claims about physical cards. It also
  owns the agent catalogue, the per-agent explanations, and
  `FAITHFUL_VIEW_FIELDS`.
- `src/shed/companion/codec.py` holds the JSON vocabulary, and is the only
  place untyped external data is validated.
- `src/shed/companion/session.py` holds the session document (a versioned
  initial position plus an ordered observation log) and the pure function
  from it to the screen.
- `src/shed/companion/api.py`, `server.py`, and `static/` are a stateless
  local HTTP server and the bundled page. Assets are an explicit allowlist,
  served from the package. There is no CDN, no remote font, and no request
  off the phone.

Nothing outside `shed.engine` and `shed.agents` is imported, and no
third-party package at all.

## What was tested, and on what

Everything described here was developed and verified on desktop Linux:

- `uv run pytest` runs the full suite, including the companion's own tests
  for unknown information, legal and illegal batches, special-card
  restrictions and burns, pickups and opponent-card tracking, refills
  against a nearly empty deck, pending draws and blind reveals, the zone
  transitions, undo, corrections, replay and persistence round trips, an
  incomplete ongoing-game state, and that a recommendation is legal and
  changes nothing.
- The server was driven over HTTP, covering bundled assets, status codes,
  the body limit, an occupied port, and a clean shutdown.
- Every agent the package builds was run over positions covering a hand
  batch, a face-up batch, a blind reveal and a forced pickup, with a
  field-access trace asserting none of them reads a view field an observed
  game cannot fill faithfully.
- The page was driven in headless Chromium at a 390x844 phone viewport,
  through setup, a recommendation, *I played this*, recording a drawn rank,
  the opponent's action, undo, and a reload, plus joining a game, a rejected
  correction and a good one, export and re-import, the blind-reveal path,
  and picking a non-default agent and confirming the choice survives a
  reload. The server was killed mid-game to confirm the reconnect banner
  appears, the game and the typed input survive, and it recovers by itself
  when the server returns.

It has not been run on an Android phone. Termux, Chrome on Android, the
battery and phantom-process settings, and the on-screen ergonomics are all
documented from the Termux project's own guidance and from testing an
equivalent desktop setup, never from a device. Treat the Android specifics
as instructions to follow rather than as a verified path, and expect to
adjust the battery settings for your manufacturer.
