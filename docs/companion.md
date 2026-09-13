# The phone companion

Track a physical two-player game of Shed on an Android phone, and ask the shipped
greedy agent what it would play. Everything runs on the phone: a small Python
server from this repository, and a page Chrome loads from `127.0.0.1`.

```bash
python -m shed.companion      # then open http://127.0.0.1:8000 in Chrome
```

- [What it is, and what it is not](#what-it-is-and-what-it-is-not)
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

## What it is, and what it is not

It is a scorekeeper with an opinion. You play with real cards; you record what
happens; it keeps the position and asks `GreedyAgent` — the same baseline the
gauntlet measures — what it would do from there.

It never deals, never shuffles, and never invents a card. Your opponent's hand is
a number. Face-down cards are a count. A pile card you never saw stays unrecorded.
When something it needs is missing, it says which observation to record instead of
guessing, and withholds the recommendation until you record it.

There is no camera recognition, no cloud, no account, and no APK. Nothing you
enter leaves the phone: after the one-time install, the whole thing works in
aeroplane mode.

## Set up Termux

Termux is a terminal emulator and Linux environment for Android. Install it from
one of the sources the Termux project supports:

- **F-Droid** — <https://f-droid.org/en/packages/com.termux/>
- **GitHub releases** — <https://github.com/termux/termux-app/releases>

Two things the Termux project is explicit about, and both bite if ignored:

- **Google Play carries an experimental branch.** The project describes it as
  Android 11+ only, with missing functionality and bugs compared with the stable
  F-Droid build. Prefer F-Droid or GitHub.
- **Never mix sources.** APKs from F-Droid, GitHub, and Google Play are signed
  with different keys. Installing Termux from one source and a Termux add-on from
  another fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE` or
  `INSTALL_FAILED_SHARED_USER_INCOMPATIBLE`. Pick one source and take everything
  from it.

Termux supports Android 7 and above. On Android 12 and later see
[Keeping Termux alive on Android](#keeping-termux-alive-on-android) before a long
session: the OS kills background ("phantom") processes, which can end the server
mid-game with `[Process completed (signal 9)]`.

Check the project's own installation page for anything that has changed since:
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

`pip install .` downloads `hatchling` to build the wheel — that is the only
download, and it is the last one. **The companion itself has no dependencies at
all**: the package needs nothing outside the Python standard library, so there is
nothing to compile, no Rust toolchain, no Node, and no service to reach. That is
also why `pip install` is quick on a phone.

If `pip install .` gives you trouble — an old `pip`, no write access, a broken
build — you can skip installing entirely and run from the checkout:

```bash
cd ~/shed
PYTHONPATH=src python -m shed.companion
```

Everything below works the same either way; only the command changes.

## Start, stop, and restart

Start it:

```bash
python -m shed.companion
```

```
Shed companion running on http://127.0.0.1:8000
Open that address in Chrome on this phone. Press Ctrl-C to stop.
```

Now open **Chrome on the same phone** and go to `http://127.0.0.1:8000`. Type the
address in full; Chrome will otherwise search for it. Add it to your home screen
(⋮ → *Add to Home screen*) and it opens in one tap next time.

**Stop it** with `Ctrl-C` in Termux (the volume-down key is Termux's Ctrl). It
shuts the socket down cleanly, so starting it again on the same port works
immediately.

**Restart it** with the same command. *Your game survives.* The server keeps no
game state at all — the whole session lives in Chrome's local storage, and the
page sends it back whenever it reconnects. Stop the server mid-turn and the page
shows a reconnect banner, keeps the game and whatever you had half-typed, and
recovers by itself within a few seconds of the server coming back. Closing Chrome,
rebooting the phone, or reinstalling the package makes no difference either.

**Another port**, if 8000 is taken:

```bash
python -m shed.companion --port 8181     # then open http://127.0.0.1:8181
```

If the port is in use you get one line saying so and another port to try, not a
traceback. `--log-requests` prints a line per request when something is not
behaving. `--host` exists but should stay at its default: `127.0.0.1` means only
this phone can reach the server, and anything else puts it on whatever network
the phone is on.

## Keeping Termux alive on Android

Android aggressively stops background work. The companion is unusual in that
being stopped costs you nothing — the game is in the browser — but reconnecting
mid-game is a nuisance. Three settings help, in order of how much they matter:

1. **Acquire a wake lock.** In the Termux notification, tap **Acquire wakelock**,
   or run `termux-wake-lock` before starting the server (and `termux-wake-unlock`
   when you are done). This is what keeps the device from suspending Termux while
   the screen is off.
2. **Exempt Termux from battery optimization.** Android *Settings → Apps → Termux
   → Battery → Unrestricted* (the exact path varies by manufacturer; Samsung,
   Xiaomi, and OnePlus each hide it somewhere different, and Xiaomi and OnePlus
   are aggressive about it). While you are there, allow Termux to run in the
   background.
3. **Android 12 and later: phantom process trimming.** Android kills background
   child processes beyond a limit, and kills processes it judges to be using too
   much CPU. On **Android 12L and 13+** the switch is in
   *Developer options → Feature flags →
   `settings_enable_monitor_phantom_procs`* — turn it off. On plain Android 12
   there is no UI for it, and it takes `adb` from a computer:

   ```bash
   adb shell "/system/bin/device_config set_sync_disabled_for_tests persistent"
   adb shell "/system/bin/device_config put activity_manager max_phantom_processes 2147483647"
   adb shell settings put global settings_enable_monitor_phantom_procs false
   ```

   The companion runs as a single process and does not spawn workers, so it is
   much less exposed to this than a typical Termux workload — but a long session
   with the screen off can still be caught by the CPU-usage rule.

Simplest habit that avoids all of it: keep the Termux notification visible, and
switch between Chrome and Termux rather than closing Termux.

## Using it at the table

**Starting a game.** Deal as usual and make your hand/table swaps physically,
then enter the position that swap produced: your three hand cards, your three
face-up cards, your opponent's three face-up cards, and who actually played
first. Face-down cards are never entered — nobody has seen them.

Suits are never asked for. In the `shed-v1` profile suits affect neither legality
nor strength, so the rank is the whole identity worth recording.

**Joining a game already in progress** is the other setup tab. Enter what you can
see and leave the rest unknown: your hand, both face-up sets, the counts of
face-down cards, how many cards your opponent is holding, the pile (with a count
for cards you never saw), and the restriction currently in force. If you have not
counted the deck, tick the box — the companion will then tell you that it needs
the count before it can advise, rather than assuming a number.

**During play:**

- **Greedy recommendation** shows the action and why that heuristic chose it. It
  records nothing until you tap **I played this** — play the cards first, then
  tell it.
- **My cards** lists every legal alternative as a button, and your hand, face-up
  cards, and face-down count underneath. When only face-down cards are left, it
  asks for the rank that came up before resolving whether it went down.
- **Opponent action** is rank → how many → confirm, plus a prominent **They
  picked up the pile**. Everything else — their refills, whose turn it is, burns,
  retained turns — is worked out for you.
- **Table status** shows whose turn it is, the restriction, the pile, the deck,
  each player's remaining cards, and an explicit list of what is *not* known.
- Whenever you draw cards, the companion asks for their ranks before anything
  else can happen. The cards are already counted into your hand; only their
  identities are outstanding.

## Recovery: undo, corrections, backup

The game is a list of observations, and the position on screen is always that
list folded up. Every recovery feature follows from that:

- **Undo** removes the last entry, whatever it was — a play, a pickup, a recorded
  draw, or a correction. It is always available.
- **Action history** is the list itself, newest first.
- **Correct state** records an explicit correction as its own entry. It shows in
  the history and Undo takes it back; nothing is rewritten behind your back. Use
  it when you missed a move, mistyped a rank, or need to record a recount.
- A correction that could not describe a real table is refused with the reason,
  and everything you typed stays on screen. The commonest one is card accounting:
  the recorded cards have to add up to 54.
- **Show export** prints the whole session as JSON. Copy it somewhere before
  anything drastic. **Restore** (on the setup screen) reads one back. An import
  that is not readable leaves the current game untouched.

The session is saved to the browser's local storage after every accepted entry,
so a refresh, a browser restart, or a phone reboot all pick up where you were.

## Updating safely

```bash
cd ~/shed
git status                  # check for local edits first
git pull
pip install . --upgrade     # only needed if you installed rather than using PYTHONPATH
```

Then **hard-reload the page** in Chrome (pull down to refresh; the server sends
`Cache-Control: no-store`, so a plain reload is normally enough).

Your saved game is not touched by an update. It carries a schema version, and a
version this release cannot read is refused rather than half-understood — the
page then offers the unreadable data for you to copy before you start a new game.
Export a running game before pulling if it matters to you.

To update Termux's own packages, `pkg upgrade`. Do that when you are not
mid-game: it can restart the shell.

## Limitations

**The advice is a greedy baseline, and nothing more.** `GreedyAgent` makes one
pass over the legal moves and scores them on two things: shed as many cards as
possible, then spend the rank it least wants to keep, from a fixed retention
table. It does not count cards, read the pile, look ahead, or model your
opponent. It is the floor a stronger agent should beat, not a solver.
[`docs/agent-baselines.md`](agent-baselines.md) is what it actually scores. No
win probability is shown, because there is nothing here that could compute one.

**Observations are only as good as what you record.** The companion cannot see
the table. If you forget to record a move, the position drifts and it may start
refusing entries that are perfectly legal at the table — that is the correction
workflow's job, and the refusal is the signal.

**Unknown information stays unknown, and that has a price.** Joining mid-game
without a deck count means no advice until you count it. A pile with unseen cards
stays partly unknown until it burns or is picked up. Your opponent's hand is a
count plus whatever a public transfer proved.

**Two players only.** The engine supports two to five; the companion tracks the
two-player game.

**One phone, one game.** There is no sync between devices. The game lives in the
browser you entered it on.

## How it works

- `src/shed/companion/observed.py` — the observed position and the reducer that
  advances it. Rank legality, the constraint transition, the burn rule, batch
  generation, and the active-zone ordering are imported from `shed.engine`
  (`can_play_rank`, `constraint_after`, `burn_reason`, `legal_batches`,
  `active_zone_for_counts`) rather than restated, so the companion and the engine
  cannot drift apart.
- `src/shed/companion/advice.py` — builds a `PlayerView` from observed
  information and calls `GreedyAgent` directly, in this process. The timed
  multiprocessing runner is not involved. No unobserved card becomes a `Card`;
  the identifiers and the single suit the view carries are bookkeeping the type
  demands, not claims about physical cards.
- `src/shed/companion/codec.py` — the JSON vocabulary, and the only place untyped
  external data is validated.
- `src/shed/companion/session.py` — the session document (a versioned initial
  position plus an ordered observation log) and the pure function from it to the
  screen.
- `src/shed/companion/api.py`, `server.py`, `static/` — a stateless local HTTP
  server and the bundled page. Assets are an explicit allowlist, served from the
  package; there is no CDN, no remote font, and no request off the phone.

Nothing outside `shed.engine` and `shed.agents` is imported, and no third-party
package at all.

## What was tested, and on what

Everything described here was developed and verified **on desktop Linux**:

- `uv run pytest` — the full suite, including the companion's own tests for
  unknown information, legal and illegal batches, special-card restrictions and
  burns, pickups and opponent-card tracking, refills against a nearly empty deck,
  pending draws and blind reveals, the zone transitions, undo, corrections,
  replay and persistence round trips, an incomplete ongoing-game state, and that
  a recommendation is legal and changes nothing.
- The server driven over HTTP: bundled assets, status codes, the body limit, an
  occupied port, and a clean shutdown.
- The page driven in headless Chromium at a 390×844 phone viewport, through
  setup → recommendation → *I played this* → recording a drawn rank → the
  opponent's action → undo → reload, plus joining a game, a rejected correction
  and a good one, export and re-import, and the blind-reveal path. The server was
  killed mid-game to confirm the reconnect banner appears, the game and the typed
  input survive, and it recovers by itself when the server returns.

**It has not been run on an Android phone.** Termux, Chrome on Android, the
battery and phantom-process settings, and the on-screen ergonomics are all
documented from the Termux project's own guidance and from testing an equivalent
desktop setup — not from a device. Treat the Android specifics as instructions to
follow rather than as a verified path, and expect to adjust the battery settings
for your manufacturer.
