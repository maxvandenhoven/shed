/* The phone interface.
 *
 * The browser owns the game. It holds the session document -- a versioned initial
 * position plus an ordered log of observations -- saves it to localStorage after
 * every accepted entry, and sends the whole thing to the Python server to be
 * folded. The server keeps nothing, so restarting it is invisible here: the page
 * simply resends what it has been saving all along.
 *
 * Three rules keep the screen honest about a slow or absent server:
 *   - every request carries an increasing id, and a reply older than the newest
 *     one already applied is dropped, so a stale recommendation cannot land;
 *   - exactly one request is in flight at a time, and the buttons are disabled
 *     while it is, so a double tap cannot record a move twice;
 *   - a failed fetch keeps the saved game and everything typed, shows a reconnect
 *     banner, and retries until the server answers again.
 */

"use strict";

const RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "JK"];
const SCHEMA_VERSION = 1;
const SESSION_KEY = "shed.companion.session.v1";
const VIEW_KEY = "shed.companion.view.v1";
const DRAFT_KEY = "shed.companion.draft.v1";
const RETRY_MS = 3000;

const app = {
  session: null,
  view: null,
  requestId: 0,
  lastApplied: 0,
  busy: false,
  offline: false,
  retry: null,
  pickers: {},
  oppCount: null,
  myRank: null,
};

/* ---------------------------------------------------------------- storage */

function store(key, value) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, JSON.stringify(value));
  } catch (err) {
    /* Private mode or a full quota. The game keeps working for this session. */
  }
}

function load(key) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? null : JSON.parse(raw);
  } catch (err) {
    return null;
  }
}

/* A stored session is validated before it is trusted: a document from another
 * schema version, or one mangled by hand, must not silently become the game. */
function readableSession(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    value.schema_version === SCHEMA_VERSION &&
    value.initial !== null &&
    typeof value.initial === "object" &&
    Array.isArray(value.events)
  );
}

function saveDraft() {
  const numbers = {};
  document.querySelectorAll('input[type="number"], input[type="text"], textarea').forEach((el) => {
    if (el.id) numbers[el.id] = el.value;
  });
  const flags = {};
  document.querySelectorAll('input[type="checkbox"], input[type="radio"]').forEach((el) => {
    if (el.id) flags[el.id] = el.checked;
    else if (el.name && el.checked) flags["name:" + el.name] = el.value;
  });
  store(DRAFT_KEY, { pickers: app.pickers, fields: numbers, flags: flags, oppCount: app.oppCount });
}

function restoreDraft() {
  const draft = load(DRAFT_KEY);
  if (!draft) return;
  if (draft.pickers && typeof draft.pickers === "object") {
    Object.keys(draft.pickers).forEach((name) => {
      if (Array.isArray(app.pickers[name]) && Array.isArray(draft.pickers[name])) {
        app.pickers[name] = draft.pickers[name].filter((code) => RANKS.includes(code));
      }
    });
  }
  Object.entries(draft.fields || {}).forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (el) el.value = value;
  });
  Object.entries(draft.flags || {}).forEach(([key, value]) => {
    if (key.startsWith("name:")) {
      const radio = document.querySelector(
        'input[name="' + key.slice(5) + '"][value="' + value + '"]'
      );
      if (radio) radio.checked = true;
    } else {
      const el = document.getElementById(key);
      if (el) el.checked = Boolean(value);
    }
  });
  if (typeof draft.oppCount === "number") app.oppCount = draft.oppCount;
  document.querySelectorAll(".picker").forEach(drawPicker);
}

/* ------------------------------------------------------------------ pickers */

function pickerName(el) {
  return el.dataset.picker;
}

function buildPicker(el) {
  const name = pickerName(el);
  const single = el.classList.contains("single");
  if (!Array.isArray(app.pickers[name])) app.pickers[name] = [];
  el.innerHTML = "";
  RANKS.forEach((code) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.rank = code;
    button.setAttribute("aria-pressed", "false");
    button.textContent = code;
    button.addEventListener("click", () => {
      const limit = Number(el.dataset.limit || "1");
      const chosen = app.pickers[name];
      if (single) {
        app.pickers[name] = chosen.length === 1 && chosen[0] === code ? [] : [code];
      } else if (chosen.length < limit) {
        chosen.push(code);
      }
      drawPicker(el);
      saveDraft();
      onPickerChange(name);
    });
    el.appendChild(button);
  });
  if (!single) {
    const picked = document.createElement("p");
    picked.className = "picked";
    el.appendChild(picked);
    const controls = document.createElement("div");
    controls.className = "row";
    const back = document.createElement("button");
    back.type = "button";
    back.textContent = "Back";
    back.addEventListener("click", () => {
      app.pickers[name].pop();
      drawPicker(el);
      saveDraft();
      onPickerChange(name);
    });
    const clear = document.createElement("button");
    clear.type = "button";
    clear.textContent = "Clear";
    clear.addEventListener("click", () => {
      app.pickers[name] = [];
      drawPicker(el);
      saveDraft();
      onPickerChange(name);
    });
    controls.appendChild(back);
    controls.appendChild(clear);
    el.appendChild(controls);
  }
  drawPicker(el);
}

function drawPicker(el) {
  const name = pickerName(el);
  const chosen = app.pickers[name] || [];
  const playable = app.view && app.view.state ? app.view.state.playable_ranks : null;
  el.querySelectorAll("button[data-rank]").forEach((button) => {
    const code = button.dataset.rank;
    button.setAttribute("aria-pressed", chosen.includes(code) ? "true" : "false");
    const marks = playable && name.startsWith("opp") ? !playable.includes(code) : false;
    button.classList.toggle("unplayable", marks);
  });
  const picked = el.querySelector(".picked");
  if (picked) {
    picked.textContent = "";
    if (chosen.length) picked.textContent = "Entered: " + chosen.join(" ");
    else {
      const empty = document.createElement("span");
      empty.className = "empty";
      empty.textContent = "nothing entered yet";
      picked.appendChild(empty);
    }
  }
}

function picked(name) {
  return (app.pickers[name] || []).slice();
}

function clearPicker(name) {
  app.pickers[name] = [];
  const el = document.querySelector('[data-picker="' + name + '"]');
  if (el) drawPicker(el);
  saveDraft();
}

/* A tap on a rank changes which buttons make sense, and no server round trip is
 * involved, so readiness is recomputed locally rather than waiting for a reply. */
function onPickerChange(name) {
  if (name === "opp-rank") app.oppCount = null;
  refreshReadiness();
}

function setEnabled(button, enabled) {
  if (!button) return;
  button.dataset.forceDisabled = enabled ? "false" : "true";
  button.disabled = !enabled;
}

function refreshReadiness() {
  if (!app.view) return;
  const state = app.view.state;
  const entry = state.pending.length ? state.pending[0] : null;
  const submit = document.getElementById("pending-submit");
  if (entry) {
    const entered = picked("pending").length;
    setEnabled(submit, entered === entry.count);
    submit.textContent = "Record " + entered + " of " + entry.count + " card(s)";
  } else {
    setEnabled(submit, false);
  }
  setEnabled(
    document.getElementById("my-reveal-submit"),
    state.my_turn && picked("my-reveal").length === 1
  );
  const theirTurn = !state.finished && state.to_act === "opponent";
  setEnabled(
    document.getElementById("opp-reveal-submit"),
    theirTurn && picked("opp-reveal").length === 1
  );
  drawOpponentCounts();
}

/* --------------------------------------------------------------- networking */

function setBusy(busy) {
  app.busy = busy;
  document.querySelectorAll("main button").forEach((button) => {
    if (button.dataset.keepEnabled === "true") return;
    button.disabled = busy || button.dataset.forceDisabled === "true";
  });
}

function setOffline(offline, detail) {
  app.offline = offline;
  const banner = document.getElementById("offline");
  banner.hidden = !offline;
  if (offline) {
    document.getElementById("offline-detail").textContent = detail || "Retrying…";
    if (app.retry === null) app.retry = window.setInterval(refresh, RETRY_MS);
  } else if (app.retry !== null) {
    window.clearInterval(app.retry);
    app.retry = null;
  }
}

function showError(id, message) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = message || "";
  el.hidden = !message;
}

async function post(path, body) {
  const id = ++app.requestId;
  const payload = Object.assign({ request_id: id }, body);
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let data = null;
  try {
    data = await response.json();
  } catch (err) {
    throw new Error("the companion sent a reply this page could not read");
  }
  return { id: id, ok: response.ok, data: data };
}

/* One request at a time, newest reply wins, nothing typed is lost on failure. */
async function send(path, body, errorTarget) {
  if (app.busy) return false;
  showError(errorTarget, "");
  setBusy(true);
  try {
    const reply = await post(path, body);
    if (reply.id < app.lastApplied) return false;
    if (reply.ok) {
      applyReply(reply.id, reply.data);
      setOffline(false);
      return true;
    }
    setOffline(false);
    showError(errorTarget, reply.data && reply.data.error ? reply.data.error : "rejected");
    return false;
  } catch (err) {
    setOffline(true, String(err.message || err));
    return false;
  } finally {
    setBusy(false);
  }
}

function applyReply(id, data) {
  app.lastApplied = id;
  app.session = data.session;
  app.view = data;
  store(SESSION_KEY, app.session);
  store(VIEW_KEY, app.view);
  render();
}

async function refresh() {
  if (!app.session || app.busy) return;
  await send("/api/state", { session: app.session }, "action-error");
}

/* ---------------------------------------------------------------- rendering */

function show(which) {
  document.getElementById("setup").hidden = which !== "setup";
  document.getElementById("game").hidden = which !== "game";
}

function text(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function groupText(groups) {
  if (!groups || !groups.length) return "none";
  return groups.map((g) => (g.count > 1 ? g.count + "x" + g.rank : g.rank)).join("  ");
}

function render() {
  if (!app.view) {
    show("setup");
    return;
  }
  show("game");
  const view = app.view;
  const state = view.state;

  const turn = document.getElementById("turn-chip");
  if (state.finished) {
    turn.textContent = (state.winner === "me" ? "You" : "Opponent") + " went out";
    turn.classList.toggle("mine", state.winner === "me");
  } else {
    turn.textContent = state.my_turn ? "Your turn" : "Opponent's turn";
    turn.classList.toggle("mine", Boolean(state.my_turn));
  }
  text("restriction-chip", state.constraint.text);
  const unseen = state.pile.unknown ? " (" + state.pile.unknown + " unseen)" : "";
  text("pile-chip", "pile " + state.pile.size + unseen);
  text("deck-chip", "deck " + (state.deck_count === null ? "?" : state.deck_count));
  text("rev", "state " + view.revision);
  if (app.shownRevision !== view.revision) {
    /* A new position means the rank half-chosen against the old one is stale. */
    app.myRank = null;
    app.shownRevision = view.revision;
  }

  showError("replay-error", view.replay_error || "");
  renderPending(state);
  renderRecommendation(view);
  renderMine(view);
  renderOpponent(state);
  renderTable(state);
  renderHistory(view);
  setEnabled(document.getElementById("undo"), view.can_undo);
  document.querySelectorAll(".picker").forEach(drawPicker);
  refreshReadiness();
}

function renderPending(state) {
  const section = document.getElementById("pending");
  const entry = state.pending.length ? state.pending[0] : null;
  section.hidden = entry === null;
  if (!entry) return;
  text("pending-prompt", entry.prompt);
  const el = document.querySelector('[data-picker="pending"]');
  el.dataset.limit = String(entry.count);
}

function renderRecommendation(view) {
  const body = document.getElementById("rec-body");
  const blocked = document.getElementById("rec-blocked");
  if (view.recommendation) {
    body.hidden = false;
    blocked.hidden = true;
    const rec = view.recommendation;
    text("rec-headline", rec.headline);
    text("rec-reasoning", rec.reasoning + " It compared " + rec.considered + " legal option(s).");
    text("rec-effect", rec.effect);
    text("rec-caveat", rec.caveat);
    const notes = document.getElementById("rec-notes");
    notes.innerHTML = "";
    rec.notes.forEach((note) => {
      const li = document.createElement("li");
      li.textContent = note;
      notes.appendChild(li);
    });
    const played = document.getElementById("rec-played");
    played.textContent =
      rec.move.kind === "reveal" ? "I turned a card over" : "I played this";
  } else {
    body.hidden = true;
    blocked.hidden = false;
    const list = document.getElementById("rec-blockers");
    list.innerHTML = "";
    view.blockers.forEach((blocker) => {
      const li = document.createElement("li");
      li.textContent = blocker.message;
      list.appendChild(li);
    });
  }
}

function renderMine(view) {
  const state = view.state;
  const mine = state.seats.find((seat) => seat.player === "me");
  text(
    "mine-hand",
    groupText(mine.hand_known) +
      (mine.hand_unknown ? "  +" + mine.hand_unknown + " unrecorded" : "")
  );
  text("mine-face-up", groupText(mine.face_up));
  text("mine-face-down", mine.face_down + " card(s), unknown");

  const zoneNote = {
    hand: "You are playing from your hand.",
    face_up: "Your hand and the deck are gone, so you play from your face-up cards.",
    face_down: "Only your face-down cards are left.",
  };
  let note = "Not your turn. Record your opponent's action below.";
  if (state.finished) note = "The game is over.";
  else if (state.my_turn) {
    note =
      zoneNote[mine.active_zone] ||
      (mine.remaining
        ? "Which zone you play from cannot be worked out yet -- see the notes above."
        : "You have no cards left.");
  }
  text("mine-zone", note);

  drawMyOptions(view, mine);

  const revealPanel = document.getElementById("mine-reveal");
  revealPanel.hidden = !view.options.some((option) => option.kind === "reveal");
}

/* Rank first, then how many: the same two taps as recording the opponent's play,
 * and the quantity is on the rank button so the whole zone reads at a glance. */
function drawMyOptions(view, mine) {
  const holder = document.getElementById("mine-options");
  holder.innerHTML = "";
  const plays = view.options.filter((option) => option.kind === "play");
  const pickup = view.options.find((option) => option.kind === "pickup");
  const suggested = view.recommendation ? view.recommendation.move : null;
  const groups = mine.active_zone === "face_up" ? mine.face_up : mine.hand_known;

  if (view.options.length && groups.length) {
    const ranks = document.createElement("div");
    ranks.className = "picker";
    groups.forEach((group) => {
      const legal = plays.filter((play) => play.rank === group.rank);
      const button = document.createElement("button");
      button.type = "button";
      button.appendChild(document.createTextNode(group.rank));
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = "x" + group.count;
      button.appendChild(badge);
      button.setAttribute("aria-pressed", app.myRank === group.rank ? "true" : "false");
      button.classList.toggle("unplayable", legal.length === 0);
      button.disabled = legal.length === 0;
      if (suggested && suggested.rank === group.rank) button.classList.add("recommended");
      button.addEventListener("click", () => {
        app.myRank = app.myRank === group.rank ? null : group.rank;
        drawMyOptions(app.view, mine);
      });
      ranks.appendChild(button);
    });
    holder.appendChild(ranks);
  }

  const chosen = plays.filter((play) => play.rank === app.myRank);
  if (chosen.length) {
    const counts = document.createElement("div");
    counts.className = "options";
    chosen.forEach((play) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Play " + play.count + "x " + play.rank;
      if (suggested && suggested.rank === play.rank && suggested.count === play.count) {
        button.classList.add("recommended");
      }
      button.addEventListener("click", () => submitMove(play, "action-error"));
      counts.appendChild(button);
    });
    holder.appendChild(counts);
  }

  if (pickup) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "danger big";
    button.textContent = "Nothing playable -- I picked up the pile";
    button.addEventListener("click", () => submitMove(pickup, "action-error"));
    holder.appendChild(button);
  }
}

function renderOpponent(state) {
  const theirs = state.seats.find((seat) => seat.player === "opponent");
  const theirTurn = !state.finished && state.to_act === "opponent";
  const revealing = theirs.active_zone === "face_down";
  document.getElementById("opp-play").hidden = revealing;
  document.getElementById("opp-reveal").hidden = !revealing;
  setEnabled(document.getElementById("opp-pickup"), theirTurn && !revealing);
}

function drawOpponentCounts() {
  const holder = document.getElementById("opp-counts");
  if (!holder) return;
  holder.innerHTML = "";
  const rank = picked("opp-rank")[0];
  const state = app.view ? app.view.state : null;
  const confirm = document.getElementById("opp-confirm");
  if (!rank || !state) {
    setEnabled(confirm, false);
    if (confirm) confirm.textContent = "Pick a rank they played";
    return;
  }
  const most = Math.max(1, Math.min(rank === "JK" ? 2 : 4, state.opponent_max_count));
  for (let count = 1; count <= most; count += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = count + "x " + rank;
    button.setAttribute("aria-pressed", app.oppCount === count ? "true" : "false");
    if (app.oppCount === count) button.classList.add("recommended");
    button.addEventListener("click", () => {
      app.oppCount = count;
      saveDraft();
      drawOpponentCounts();
    });
    holder.appendChild(button);
  }
  const theirTurn = state.to_act === "opponent" && !state.finished;
  setEnabled(confirm, theirTurn && app.oppCount !== null);
  confirm.textContent = app.oppCount
    ? "Confirm: they played " + app.oppCount + "x " + rank
    : "Pick how many";
}

function renderTable(state) {
  const rows = document.getElementById("status-rows");
  rows.innerHTML = "";
  const add = (label, value) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = label;
    const td = document.createElement("td");
    td.textContent = value;
    tr.appendChild(th);
    tr.appendChild(td);
    rows.appendChild(tr);
  };
  add("To act", state.finished ? "game over" : state.to_act_name);
  add("Restriction", state.constraint.text);
  add("Pile", state.pile.size + " card(s)" + (state.pile.top ? ", top " + state.pile.top : ""));
  add("Deck", state.deck_count === null ? "not counted" : state.deck_count + " card(s)");
  add("Burned", state.burned_count + " card(s)");
  state.seats.forEach((seat) => {
    add(
      seat.player === "me" ? "You hold" : "Opponent holds",
      seat.remaining +
        " (" +
        seat.hand_count +
        " hand, " +
        seat.face_up_count +
        " face up, " +
        seat.face_down +
        " face down)"
    );
  });

  const notes = document.getElementById("uncertainty");
  notes.innerHTML = "";
  const uncertain = [];
  if (state.deck_count === null) uncertain.push("The deck has not been counted.");
  if (state.pile.unknown) {
    uncertain.push(state.pile.unknown + " card(s) in the pile were never seen.");
  }
  const theirs = state.seats.find((seat) => seat.player === "opponent");
  if (theirs.hand_unknown) {
    uncertain.push(theirs.hand_unknown + " of their hand cards are unknown.");
  }
  if (theirs.hand_known.length) {
    uncertain.push("Known in their hand: " + groupText(theirs.hand_known) + ".");
  }
  uncertain.push("Face-down cards are unknown to everybody until turned over.");
  uncertain.forEach((note) => {
    const li = document.createElement("li");
    li.textContent = note;
    notes.appendChild(li);
  });
}

function renderHistory(view) {
  const list = document.getElementById("history");
  list.innerHTML = "";
  view.history
    .slice()
    .reverse()
    .forEach((entry) => {
      const li = document.createElement("li");
      li.value = entry.index;
      li.textContent = entry.text;
      list.appendChild(li);
    });
  if (!view.history.length) {
    const li = document.createElement("li");
    li.className = "hint";
    li.textContent = "Nothing recorded yet.";
    list.appendChild(li);
  }
}

/* ------------------------------------------------------------------ actions */

function submitMove(move, errorTarget) {
  if (move.kind === "reveal") {
    /* A reveal cannot be recorded until the card has been turned over, so the
     * button opens the rank entry instead of sending anything. */
    document.getElementById("mine-reveal").hidden = false;
    showError(errorTarget, "");
    document.getElementById("mine-reveal").scrollIntoView({ block: "center" });
    return;
  }
  const event =
    move.kind === "pickup"
      ? { kind: "pickup", player: "me" }
      : { kind: "play", player: "me", rank: move.rank, count: move.count };
  send("/api/event", { session: app.session, event: event }, errorTarget);
}

function numberField(id) {
  const el = document.getElementById(id);
  if (!el || el.value.trim() === "") return null;
  const value = Number(el.value);
  return Number.isInteger(value) && value >= 0 ? value : null;
}

function buildPatch() {
  const patch = {};
  if (document.getElementById("fix-my-hand-empty").checked) patch.my_hand = [];
  else if (picked("fix-my-hand").length) patch.my_hand = picked("fix-my-hand");
  if (document.getElementById("fix-my-face-up-empty").checked) patch.my_face_up = [];
  else if (picked("fix-my-face-up").length) patch.my_face_up = picked("fix-my-face-up");
  if (document.getElementById("fix-opp-face-up-empty").checked) patch.opponent_face_up = [];
  else if (picked("fix-opp-face-up").length) {
    patch.opponent_face_up = picked("fix-opp-face-up");
  }
  const pairs = [
    ["fix-my-face-down", "my_face_down"],
    ["fix-opp-face-down", "opponent_face_down"],
    ["fix-deck-count", "deck_count"],
    ["fix-burned-count", "burned_count"],
  ];
  pairs.forEach(([id, field]) => {
    const value = numberField(id);
    if (value !== null) patch[field] = value;
  });
  const theirHand = numberField("fix-opp-hand-count");
  if (theirHand !== null) {
    const known = app.view
      ? app.view.state.seats.find((seat) => seat.player === "opponent").hand_known
      : [];
    const knownTotal = known.reduce((sum, group) => sum + group.count, 0);
    patch.opponent_hand_unknown = Math.max(0, theirHand - knownTotal);
  }
  const toAct = document.querySelector('input[name="fix-to-act"]:checked');
  if (toAct && toAct.value) patch.to_act = toAct.value;
  return patch;
}

function clearCorrectionForm() {
  ["fix-my-hand", "fix-my-face-up", "fix-opp-face-up"].forEach(clearPicker);
  [
    "fix-my-face-down",
    "fix-opp-face-down",
    "fix-opp-hand-count",
    "fix-deck-count",
    "fix-burned-count",
    "fix-note",
  ].forEach((id) => {
    document.getElementById(id).value = "";
  });
  ["fix-my-hand-empty", "fix-my-face-up-empty", "fix-opp-face-up-empty"].forEach((id) => {
    document.getElementById(id).checked = false;
  });
  const none = document.querySelector('input[name="fix-to-act"][value=""]');
  if (none) none.checked = true;
  saveDraft();
}

function constraintFromForm() {
  const kind = document.querySelector('input[name="join-constraint"]:checked').value;
  if (kind === "unrestricted") return { kind: "unrestricted" };
  if (kind === "at_most") return { kind: "at_most", rank: "7" };
  const rank = picked("join-constraint-rank")[0];
  return { kind: "at_least", rank: rank || "2" };
}

function wire() {
  document.querySelectorAll(".picker").forEach(buildPicker);

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((other) => {
        const active = other === tab;
        other.setAttribute("aria-selected", active ? "true" : "false");
        document.getElementById(other.dataset.panel).hidden = !active;
      });
    });
  });

  document.getElementById("start-new").addEventListener("click", () => {
    send(
      "/api/new",
      {
        my_hand: picked("new-my-hand"),
        my_face_up: picked("new-my-face-up"),
        opponent_face_up: picked("new-opp-face-up"),
        starting_player: document.querySelector('input[name="new-starter"]:checked').value,
      },
      "new-error"
    );
  });

  document.getElementById("start-join").addEventListener("click", () => {
    const unknownDeck = document.getElementById("join-deck-unknown").checked;
    const unseenPile = Number(document.getElementById("join-pile-unknown").value || "0");
    const pile = picked("join-pile").slice();
    for (let index = 0; index < unseenPile; index += 1) pile.unshift(null);
    send(
      "/api/join",
      {
        my_hand: picked("join-my-hand"),
        my_face_up: picked("join-my-face-up"),
        my_face_down: Number(document.getElementById("join-my-face-down").value || "0"),
        opponent_hand_count: Number(document.getElementById("join-opp-hand-count").value || "0"),
        opponent_hand_known: picked("join-opp-hand-known"),
        opponent_face_up: picked("join-opp-face-up"),
        opponent_face_down: Number(document.getElementById("join-opp-face-down").value || "0"),
        deck_count: unknownDeck ? null : Number(document.getElementById("join-deck-count").value || "0"),
        pile: pile,
        constraint: constraintFromForm(),
        to_act: document.querySelector('input[name="join-to-act"]:checked').value,
      },
      "join-error"
    );
  });

  document.getElementById("do-import").addEventListener("click", async () => {
    const raw = document.getElementById("import-text").value;
    let parsed = null;
    try {
      parsed = JSON.parse(raw);
    } catch (err) {
      showError("import-error", "That is not valid JSON, so nothing was changed.");
      return;
    }
    if (!readableSession(parsed)) {
      showError("import-error", "That is not a session this version reads; nothing changed.");
      return;
    }
    /* The server has the last word: an import only replaces the current game once
     * its whole log has folded cleanly here. */
    const kept = app.session;
    const ok = await send("/api/state", { session: parsed }, "import-error");
    if (!ok) app.session = kept;
  });

  document.getElementById("rec-played").addEventListener("click", () => {
    if (app.view && app.view.recommendation) {
      submitMove(app.view.recommendation.move, "action-error");
    }
  });

  document.getElementById("pending-submit").addEventListener("click", async () => {
    const ranks = picked("pending");
    const ok = await send(
      "/api/event",
      { session: app.session, event: { kind: "record", ranks: ranks } },
      "action-error"
    );
    if (ok) clearPicker("pending");
  });

  document.getElementById("my-reveal-submit").addEventListener("click", async () => {
    const rank = picked("my-reveal")[0];
    if (!rank) return;
    const ok = await send(
      "/api/event",
      { session: app.session, event: { kind: "reveal", player: "me", rank: rank } },
      "action-error"
    );
    if (ok) clearPicker("my-reveal");
  });

  document.getElementById("opp-confirm").addEventListener("click", async () => {
    const rank = picked("opp-rank")[0];
    if (!rank || app.oppCount === null) return;
    const ok = await send(
      "/api/event",
      {
        session: app.session,
        event: { kind: "play", player: "opponent", rank: rank, count: app.oppCount },
      },
      "action-error"
    );
    if (ok) {
      clearPicker("opp-rank");
      app.oppCount = null;
      drawOpponentCounts();
    }
  });

  document.getElementById("opp-reveal-submit").addEventListener("click", async () => {
    const rank = picked("opp-reveal")[0];
    if (!rank) return;
    const ok = await send(
      "/api/event",
      { session: app.session, event: { kind: "reveal", player: "opponent", rank: rank } },
      "action-error"
    );
    if (ok) clearPicker("opp-reveal");
  });

  document.getElementById("opp-pickup").addEventListener("click", () => {
    send(
      "/api/event",
      { session: app.session, event: { kind: "pickup", player: "opponent" } },
      "action-error"
    );
  });

  document.getElementById("undo").addEventListener("click", () => {
    send("/api/undo", { session: app.session }, "action-error");
  });

  document.getElementById("toggle-correct").addEventListener("click", () => {
    const panel = document.getElementById("correct");
    panel.hidden = !panel.hidden;
  });

  document.getElementById("fix-submit").addEventListener("click", async () => {
    const patch = buildPatch();
    if (!Object.keys(patch).length) {
      showError("action-error", "Fill in at least one field before recording a correction.");
      return;
    }
    const note = document.getElementById("fix-note").value;
    const ok = await send(
      "/api/event",
      { session: app.session, event: { kind: "correct", patch: patch, note: note } },
      "action-error"
    );
    if (ok) {
      clearCorrectionForm();
      document.getElementById("correct").hidden = true;
    }
  });

  document.getElementById("export").addEventListener("click", () => {
    const area = document.getElementById("export-text");
    area.hidden = !area.hidden;
    if (!area.hidden) {
      area.value = JSON.stringify(app.session, null, 2);
      area.select();
    }
  });

  document.getElementById("restart").addEventListener("click", () => {
    const message =
      "Start a different game? Copy the export first if you want this one back.";
    if (!window.confirm(message)) return;
    app.session = null;
    app.view = null;
    store(SESSION_KEY, null);
    store(VIEW_KEY, null);
    show("setup");
  });

  document.querySelectorAll("input, textarea").forEach((el) => {
    el.addEventListener("change", saveDraft);
    el.addEventListener("input", saveDraft);
  });

  document.getElementById("offline").dataset.keepEnabled = "true";
}

function start() {
  wire();
  restoreDraft();
  const saved = load(SESSION_KEY);
  const savedView = load(VIEW_KEY);
  if (readableSession(saved)) {
    app.session = saved;
    if (savedView && savedView.state) {
      /* Draw the last screen we had before talking to the server, so a phone that
       * wakes up before Termux does still shows the game rather than the setup form. */
      app.view = savedView;
      render();
    }
    refresh();
  } else {
    if (saved !== null) {
      document.getElementById("import-text").value = JSON.stringify(saved, null, 2);
      showError(
        "import-error",
        "A saved game was found that this version cannot read. It is shown here so " +
          "you can keep a copy; starting a new game will replace it."
      );
    }
    show("setup");
  }
  window.addEventListener("pageshow", () => {
    if (app.session) refresh();
  });
}

document.addEventListener("DOMContentLoaded", start);
