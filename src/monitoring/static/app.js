"use strict";

const POLL_INTERVAL_MS = 2000;
const MAX_VISIBLE_RECORDS = 1000;

const state = {
  runs: [],
  selectedRunId: null,
  summary: null,
  records: [],
  paused: false,
  polling: false,
  selectedMetric: null,
  replayEntries: [],
  selectedReplayId: null,
  replay: null,
  replayMove: 0,
  replayTimer: null,
};

const elements = {
  connection: document.querySelector("#connection-state"),
  refreshToggle: document.querySelector("#refresh-toggle"),
  runCount: document.querySelector("#run-count"),
  runList: document.querySelector("#run-list"),
  emptyState: document.querySelector("#empty-state"),
  runView: document.querySelector("#run-view"),
  runStatus: document.querySelector("#run-status"),
  runUpdated: document.querySelector("#run-updated"),
  runName: document.querySelector("#run-name"),
  runContext: document.querySelector("#run-context"),
  latestStep: document.querySelector("#latest-step"),
  signalLoss: document.querySelector("#signal-loss"),
  signalPolicyLoss: document.querySelector("#signal-policy-loss"),
  signalThroughput: document.querySelector("#signal-throughput"),
  signalReplay: document.querySelector("#signal-replay"),
  signalReplayDetail: document.querySelector("#signal-replay-detail"),
  signalSearchCard: document.querySelector("#signal-search-card"),
  signalSearch: document.querySelector("#signal-search"),
  signalSearchDetail: document.querySelector("#signal-search-detail"),
  signalEvaluation: document.querySelector("#signal-evaluation"),
  signalEvaluationDetail: document.querySelector("#signal-evaluation-detail"),
  metricSelect: document.querySelector("#metric-select"),
  metricChart: document.querySelector("#metric-chart"),
  chartDescription: document.querySelector("#chart-description"),
  chartEmpty: document.querySelector("#chart-empty"),
  chartSummary: document.querySelector("#chart-summary"),
  eventCount: document.querySelector("#event-count"),
  eventTable: document.querySelector("#event-table"),
  metricFilter: document.querySelector("#metric-filter"),
  metricInventory: document.querySelector("#metric-inventory"),
  replaySelect: document.querySelector("#replay-select"),
  replayEmpty: document.querySelector("#replay-empty"),
  replayView: document.querySelector("#replay-view"),
  board: document.querySelector("#pente-board"),
  replayResult: document.querySelector("#replay-result"),
  replayMeta: document.querySelector("#replay-meta"),
  captureOne: document.querySelector("#capture-one"),
  captureTwo: document.querySelector("#capture-two"),
  moveLabel: document.querySelector("#move-label"),
  moveRange: document.querySelector("#move-range"),
  replayPrevious: document.querySelector("#replay-previous"),
  replayPlay: document.querySelector("#replay-play"),
  replayNext: document.querySelector("#replay-next"),
  replaySpeed: document.querySelector("#replay-speed"),
  toast: document.querySelector("#toast"),
};

function assertElements() {
  for (const [name, element] of Object.entries(elements)) {
    if (!element) {
      throw new Error(`Dashboard element is missing: ${name}`);
    }
  }
}

async function fetchJson(path) {
  const response = await fetch(path, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

async function refresh() {
  if (state.paused || state.polling) {
    return;
  }
  state.polling = true;
  try {
    const payload = await fetchJson("/api/runs");
    state.runs = payload.runs;
    if (!state.runs.some((run) => run.id === state.selectedRunId)) {
      state.selectedRunId = state.runs[0]?.id ?? null;
      state.selectedReplayId = null;
      state.replay = null;
    }
    renderRunList();
    setConnection("online", "Live");
    if (state.selectedRunId) {
      await refreshSelectedRun(state.selectedRunId);
    } else {
      showEmptyState();
    }
  } catch (error) {
    setConnection("offline", "Offline");
    showToast(error.message);
  } finally {
    state.polling = false;
  }
}

async function refreshSelectedRun(runId) {
  const encodedRunId = encodeURIComponent(runId);
  const summary = await fetchJson(`/api/runs/${encodedRunId}/summary`);
  if (state.selectedRunId !== runId) {
    return;
  }

  const after = Math.max(0, summary.record_count - MAX_VISIBLE_RECORDS);
  const [recordsPayload, replayPayload] = await Promise.all([
    fetchJson(`/api/runs/${encodedRunId}/records?after=${after}&limit=${MAX_VISIBLE_RECORDS}`),
    fetchJson(`/api/replays?run_id=${encodeURIComponent(summary.run_key)}`),
  ]);
  if (state.selectedRunId !== runId) {
    return;
  }

  state.summary = summary;
  state.records = recordsPayload.records;
  state.replayEntries = replayPayload.replays;
  renderRun();
  await renderReplayList();
}

function renderRunList() {
  elements.runCount.textContent = String(state.runs.length);
  const fragment = document.createDocumentFragment();
  for (const run of state.runs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "run-item";
    button.setAttribute("role", "listitem");
    if (run.id === state.selectedRunId) {
      button.classList.add("selected");
      button.setAttribute("aria-current", "true");
    }

    const status = document.createElement("span");
    status.className = `run-item-status ${run.status}`;
    status.setAttribute("aria-label", run.status);

    const content = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = run.name;
    const detail = document.createElement("small");
    detail.textContent = `${formatInteger(run.record_count)} records, ${relativeTime(run.modified_at_unix)}`;
    content.append(title, detail);
    button.append(status, content);
    button.addEventListener("click", () => selectRun(run.id));
    fragment.append(button);
  }
  elements.runList.replaceChildren(fragment);
}

async function selectRun(runId) {
  if (state.selectedRunId === runId) {
    return;
  }
  stopReplay();
  state.selectedRunId = runId;
  state.selectedReplayId = null;
  state.replay = null;
  state.records = [];
  state.summary = null;
  renderRunList();
  try {
    await refreshSelectedRun(runId);
  } catch (error) {
    showToast(error.message);
  }
}

function showEmptyState() {
  elements.emptyState.hidden = false;
  elements.runView.hidden = true;
}

function renderRun() {
  const summary = state.summary;
  if (!summary) {
    return;
  }
  elements.emptyState.hidden = true;
  elements.runView.hidden = false;
  elements.runStatus.textContent = summary.status;
  elements.runStatus.className = `run-status ${summary.status}`;
  elements.runUpdated.textContent = `Updated ${relativeTime(summary.modified_at_unix)}`;
  elements.runName.textContent = summary.name;
  elements.latestStep.textContent = formatInteger(summary.last_step ?? 0);
  elements.runContext.textContent = `${formatInteger(summary.record_count)} records, ${formatInteger(Object.keys(summary.event_counts).length)} event types${summary.issues.length ? `, ${summary.issues.length} file issue(s)` : ""}.`;

  renderSignals(summary.latest_metrics);
  renderMetricSelector(summary.numeric_metrics);
  renderChart();
  renderEvents();
  renderMetricInventory();
}

function renderSignals(metrics) {
  elements.signalLoss.textContent = formatMetric(metrics.loss);
  elements.signalPolicyLoss.textContent = `Policy loss: ${formatMetric(metrics.policy_loss)}`;
  elements.signalThroughput.textContent = formatMetric(metrics.positions_per_second);
  elements.signalReplay.textContent = formatIntegerOrDash(metrics.replay_positions);
  elements.signalReplayDetail.textContent = `${formatIntegerOrDash(metrics.replay_unique_positions)} unique`;

  const collapseRate = numberOrNull(metrics.search_collapse_rate);
  const collapsed = Number(metrics.search_collapse_detected) > 0;
  elements.signalSearch.textContent = collapseRate === null ? "--" : formatPercent(collapseRate);
  elements.signalSearchDetail.textContent = `${formatIntegerOrDash(metrics.search_collapse_eligible_roots)} eligible roots`;
  elements.signalSearchCard.classList.toggle("alert", collapsed);

  const evaluations = [
    [metrics.current_vs_random_decisive_win_rate, "vs random"],
    [metrics.tactical_accuracy, "tactical accuracy"],
    [metrics.current_decisive_win_rate, "vs previous model"],
  ];
  const evaluation = evaluations.find(([value]) => numberOrNull(value) !== null);
  elements.signalEvaluation.textContent = evaluation ? formatPercent(Number(evaluation[0])) : "--";
  elements.signalEvaluationDetail.textContent = evaluation ? evaluation[1] : "No result";
}

function renderMetricSelector(metricNames) {
  const previous = state.selectedMetric;
  const preferred = [
    "loss",
    "positions_per_second",
    "leaf_evaluations_per_second",
    "policy_loss",
    "value_loss",
  ];
  if (!previous || !metricNames.includes(previous)) {
    state.selectedMetric = preferred.find((name) => metricNames.includes(name)) ?? metricNames[0] ?? null;
  }

  const fragment = document.createDocumentFragment();
  for (const name of metricNames) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = humanize(name);
    option.selected = name === state.selectedMetric;
    fragment.append(option);
  }
  elements.metricSelect.replaceChildren(fragment);
  elements.metricSelect.disabled = metricNames.length === 0;
}

function renderChart() {
  const metric = state.selectedMetric;
  const samples = metric
    ? state.records
        .map((record, index) => ({
          value: numberOrNull(record.metrics[metric]),
          step: record.step,
          index,
        }))
        .filter((sample) => sample.value !== null)
    : [];
  elements.chartEmpty.hidden = samples.length > 0;
  elements.metricChart.hidden = samples.length === 0;
  renderChartSummary(metric);

  if (!samples.length) {
    elements.chartDescription.textContent = "No numeric values for this metric.";
    return;
  }

  const canvas = elements.metricChart;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, rect.width);
  const height = Math.max(220, rect.height);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const styles = getComputedStyle(document.documentElement);
  const lineColor = styles.getPropertyValue("--pico-muted-border-color").trim();
  const mutedColor = styles.getPropertyValue("--pico-muted-color").trim();
  const primaryColor = styles.getPropertyValue("--pico-primary").trim();
  const padding = { top: 22, right: 18, bottom: 34, left: 62 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = samples.map((sample) => sample.value);
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) {
    const margin = Math.abs(minimum) * 0.1 || 1;
    minimum -= margin;
    maximum += margin;
  }
  const range = maximum - minimum;
  const xFor = (index) => padding.left + (samples.length === 1 ? plotWidth / 2 : (index / (samples.length - 1)) * plotWidth);
  const yFor = (value) => padding.top + ((maximum - value) / range) * plotHeight;

  context.lineWidth = 1;
  context.strokeStyle = lineColor;
  context.fillStyle = mutedColor;
  context.font = "11px ui-monospace, monospace";
  context.textAlign = "right";
  context.textBaseline = "middle";
  for (let index = 0; index <= 4; index += 1) {
    const y = padding.top + (index / 4) * plotHeight;
    const value = maximum - (index / 4) * range;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    context.fillText(compactNumber(value), padding.left - 9, y);
  }

  context.beginPath();
  samples.forEach((sample, index) => {
    const x = xFor(index);
    const y = yFor(sample.value);
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.lineTo(xFor(samples.length - 1), height - padding.bottom);
  context.lineTo(xFor(0), height - padding.bottom);
  context.closePath();
  context.globalAlpha = 0.12;
  context.fillStyle = primaryColor;
  context.fill();
  context.globalAlpha = 1;

  context.beginPath();
  samples.forEach((sample, index) => {
    const x = xFor(index);
    const y = yFor(sample.value);
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.strokeStyle = primaryColor;
  context.lineWidth = 2;
  context.stroke();

  const latest = samples[samples.length - 1];
  context.beginPath();
  context.arc(xFor(samples.length - 1), yFor(latest.value), 4, 0, Math.PI * 2);
  context.fillStyle = primaryColor;
  context.fill();

  context.fillStyle = mutedColor;
  context.textAlign = "left";
  context.textBaseline = "bottom";
  context.fillText(`step ${samples[0].step}`, padding.left, height - 8);
  context.textAlign = "right";
  context.fillText(`step ${latest.step}`, width - padding.right, height - 8);
  elements.chartDescription.textContent = `${humanize(metric)}: ${formatMetric(Math.min(...values))} to ${formatMetric(Math.max(...values))}, ${samples.length} values.`;
}

function renderChartSummary(metric) {
  const statistics = metric ? state.summary?.statistics[metric] : null;
  const values = statistics
    ? [
        ["Latest", statistics.latest],
        ["Mean", statistics.mean],
        ["Min", statistics.minimum],
        ["Max", statistics.maximum],
      ]
    : [];
  const fragment = document.createDocumentFragment();
  for (const [label, value] of values) {
    const item = document.createElement("div");
    item.className = "chart-stat";
    const labelElement = document.createElement("span");
    labelElement.textContent = label;
    const valueElement = document.createElement("strong");
    valueElement.textContent = formatMetric(value);
    item.append(labelElement, valueElement);
    fragment.append(item);
  }
  elements.chartSummary.replaceChildren(fragment);
}

function renderEvents() {
  const recent = state.records.slice(-50).reverse();
  elements.eventCount.textContent = formatInteger(state.summary?.record_count ?? 0);
  const fragment = document.createDocumentFragment();
  for (const record of recent) {
    const row = document.createElement("tr");
    for (const value of [humanize(record.event), formatInteger(record.step), formatClock(record.timestamp_unix)]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    fragment.append(row);
  }
  elements.eventTable.replaceChildren(fragment);
}

function renderMetricInventory() {
  const filter = elements.metricFilter.value.trim().toLowerCase();
  const metrics = Object.entries(state.summary?.latest_metrics ?? {})
    .filter(([name]) => !filter || name.toLowerCase().includes(filter))
    .sort(([left], [right]) => left.localeCompare(right));
  const fragment = document.createDocumentFragment();
  for (const [name, value] of metrics) {
    const row = document.createElement("div");
    row.className = "metric-row";
    row.title = `${name}: ${String(value)}`;
    const label = document.createElement("span");
    label.textContent = humanize(name);
    const strong = document.createElement("strong");
    strong.textContent = formatMetric(value);
    row.append(label, strong);
    fragment.append(row);
  }
  elements.metricInventory.replaceChildren(fragment);
}

async function renderReplayList() {
  const previous = state.selectedReplayId;
  if (!state.replayEntries.some((entry) => entry.id === previous)) {
    state.selectedReplayId = state.replayEntries[0]?.id ?? null;
    state.replay = null;
  }
  const fragment = document.createDocumentFragment();
  if (!state.replayEntries.length) {
    const option = document.createElement("option");
    option.textContent = "No replays";
    option.value = "";
    fragment.append(option);
  } else {
    for (const entry of state.replayEntries) {
      const option = document.createElement("option");
      option.value = entry.id;
      option.textContent = `${entry.game_id} | ${entry.move_count} moves`;
      option.selected = entry.id === state.selectedReplayId;
      fragment.append(option);
    }
  }
  elements.replaySelect.replaceChildren(fragment);
  elements.replaySelect.disabled = state.replayEntries.length === 0;
  elements.replayEmpty.hidden = state.replayEntries.length > 0;
  elements.replayView.hidden = state.replayEntries.length === 0;

  if (state.selectedReplayId && (!state.replay || state.replay.id !== state.selectedReplayId)) {
    await loadReplay(state.selectedReplayId);
  } else if (state.replay) {
    renderReplay();
  }
}

async function loadReplay(replayId) {
  stopReplay();
  try {
    const replay = await fetchJson(`/api/replays/${encodeURIComponent(replayId)}`);
    if (state.selectedReplayId !== replayId) {
      return;
    }
    state.replay = replay;
    state.replayMove = 0;
    renderReplay();
  } catch (error) {
    showToast(error.message);
  }
}

function renderReplay() {
  const replay = state.replay;
  if (!replay) {
    return;
  }
  const move = Math.max(0, Math.min(state.replayMove, replay.actions.length));
  state.replayMove = move;
  const position = reconstructPosition(replay, move);
  elements.moveRange.max = String(replay.actions.length);
  elements.moveRange.value = String(move);
  elements.moveLabel.textContent = move === 0
    ? "Start"
    : `Move ${move} of ${replay.actions.length}, ${coordinate(replay.actions[move - 1], replay.board_size)}`;
  elements.captureOne.textContent = String(position.captures[1]);
  elements.captureTwo.textContent = String(position.captures[-1]);
  elements.replayMeta.textContent = `${replay.ruleset}, ${replay.board_size} x ${replay.board_size}, ${replay.actions.length} moves`;
  elements.replayResult.textContent = replayResult(replay);
  elements.replayPrevious.disabled = move === 0;
  elements.replayNext.disabled = move === replay.actions.length;
  renderBoard(replay.board_size, position.board, position.lastAction, position.issue);
}

function reconstructPosition(replay, moveCount) {
  const board = new Array(replay.board_size * replay.board_size).fill(0);
  const captures = { 1: 0, [-1]: 0 };
  let lastAction = null;
  let issue = null;
  for (let index = 0; index < moveCount; index += 1) {
    const action = replay.actions[index];
    const player = index % 2 === 0 ? 1 : -1;
    if (board[action] !== 0) {
      issue = `Invalid replay move at step ${index + 1}`;
      break;
    }
    board[action] = player;
    captures[player] += applyCaptures(board, replay.board_size, action, player);
    lastAction = action;
  }
  return { board, captures, lastAction, issue };
}

function applyCaptures(board, size, action, player) {
  const row = Math.floor(action / size);
  const column = action % size;
  const opponent = -player;
  let captured = 0;
  for (const [rowStep, columnStep] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    for (const direction of [-1, 1]) {
      const rowOne = row + direction * rowStep;
      const columnOne = column + direction * columnStep;
      const rowTwo = row + direction * rowStep * 2;
      const columnTwo = column + direction * columnStep * 2;
      const rowThree = row + direction * rowStep * 3;
      const columnThree = column + direction * columnStep * 3;
      if (rowThree < 0 || rowThree >= size || columnThree < 0 || columnThree >= size) {
        continue;
      }
      const one = rowOne * size + columnOne;
      const two = rowTwo * size + columnTwo;
      const three = rowThree * size + columnThree;
      if (board[one] === opponent && board[two] === opponent && board[three] === player) {
        board[one] = 0;
        board[two] = 0;
        captured += 1;
      }
    }
  }
  return captured;
}

function renderBoard(size, board, lastAction, issue) {
  elements.board.style.setProperty("--board-size", String(size));
  const fragment = document.createDocumentFragment();
  board.forEach((player, action) => {
    const cell = document.createElement("div");
    cell.className = "board-cell";
    cell.setAttribute("aria-hidden", "true");
    if (player !== 0) {
      const stone = document.createElement("span");
      stone.className = `stone ${player === 1 ? "player-one" : "player-two"}`;
      if (action === lastAction) {
        stone.classList.add("last");
      }
      cell.append(stone);
    }
    fragment.append(cell);
  });
  elements.board.replaceChildren(fragment);
  elements.board.setAttribute(
    "aria-label",
    issue || `Pente board after ${state.replayMove} moves`,
  );
  if (issue) {
    showToast(issue);
  }
}

function replayResult(replay) {
  if (replay.winner === null) {
    return "Draw";
  }
  const player = replay.winner === 1 ? "Player 1" : "Player 2";
  return replay.win_reason ? `${player}: ${humanize(replay.win_reason)}` : `${player} wins`;
}

function coordinate(action, boardSize) {
  const letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
  const row = Math.floor(action / boardSize);
  const column = action % boardSize;
  return `${letters[column] ?? column + 1}${boardSize - row}`;
}

function stepReplay(delta) {
  if (!state.replay) {
    return;
  }
  state.replayMove = Math.max(0, Math.min(state.replay.actions.length, state.replayMove + delta));
  renderReplay();
  if (state.replayMove === state.replay.actions.length) {
    stopReplay();
  }
}

function toggleReplay() {
  if (state.replayTimer !== null) {
    stopReplay();
    return;
  }
  if (!state.replay) {
    return;
  }
  if (state.replayMove >= state.replay.actions.length) {
    state.replayMove = 0;
    renderReplay();
  }
  elements.replayPlay.textContent = "Pause";
  state.replayTimer = window.setInterval(() => stepReplay(1), Number(elements.replaySpeed.value));
}

function stopReplay() {
  if (state.replayTimer !== null) {
    window.clearInterval(state.replayTimer);
    state.replayTimer = null;
  }
  elements.replayPlay.textContent = "Play";
}

function setConnection(kind, label) {
  elements.connection.className = `connection-state ${kind}`;
  elements.connection.textContent = label;
}

let toastTimer = null;
function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  if (toastTimer !== null) {
    window.clearTimeout(toastTimer);
  }
  toastTimer = window.setTimeout(() => elements.toast.classList.remove("visible"), 4500);
}

function humanize(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function numberOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatMetric(value) {
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  if (value === null || value === undefined) {
    return "--";
  }
  if (typeof value !== "number") {
    return String(value);
  }
  if (!Number.isFinite(value)) {
    return "--";
  }
  const absolute = Math.abs(value);
  if (Number.isInteger(value) && absolute < 1_000_000) {
    return value.toLocaleString();
  }
  if ((absolute > 0 && absolute < 0.001) || absolute >= 1_000_000) {
    return value.toExponential(3);
  }
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function compactNumber(value) {
  return new Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(value);
}

function formatInteger(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function formatIntegerOrDash(value) {
  return numberOrNull(value) === null ? "--" : formatInteger(value);
}

function formatPercent(value) {
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(value);
}

function relativeTime(timestamp) {
  if (!Number.isFinite(timestamp)) {
    return "never";
  }
  const seconds = Math.max(0, Math.round(Date.now() / 1000 - timestamp));
  if (seconds < 5) {
    return "just now";
  }
  if (seconds < 60) {
    return `${seconds}s ago`;
  }
  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}m ago`;
  }
  if (seconds < 86400) {
    return `${Math.floor(seconds / 3600)}h ago`;
  }
  return `${Math.floor(seconds / 86400)}d ago`;
}

function formatClock(timestamp) {
  return new Date(timestamp * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function bindEvents() {
  elements.refreshToggle.addEventListener("click", () => {
    state.paused = !state.paused;
    elements.refreshToggle.setAttribute("aria-pressed", String(state.paused));
    elements.refreshToggle.textContent = state.paused ? "Resume" : "Pause";
    setConnection(state.paused ? "paused" : "online", state.paused ? "Paused" : "Live");
    if (!state.paused) {
      void refresh();
    }
  });
  elements.metricSelect.addEventListener("change", () => {
    state.selectedMetric = elements.metricSelect.value;
    renderChart();
  });
  elements.metricFilter.addEventListener("input", renderMetricInventory);
  elements.replaySelect.addEventListener("change", () => {
    state.selectedReplayId = elements.replaySelect.value || null;
    state.replay = null;
    if (state.selectedReplayId) {
      void loadReplay(state.selectedReplayId);
    }
  });
  elements.moveRange.addEventListener("input", () => {
    stopReplay();
    state.replayMove = Number(elements.moveRange.value);
    renderReplay();
  });
  elements.replayPrevious.addEventListener("click", () => {
    stopReplay();
    stepReplay(-1);
  });
  elements.replayNext.addEventListener("click", () => {
    stopReplay();
    stepReplay(1);
  });
  elements.replayPlay.addEventListener("click", toggleReplay);
  elements.replaySpeed.addEventListener("change", () => {
    if (state.replayTimer !== null) {
      stopReplay();
      toggleReplay();
    }
  });
  window.addEventListener("resize", renderChart);
}

assertElements();
bindEvents();
void refresh();
window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
