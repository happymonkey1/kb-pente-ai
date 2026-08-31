"use strict";

const POLL_INTERVAL_MS = 2000;
const MAX_VISIBLE_RECORDS = 1000;
const THEME_STORAGE_KEY = "kb-pente-monitor-theme";

const state = {
  runs: [],
  selectedRunId: null,
  summary: null,
  records: [],
  paused: false,
  polling: false,
  selectedTab: "overview",
  selectedMetric: null,
  replayEntries: [],
  selectedReplayId: null,
  replay: null,
  replayMove: 0,
  replayTimer: null,
  testLauncher: {
    enabled: false,
    tests: [],
    active_run: null,
    recent_runs: [],
  },
  selectedTestId: null,
  launchingTest: false,
};

const elements = {
  connection: document.querySelector("#connection-state"),
  themeSelect: document.querySelector("#theme-select"),
  refreshToggle: document.querySelector("#refresh-toggle"),
  testLauncherOpen: document.querySelector("#test-launcher-open"),
  runCount: document.querySelector("#run-count"),
  runList: document.querySelector("#run-list"),
  emptyState: document.querySelector("#empty-state"),
  runView: document.querySelector("#run-view"),
  runStatus: document.querySelector("#run-status"),
  runUpdated: document.querySelector("#run-updated"),
  runName: document.querySelector("#run-name"),
  latestStep: document.querySelector("#latest-step"),
  runTabs: document.querySelector("#run-tabs"),
  overviewTab: document.querySelector("#tab-overview"),
  architectureTab: document.querySelector("#tab-architecture"),
  metricsTab: document.querySelector("#tab-metrics"),
  replayTab: document.querySelector("#tab-replay"),
  overviewPanel: document.querySelector("#panel-overview"),
  architecturePanel: document.querySelector("#panel-architecture"),
  metricsPanel: document.querySelector("#panel-metrics"),
  replayPanel: document.querySelector("#panel-replay"),
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
  signalDevice: document.querySelector("#signal-device"),
  signalDeviceDetail: document.querySelector("#signal-device-detail"),
  deviceMetricsState: document.querySelector("#device-metrics-state"),
  deviceMetricsEmpty: document.querySelector("#device-metrics-empty"),
  deviceMetricsTable: document.querySelector("#device-metrics-table"),
  deviceMetricsHead: document.querySelector("#device-metrics-head"),
  deviceMetricsBody: document.querySelector("#device-metrics-body"),
  architectureSource: document.querySelector("#architecture-source"),
  architectureEmpty: document.querySelector("#architecture-empty"),
  architectureView: document.querySelector("#architecture-view"),
  architectureParameters: document.querySelector("#architecture-parameters"),
  architectureFlops: document.querySelector("#architecture-flops"),
  architectureSize: document.querySelector("#architecture-size"),
  architectureLayers: document.querySelector("#architecture-layers"),
  architectureDevice: document.querySelector("#architecture-device"),
  architectureInputShape: document.querySelector("#architecture-input-shape"),
  architectureStemShape: document.querySelector("#architecture-stem-shape"),
  architectureTowerShape: document.querySelector("#architecture-tower-shape"),
  architecturePolicyShape: document.querySelector("#architecture-policy-shape"),
  architectureValueShape: document.querySelector("#architecture-value-shape"),
  architectureParameterBars: document.querySelector("#architecture-parameter-bars"),
  architectureConfig: document.querySelector("#architecture-config"),
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
  testLauncherDialog: document.querySelector("#test-launcher-dialog"),
  testLauncherCloseIcon: document.querySelector("#test-launcher-close-icon"),
  testLauncherClose: document.querySelector("#test-launcher-close"),
  testLauncherMessage: document.querySelector("#test-launcher-message"),
  testLauncherControls: document.querySelector("#test-launcher-controls"),
  testSelect: document.querySelector("#test-select"),
  testDescription: document.querySelector("#test-description"),
  testCommand: document.querySelector("#test-command"),
  testActiveRun: document.querySelector("#test-active-run"),
  testRecentRuns: document.querySelector("#test-recent-runs"),
  testRecentRunsBody: document.querySelector("#test-recent-runs-body"),
  testLauncherConfirmOpen: document.querySelector("#test-launcher-confirm-open"),
  testConfirmDialog: document.querySelector("#test-confirm-dialog"),
  testConfirmDescription: document.querySelector("#test-confirm-description"),
  testConfirmCommand: document.querySelector("#test-confirm-command"),
  testConfirmCancel: document.querySelector("#test-confirm-cancel"),
  testConfirmLaunch: document.querySelector("#test-confirm-launch"),
  toast: document.querySelector("#toast"),
};

function assertElements() {
  for (const [name, element] of Object.entries(elements)) {
    if (!element) {
      throw new Error(`Dashboard element is missing: ${name}`);
    }
  }
}

function tabEntries() {
  return [
    ["overview", elements.overviewTab, elements.overviewPanel],
    ["architecture", elements.architectureTab, elements.architecturePanel],
    ["metrics", elements.metricsTab, elements.metricsPanel],
    ["replay", elements.replayTab, elements.replayPanel],
  ];
}

function selectTab(name, focus = false) {
  const entries = tabEntries();
  const selected = entries.find(([entryName]) => entryName === name) ?? entries[0];
  state.selectedTab = selected[0];
  for (const [entryName, button, panel] of entries) {
    const active = entryName === state.selectedTab;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    panel.hidden = !active;
  }
  if (state.selectedTab !== "replay") {
    stopReplay();
  }
  if (state.selectedTab === "overview") {
    renderChart();
  }
  if (focus) {
    selected[1].focus();
  }
}

function themePreference() {
  try {
    const preference = window.localStorage.getItem(THEME_STORAGE_KEY);
    return preference === "light" || preference === "dark" ? preference : "system";
  } catch {
    return "system";
  }
}

function setTheme(preference) {
  const selected = preference === "light" || preference === "dark" ? preference : "system";
  if (selected === "system") {
    document.documentElement.removeAttribute("data-theme");
  } else {
    document.documentElement.dataset.theme = selected;
  }
  elements.themeSelect.value = selected;
  try {
    if (selected === "system") {
      window.localStorage.removeItem(THEME_STORAGE_KEY);
    } else {
      window.localStorage.setItem(THEME_STORAGE_KEY, selected);
    }
  } catch {
    // The selected theme still applies for this page when storage is unavailable.
  }
  renderChart();
}

async function fetchJson(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { Accept: "application/json", ...(options.headers ?? {}) },
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
    const [payload, testLauncher] = await Promise.all([
      fetchJson("/api/runs"),
      fetchJson("/api/test-launcher"),
    ]);
    state.runs = payload.runs;
    state.testLauncher = testLauncher;
    if (!state.runs.some((run) => run.id === state.selectedRunId)) {
      state.selectedRunId = state.runs[0]?.id ?? null;
      state.selectedReplayId = null;
      state.replay = null;
    }
    renderRunList();
    renderTestLauncher();
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

async function refreshTestLauncher() {
  try {
    state.testLauncher = await fetchJson("/api/test-launcher");
    renderTestLauncher();
  } catch (error) {
    showToast(error.message);
  }
}

function selectedTest() {
  const tests = Array.isArray(state.testLauncher.tests) ? state.testLauncher.tests : [];
  return tests.find((test) => test.id === state.selectedTestId) ?? null;
}

function renderTestLauncher() {
  const launcher = state.testLauncher;
  const tests = Array.isArray(launcher.tests) ? launcher.tests : [];
  if (!tests.some((test) => test.id === state.selectedTestId)) {
    state.selectedTestId = tests[0]?.id ?? null;
  }

  const options = document.createDocumentFragment();
  for (const test of tests) {
    const option = document.createElement("option");
    option.value = test.id;
    option.textContent = test.name;
    option.selected = test.id === state.selectedTestId;
    options.append(option);
  }
  elements.testSelect.replaceChildren(options);

  const test = selectedTest();
  elements.testLauncherControls.hidden = !launcher.enabled;
  elements.testLauncherMessage.textContent = launcher.enabled
    ? "Choose one configured test. Only one test can run at a time."
    : "Test launching is off. Start the server with --test-config to enable it.";
  elements.testDescription.textContent = test?.description || "No description.";
  elements.testCommand.textContent = test?.command || "";

  const activeRun = launcher.active_run;
  elements.testActiveRun.textContent = activeRun
    ? `${activeRun.name} is ${activeRun.status}. Started ${relativeTime(activeRun.started_at_unix)}.`
    : "No test is running.";

  const recentRuns = (launcher.recent_runs ?? []).filter((run) => run.status !== "running");
  const rows = document.createDocumentFragment();
  for (const run of recentRuns) {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    const result = document.createElement("td");
    const started = document.createElement("td");
    name.textContent = run.name;
    result.textContent = run.exit_code === null
      ? humanize(run.status)
      : `${humanize(run.status)} (${run.exit_code})`;
    started.textContent = relativeTime(run.started_at_unix);
    row.append(name, result, started);
    rows.append(row);
  }
  elements.testRecentRunsBody.replaceChildren(rows);
  elements.testRecentRuns.hidden = recentRuns.length === 0;

  const cannotLaunch = !launcher.enabled || !test || activeRun !== null || state.launchingTest;
  elements.testLauncherConfirmOpen.disabled = cannotLaunch;
  elements.testLauncherConfirmOpen.textContent = state.launchingTest ? "Starting" : "Launch";
  elements.testLauncherOpen.setAttribute(
    "aria-label",
    activeRun ? `Test running: ${activeRun.name}` : "Run test",
  );
}

function openTestConfirmation() {
  const test = selectedTest();
  if (!test || elements.testLauncherConfirmOpen.disabled) {
    return;
  }
  elements.testConfirmDescription.textContent = test.name;
  elements.testConfirmCommand.textContent = test.command;
  elements.testLauncherDialog.close();
  elements.testConfirmDialog.showModal();
}

async function launchSelectedTest() {
  const test = selectedTest();
  if (!test || state.launchingTest) {
    return;
  }

  state.launchingTest = true;
  elements.testConfirmLaunch.disabled = true;
  renderTestLauncher();
  try {
    await fetchJson("/api/test-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ test_id: test.id }),
    });
    elements.testConfirmDialog.close();
    showToast(`${test.name} started.`);
    await refreshTestLauncher();
    elements.testLauncherDialog.showModal();
  } catch (error) {
    showToast(error.message);
  } finally {
    state.launchingTest = false;
    elements.testConfirmLaunch.disabled = false;
    renderTestLauncher();
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

  renderSignals(summary.latest_metrics);
  renderDevice(summary.device);
  renderArchitecture(summary.architecture);
  renderMetricSelector(summary.numeric_metrics);
  renderChart();
  renderEvents();
  renderMetricInventory();
}

function renderArchitecture(architecture) {
  if (!architecture?.available) {
    elements.architectureSource.textContent = "Not available";
    elements.architectureEmpty.textContent = architecture?.reason
      || "No architecture data for this run.";
    elements.architectureEmpty.hidden = false;
    elements.architectureView.hidden = true;
    return;
  }

  const config = architecture.config;
  const metrics = architecture.metrics;
  const runtime = architecture.runtime ?? {};
  const manifestStep = architecture.manifest?.start_iteration;
  elements.architectureSource.textContent = Number.isInteger(manifestStep)
    ? `Run manifest, step ${formatInteger(manifestStep)}`
    : "Run manifest";
  elements.architectureEmpty.hidden = true;
  elements.architectureView.hidden = false;

  elements.architectureParameters.textContent = compactNumber(metrics.parameter_count);
  elements.architectureParameters.title = formatInteger(metrics.parameter_count);
  elements.architectureFlops.textContent = `${compactNumber(
    metrics.multiply_accumulates_per_position,
  )} MACs`;
  elements.architectureFlops.title = formatInteger(
    metrics.multiply_accumulates_per_position,
  );
  elements.architectureSize.textContent = formatBytes(metrics.estimated_fp32_bytes);
  elements.architectureLayers.textContent = formatInteger(metrics.parameterized_layer_count);
  elements.architectureDevice.textContent = architectureRuntimeLabel(runtime);

  elements.architectureInputShape.textContent = [
    config.input_planes,
    config.board_size,
    config.board_size,
  ].join(" x ");
  elements.architectureStemShape.textContent = `${formatInteger(config.channels)} channels`;
  elements.architectureTowerShape.textContent = `${formatInteger(config.residual_blocks)} blocks`;
  elements.architecturePolicyShape.textContent = `${formatInteger(config.action_size)} actions`;
  elements.architectureValueShape.textContent = `${formatInteger(
    config.value_hidden_size,
  )} hidden -> 1`;

  const stageLabels = {
    stem: "Stem",
    residual_tower: "Residual tower",
    policy_head: "Policy head",
    value_head: "Value head",
  };
  const parameterRows = document.createDocumentFragment();
  for (const [name, value] of Object.entries(metrics.parameters_by_stage)) {
    const row = document.createElement("div");
    row.className = "parameter-row";
    const label = document.createElement("span");
    const progress = document.createElement("progress");
    const amount = document.createElement("strong");
    label.textContent = stageLabels[name] ?? humanize(name);
    progress.max = metrics.parameter_count;
    progress.value = value;
    progress.setAttribute(
      "aria-label",
      `${label.textContent}: ${formatInteger(value)} parameters`,
    );
    amount.textContent = `${formatPercent(value / metrics.parameter_count)} · ${compactNumber(value)}`;
    row.append(label, progress, amount);
    parameterRows.append(row);
  }
  elements.architectureParameterBars.replaceChildren(parameterRows);

  const configuration = [
    ["Board", `${config.board_size} x ${config.board_size}`],
    ["Input planes", formatInteger(config.input_planes)],
    ["Trunk channels", formatInteger(config.channels)],
    ["Residual blocks", formatInteger(config.residual_blocks)],
    ["Value hidden size", formatInteger(config.value_hidden_size)],
    ["Policy outputs", formatInteger(config.action_size)],
    ["Estimated FLOPs", formatInteger(metrics.estimated_flops_per_position)],
    ["Trunk activation values", formatInteger(metrics.trunk_activation_values_per_position)],
    ["Ruleset", architecture.ruleset ? humanize(architecture.ruleset) : "Not reported"],
    ["Compiled", architectureCompiledLabel(runtime.compiled)],
    ["Device", runtime.device_name || runtime.device || "Not reported"],
    ["Torch", runtime.torch || "Not reported"],
  ];
  const details = document.createDocumentFragment();
  for (const [name, value] of configuration) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = name;
    description.textContent = value;
    row.append(term, description);
    details.append(row);
  }
  elements.architectureConfig.replaceChildren(details);
}

function architectureRuntimeLabel(runtime) {
  const device = typeof runtime.device === "string" ? runtime.device.toUpperCase() : "Unknown";
  return runtime.compiled === true ? `${device}, compiled` : device;
}

function architectureCompiledLabel(compiled) {
  if (compiled === true) {
    return "Yes";
  }
  if (compiled === false) {
    return "No";
  }
  return "Not reported";
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

function renderDevice(device) {
  const deviceType = device?.type ?? "unknown";
  const phases = device?.phases ?? {};
  const rows = [
    ["Self-play", phases.self_play],
    ["Learner", phases.learner],
  ].filter(([, metrics]) => metrics !== null && metrics !== undefined);

  elements.signalDevice.textContent = deviceType === "unknown" ? "--" : deviceType.toUpperCase();
  elements.deviceMetricsHead.replaceChildren();
  elements.deviceMetricsBody.replaceChildren();

  if (deviceType !== "cuda" && deviceType !== "cpu") {
    elements.signalDeviceDetail.textContent = "Not reported";
    elements.deviceMetricsState.textContent = "Not reported";
    elements.deviceMetricsEmpty.textContent = "This run does not report a device.";
    elements.deviceMetricsEmpty.hidden = false;
    elements.deviceMetricsTable.hidden = true;
    return;
  }

  const preferred = phases.learner ?? phases.self_play;
  const utilizationName = deviceType === "cuda"
    ? "mean_utilization_percent"
    : "mean_process_utilization_percent";
  const average = numberOrNull(preferred?.[utilizationName]);
  const coreCount = Number.isInteger(device?.logical_core_count)
    ? `, ${device.logical_core_count} logical cores`
    : "";
  elements.signalDeviceDetail.textContent = average === null
    ? `${deviceType.toUpperCase()} training${coreCount}`
    : `${formatDevicePercent(average)} average ${deviceType.toUpperCase()} load`;
  elements.deviceMetricsState.textContent = deviceType.toUpperCase();

  if (!rows.length) {
    elements.deviceMetricsEmpty.textContent = `${deviceType.toUpperCase()} run. No device samples yet.`;
    elements.deviceMetricsEmpty.hidden = false;
    elements.deviceMetricsTable.hidden = true;
    return;
  }

  const columns = deviceType === "cuda"
    ? [
        ["Phase", null, null],
        ["Samples", "utilization_samples", formatIntegerOrDash],
        ["GPU avg", "mean_utilization_percent", formatDevicePercent],
        ["GPU p95", "p95_utilization_percent", formatDevicePercent],
        ["GPU max", "max_utilization_percent", formatDevicePercent],
        ["Memory activity avg", "mean_device_memory_percent", formatDevicePercent],
        ["Memory activity max", "max_device_memory_percent", formatDevicePercent],
        ["Torch allocated peak", "peak_memory_allocated_bytes", formatBytes],
        ["Torch reserved peak", "peak_memory_reserved_bytes", formatBytes],
        ["Errors", "utilization_sampling_errors", formatIntegerOrDash],
      ]
    : [
        ["Phase", null, null],
        ["Samples", "utilization_samples", formatIntegerOrDash],
        ["Process CPU avg", "mean_process_utilization_percent", formatDevicePercent],
        ["Process CPU p95", "p95_process_utilization_percent", formatDevicePercent],
        ["Process CPU max", "max_process_utilization_percent", formatDevicePercent],
        ["Memory avg", "mean_resident_memory_bytes", formatBytes],
        ["Memory peak", "peak_resident_memory_bytes", formatBytes],
        ["Errors", "sampling_errors", formatIntegerOrDash],
      ];
  const headingFragment = document.createDocumentFragment();
  for (const [label] of columns) {
    const heading = document.createElement("th");
    heading.scope = "col";
    heading.textContent = label;
    headingFragment.append(heading);
  }
  elements.deviceMetricsHead.replaceChildren(headingFragment);

  const fragment = document.createDocumentFragment();
  for (const [label, metrics] of rows) {
    const row = document.createElement("tr");
    const values = columns.map(([, name, formatter]) => (
      name === null ? label : formatter(metrics[name])
    ));
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    fragment.append(row);
  }
  elements.deviceMetricsBody.replaceChildren(fragment);
  elements.deviceMetricsEmpty.hidden = true;
  elements.deviceMetricsTable.hidden = false;
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

function formatDevicePercent(value) {
  const number = numberOrNull(value);
  return number === null ? "--" : `${formatMetric(number)}%`;
}

function formatBytes(value) {
  const number = numberOrNull(value);
  if (number === null) {
    return "--";
  }
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let scaled = Math.max(0, number);
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${scaled.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${units[unit]}`;
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
  elements.themeSelect.addEventListener("change", () => {
    setTheme(elements.themeSelect.value);
  });
  elements.testLauncherOpen.addEventListener("click", () => {
    renderTestLauncher();
    elements.testLauncherDialog.showModal();
    void refreshTestLauncher();
  });
  elements.testLauncherCloseIcon.addEventListener("click", () => {
    elements.testLauncherDialog.close();
  });
  elements.testLauncherClose.addEventListener("click", () => {
    elements.testLauncherDialog.close();
  });
  elements.testSelect.addEventListener("change", () => {
    state.selectedTestId = elements.testSelect.value || null;
    renderTestLauncher();
  });
  elements.testLauncherConfirmOpen.addEventListener("click", openTestConfirmation);
  elements.testConfirmCancel.addEventListener("click", () => {
    elements.testConfirmDialog.close();
    elements.testLauncherDialog.showModal();
  });
  elements.testConfirmLaunch.addEventListener("click", () => {
    void launchSelectedTest();
  });
  elements.runTabs.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }
    const button = event.target.closest("[role=tab]");
    if (button?.dataset.tab) {
      selectTab(button.dataset.tab);
    }
  });
  elements.runTabs.addEventListener("keydown", (event) => {
    const entries = tabEntries();
    const currentIndex = entries.findIndex(([, button]) => button === event.target);
    if (currentIndex < 0) {
      return;
    }
    let nextIndex = currentIndex;
    if (event.key === "ArrowLeft") {
      nextIndex = (currentIndex - 1 + entries.length) % entries.length;
    } else if (event.key === "ArrowRight") {
      nextIndex = (currentIndex + 1) % entries.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = entries.length - 1;
    } else {
      return;
    }
    event.preventDefault();
    selectTab(entries[nextIndex][0], true);
  });
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
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (elements.themeSelect.value === "system") {
      renderChart();
    }
  });
}

assertElements();
elements.themeSelect.value = themePreference();
bindEvents();
selectTab(state.selectedTab);
renderTestLauncher();
void refresh();
window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
window.setInterval(() => {
  if (elements.testLauncherDialog.open || elements.testConfirmDialog.open) {
    void refreshTestLauncher();
  }
}, POLL_INTERVAL_MS);
