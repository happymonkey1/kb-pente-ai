# Local training monitor

The monitoring server reads JSONL training telemetry. It uses only the Python standard library, binds to loopback by default, and has no effect on the training hot path. Test launching is off unless a local command catalog is provided.

Start it from the repository root:

```bash
./script/run-venv.sh python -m src.monitoring \
  --metrics-root metrics \
  --replay-root replays
```

Open `http://127.0.0.1:8765`. The dashboard polls for appended telemetry every two seconds. Overview shows current training signals, metric history, recent events, and CPU or CUDA phase metrics. Latest metrics contains the complete metric inventory. Replay contains optional game playback.

NN architecture shows the PenteNet input, stem, residual tower, policy head, and value head. It also shows parameter counts, estimated multiply-accumulate operations, FP32 weight size, trainable layer count, configuration, runtime, and parameter distribution.

Theme defaults to the operating system setting. The header control can select System, Light, or Dark. Explicit Light and Dark choices are saved in browser storage for later visits.

## NN architecture data

Architecture data comes from `run-manifest-step-*.json`. The monitor matches a manifest to telemetry using `outputs.telemetry` or `training_run_id`. It reads the existing `model`, `runtime`, and `program_arguments.ruleset` fields, so training does not need to repeat static model data in telemetry.

By default, the server checks the metrics directory, the current directory, and immediate child directories. Add more locations by repeating `--manifest-root`:

```bash
./script/run-venv.sh python -m src.monitoring \
  --metrics-root metrics \
  --manifest-root . \
  --manifest-root /path/to/model-runs
```

The current manifest provides all required fields:

| Field | Use |
|---|---|
| `training_run_id` | Stable run match. |
| `outputs.telemetry` | Exact telemetry file match. |
| `model.board_size` and `model.action_size` | Input and policy output shapes. |
| `model.input_planes` | Input feature depth. |
| `model.num_res_blocks` and `model.num_channels` | Residual tower shape. |
| `model.hidden_fc_size` | Value head shape. |
| `runtime.device`, `runtime.device_name`, and `runtime.compiled` | Runtime summary. |

Parameter counts are exact for the current PenteNet definition. Compute is an estimate based on convolution and dense multiply-accumulate operations for one position. It does not include batch norm, activation, softmax, memory transfer, compiler fusion, or MCTS work. Runs without a matching manifest show an unavailable state instead of guessed values.

## Launch tests

Test launching uses a JSON catalog so the browser can select only commands approved by the operator. It never accepts a command or extra arguments from an HTTP request. Only one launched test can run at a time, and the dashboard asks for confirmation before starting it.

The checked-in example includes focused monitoring tests plus CPU and CUDA search benchmarks. Review it before enabling the feature, then start the monitor with:

```bash
./script/run-venv.sh python -m src.monitoring \
  --metrics-root metrics \
  --replay-root replays \
  --test-config docs/monitoring-tests.example.json \
  --test-root . \
  --test-log-root .monitoring/test-runs
```

The catalog format is:

```json
{
  "schema_version": 1,
  "tests": [
    {
      "id": "monitoring-unit-tests",
      "name": "Monitoring unit tests",
      "description": "Run focused monitoring tests.",
      "command": ["./script/run-venv.sh", "python", "-m", "unittest"]
    }
  ]
}
```

Commands run without a shell from `--test-root`. Output is written to `--test-log-root`. Run status is kept in server memory and resets when the server restarts. Test launching is rejected when the server binds to a non-loopback address.

CUDA rows show sampled GPU load, memory-controller activity, peak Torch allocation and reservation, and sampling errors. Memory activity is the percentage of time that global device memory was being read or written. It is not the percentage of memory capacity in use. CPU rows show normalized process CPU use and resident memory when the producer emits the CPU metric contract below. Older CPU records without those fields are still shown as CPU but have no device samples. Older CUDA records without the current prefixed metrics cannot be distinguished from CPU records.

The dashboard serves a local copy of Pico CSS 2.1.1 from `src/monitoring/static/pico.min.css`. It does not load styles, scripts, fonts, or other assets from a CDN. Pico's MIT license is stored beside the stylesheet in `pico.LICENSE.md`.

## APIs

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Server and configured-root availability. |
| `GET /api/runs` | Discovered telemetry runs and activity status. |
| `GET /api/runs/{id}/summary` | Latest metrics, device and architecture summaries, event counts, and numeric statistics. |
| `GET /api/runs/{id}/records?after=0&limit=500` | Bounded append-order records. |
| `GET /api/replays?run_id={run_key}` | Replay sample metadata for a run. |
| `GET /api/replays/{id}` | One validated replay sample. |
| `GET /api/test-launcher` | Configured tests, active test, and recent results. |
| `POST /api/test-runs` | Launch one configured test by `test_id`. Returns 202, 403 when disabled, or 409 when another test is active. |
| `GET /metrics` | Prometheus text metrics for the monitoring server. |

Run identifiers are relative paths beneath the configured metrics root. Responses are capped, files have a configurable size limit, path traversal is rejected, and a partial trailing JSONL line is ignored while a producer finishes its append.

Current telemetry records include a top-level `run_id`. The monitor uses it to match replay samples even when the telemetry filename differs from the training run identifier. Records written before `run_id` was added remain supported and use the relative filename without `.jsonl`.

## CPU telemetry producer contract

The dashboard is ready for CPU phase metrics, but the training process must measure and emit them. Add these values to each `training_iteration` metric object for a CPU run:

| Metric | Type | Meaning |
|---|---|---|
| `device_type` | string | Required value: `cpu`. CUDA runs should emit `cuda`. |
| `cpu_logical_core_count` | integer | Number of logical CPUs used to normalize process utilization. |
| `{phase}_cpu_utilization_samples` | integer | Successful samples collected during the phase. |
| `{phase}_cpu_mean_process_utilization_percent` | number | Mean training-process CPU use from 0 to 100 percent. |
| `{phase}_cpu_p95_process_utilization_percent` | number | 95th percentile training-process CPU use from 0 to 100 percent. |
| `{phase}_cpu_max_process_utilization_percent` | number | Maximum training-process CPU use from 0 to 100 percent. |
| `{phase}_cpu_mean_resident_memory_bytes` | integer | Mean resident memory of the training process. |
| `{phase}_cpu_peak_resident_memory_bytes` | integer | Maximum sampled resident memory of the training process. |
| `{phase}_cpu_sampling_errors` | integer | Failed utilization or memory samples. |

`{phase}` is `self_play` or `learner`. Professional-only iterations omit the self-play fields. Sample at the same 250 ms interval as CUDA telemetry. Normalize each process CPU sample as:

```text
100 * process_cpu_time_delta / (wall_time_delta * cpu_logical_core_count)
```

Clamp measurement noise to the range from 0 to 100. Measure the training process only, not total host CPU use. Resident memory is the current process RSS at each sample. Sampling must be bounded, must stop when the phase ends or raises, and must report errors without leaking a sampling thread.

## Replay samples

The server never loads pickle replay-buffer snapshots. Optional browser playback consumes safe JSONL sidecars with one completed game per line:

```json
{"schema_version":1,"run_id":"training","game_id":"iteration-1-game-0","recorded_at_unix":0.0,"board_size":19,"ruleset":"standard","actions":[180,0],"winner":1,"win_reason":"line"}
```

`src.monitoring.replay_writer.JsonlReplaySampleSink` provides a locked append writer. Producer integration should record only a bounded deterministic sample of completed games. It should not serialize positions, policies, models, or replay-buffer entries for the webserver.

## Network exposure

The server has no authentication because its default role is a local development tool. A non-loopback bind is rejected unless `--allow-remote` is explicit. Test launching stays unavailable on non-loopback binds even with `--allow-remote`. Use an authenticated reverse proxy before exposing read-only monitoring beyond a trusted machine.
