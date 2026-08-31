# Local training monitor

The monitoring server is a read-only observer for JSONL training telemetry. It uses only the Python standard library, binds to loopback by default, and has no effect on the training hot path.

Start it from the repository root:

```bash
./script/run-venv.sh python -m src.monitoring \
  --metrics-root metrics \
  --replay-root replays
```

Open `http://127.0.0.1:8765`. The dashboard polls for appended telemetry every two seconds. Overview shows current training signals, metric history, recent events, and CPU or CUDA phase metrics. Latest metrics contains the complete metric inventory. Replay contains optional game playback.

Theme defaults to the operating system setting. The header control can select System, Light, or Dark. Explicit Light and Dark choices are saved in browser storage for later visits.

CUDA rows show sampled GPU load, memory-controller activity, peak Torch allocation and reservation, and sampling errors. Memory activity is the percentage of time that global device memory was being read or written. It is not the percentage of memory capacity in use. CPU rows show normalized process CPU use and resident memory when the producer emits the CPU metric contract below. Older CPU records without those fields are still shown as CPU but have no device samples. Older CUDA records without the current prefixed metrics cannot be distinguished from CPU records.

The dashboard serves a local copy of Pico CSS 2.1.1 from `src/monitoring/static/pico.min.css`. It does not load styles, scripts, fonts, or other assets from a CDN. Pico's MIT license is stored beside the stylesheet in `pico.LICENSE.md`.

## Read-only APIs

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Server and configured-root availability. |
| `GET /api/runs` | Discovered telemetry runs and activity status. |
| `GET /api/runs/{id}/summary` | Latest metrics, device summary, event counts, and numeric statistics. |
| `GET /api/runs/{id}/records?after=0&limit=500` | Bounded append-order records. |
| `GET /api/replays?run_id={run_key}` | Replay sample metadata for a run. |
| `GET /api/replays/{id}` | One validated replay sample. |
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

The server has no authentication because its default role is a local development tool. A non-loopback bind is rejected unless `--allow-remote` is explicit. Use an authenticated reverse proxy before exposing it beyond a trusted machine.
