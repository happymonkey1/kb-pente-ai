# Local training monitor

The monitoring server is a read-only observer for JSONL training telemetry. It uses only the Python standard library, binds to loopback by default, and has no effect on the training hot path.

Start it from the repository root:

```bash
./script/run-venv.sh python -m src.monitoring \
  --metrics-root metrics \
  --replay-root replays
```

Open `http://127.0.0.1:8765`. The dashboard polls for appended telemetry every two seconds. It provides current training signals, numeric metric histories and statistics, recent events, a complete latest-value inventory, and optional replay playback.

The dashboard serves a local copy of Pico CSS 2.1.1 from `src/monitoring/static/pico.min.css`. It does not load styles, scripts, fonts, or other assets from a CDN. Pico's MIT license is stored beside the stylesheet in `pico.LICENSE.md`.

## Read-only APIs

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Server and configured-root availability. |
| `GET /api/runs` | Discovered telemetry runs and activity status. |
| `GET /api/runs/{id}/summary` | Latest metrics, event counts, and numeric statistics. |
| `GET /api/runs/{id}/records?after=0&limit=500` | Bounded append-order records. |
| `GET /api/replays?run_id={run_key}` | Replay sample metadata for a run. |
| `GET /api/replays/{id}` | One validated replay sample. |
| `GET /metrics` | Prometheus text metrics for the monitoring server. |

Run identifiers are relative paths beneath the configured metrics root. Responses are capped, files have a configurable size limit, path traversal is rejected, and a partial trailing JSONL line is ignored while a producer finishes its append.

## Replay samples

The server never loads pickle replay-buffer snapshots. Optional browser playback consumes safe JSONL sidecars with one completed game per line:

```json
{"schema_version":1,"run_id":"training","game_id":"iteration-1-game-0","recorded_at_unix":0.0,"board_size":19,"ruleset":"standard","actions":[180,0],"winner":1,"win_reason":"line"}
```

`src.monitoring.replay_writer.JsonlReplaySampleSink` provides a locked append writer. Producer integration should record only a bounded deterministic sample of completed games. It should not serialize positions, policies, models, or replay-buffer entries for the webserver.

## Network exposure

The server has no authentication because its default role is a local development tool. A non-loopback bind is rejected unless `--allow-remote` is explicit. Use an authenticated reverse proxy before exposing it beyond a trusted machine.
