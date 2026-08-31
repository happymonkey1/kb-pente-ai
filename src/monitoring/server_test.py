from __future__ import annotations

from http import HTTPStatus
from http.client import HTTPMessage
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from src.monitoring.models import ReplaySample
from src.monitoring.replay_writer import JsonlReplaySampleSink
from src.monitoring.server import MonitoringHTTPServer, build_server


class MonitoringServerTest(unittest.TestCase):
    temporary_directory: TemporaryDirectory[str]
    server: MonitoringHTTPServer
    server_thread: threading.Thread
    base_url: str

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        metrics = root / "metrics"
        replays = root / "replays"
        metrics.mkdir()
        telemetry = {
            "schema_version": 1,
            "timestamp_unix": 100.0,
            "run_id": "training",
            "event": "training_iteration",
            "step": 4,
            "metrics": {"loss": 0.75, "games": 8},
        }
        (metrics / "training.jsonl").write_text(json.dumps(telemetry) + "\n", encoding="utf-8")
        manifest_root = root / "model"
        manifest_root.mkdir()
        (manifest_root / "run-manifest-step-0.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at_utc": "2026-08-30T00:00:00+00:00",
                    "training_run_id": "training",
                    "start_iteration": 0,
                    "model": {
                        "board_size": 5,
                        "action_size": 25,
                        "input_planes": 4,
                        "num_res_blocks": 1,
                        "num_channels": 8,
                        "hidden_fc_size": 16,
                    },
                    "outputs": {"telemetry": str(metrics / "training.jsonl")},
                    "runtime": {"device": "cpu", "compiled": False},
                    "program_arguments": {"ruleset": "freestyle"},
                }
            ),
            encoding="utf-8",
        )
        JsonlReplaySampleSink(replays / "samples.jsonl").emit(
            ReplaySample.from_object(
                {
                    "schema_version": 1,
                    "run_id": "training",
                    "game_id": "sample-1",
                    "recorded_at_unix": 101.0,
                    "board_size": 5,
                    "ruleset": "freestyle",
                    "actions": [0, 1, 5],
                    "winner": 1,
                    "win_reason": "line",
                }
            )
        )
        try:
            self.server = build_server(
                host="127.0.0.1",
                port=0,
                metrics_root=metrics,
                replay_root=replays,
                manifest_roots=[root],
            )
        except PermissionError as error:
            self.temporary_directory.cleanup()
            self.skipTest(f"Local sockets are unavailable in this environment: {error}")
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)
        self.temporary_directory.cleanup()

    def test_serves_dashboard_run_api_records_and_security_headers(self) -> None:
        dashboard, headers = self._get("/")
        pico_css, pico_headers = self._get("/pico.min.css")
        theme_script, theme_headers = self._get("/theme.js")
        runs, _ = self._get_json("/api/runs")
        test_launcher, _ = self._get_json("/api/test-launcher")
        summary, _ = self._get_json("/api/runs/training.jsonl/summary")
        records, _ = self._get_json("/api/runs/training.jsonl/records?after=0&limit=10")

        dashboard_text = dashboard.decode("utf-8")
        self.assertIn("Training monitor", dashboard_text)
        self.assertIn('href="/pico.min.css"', dashboard_text)
        self.assertNotIn("cdn.", dashboard_text)
        self.assertIn(b"Pico CSS", pico_css[:200])
        self.assertEqual("text/css; charset=utf-8", pico_headers["Content-Type"])
        self.assertIn(b"localStorage", theme_script)
        self.assertEqual("text/javascript; charset=utf-8", theme_headers["Content-Type"])
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual("nosniff", headers["X-Content-Type-Options"])
        self.assertIn("Device metrics", dashboard_text)
        self.assertIn('id="theme-select"', dashboard_text)
        self.assertIn('role="tablist"', dashboard_text)
        self.assertIn('data-tab="metrics"', dashboard_text)
        self.assertIn('data-tab="replay"', dashboard_text)
        self.assertIn('data-tab="architecture"', dashboard_text)
        self.assertIn('id="test-launcher-dialog"', dashboard_text)
        self.assertLess(
            dashboard_text.index('id="test-launcher-open"'),
            dashboard_text.index('id="theme-select"'),
        )
        self.assertNotIn("Reads JSONL files", dashboard_text)
        self.assertNotIn('id="run-context"', dashboard_text)
        self.assertEqual(
            {
                "active_run": None,
                "enabled": False,
                "recent_runs": [],
                "tests": [],
            },
            test_launcher,
        )
        self.assertEqual("training.jsonl", runs["runs"][0]["id"])
        self.assertEqual(0.75, summary["latest_metrics"]["loss"])
        self.assertEqual("cpu", summary["device"]["type"])
        self.assertEqual(3_253, summary["architecture"]["metrics"]["parameter_count"])
        self.assertEqual(1, summary["architecture"]["config"]["residual_blocks"])
        self.assertEqual("training", summary["run_key"])
        self.assertEqual("training", records["records"][0]["run_id"])
        self.assertEqual(4, records["records"][0]["step"])

    def test_serves_replay_details_and_prometheus_instrumentation(self) -> None:
        listing, _ = self._get_json("/api/replays?run_id=training")
        replay_id = quote(listing["replays"][0]["id"], safe="")
        replay, _ = self._get_json(f"/api/replays/{replay_id}")
        metrics, _ = self._get("/metrics")

        self.assertEqual([0, 1, 5], replay["actions"])
        metrics_text = metrics.decode("utf-8")
        self.assertIn("kb_pente_monitor_http_requests_total", metrics_text)
        self.assertIn('route="replays",status="200"', metrics_text)

    def test_rejects_traversal_and_invalid_query_bounds(self) -> None:
        traversal = quote("../training.jsonl", safe="")
        self._assert_http_status(
            f"/api/runs/{traversal}/summary",
            HTTPStatus.BAD_REQUEST,
        )
        self._assert_http_status(
            "/api/runs/training.jsonl/records?limit=5000",
            HTTPStatus.BAD_REQUEST,
        )
        self._assert_http_status("/missing", HTTPStatus.NOT_FOUND)

    def _get(self, path: str) -> tuple[bytes, HTTPMessage]:
        request = Request(self.base_url + path, headers={"Connection": "close"})
        with urlopen(request, timeout=2) as response:
            return response.read(), response.headers

    def _get_json(self, path: str) -> tuple[dict[str, Any], HTTPMessage]:
        body, headers = self._get(path)
        return json.loads(body), headers

    def _assert_http_status(self, path: str, status: HTTPStatus) -> None:
        with self.assertRaises(HTTPError) as context:
            self._get(path)
        self.assertEqual(status, context.exception.code)


if __name__ == "__main__":
    unittest.main()
