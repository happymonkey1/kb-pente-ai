from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
import time
from typing import cast
from urllib.parse import parse_qs, unquote, urlsplit

from src.monitoring.store import (
    ArtifactIdentifierError,
    ArtifactNotFoundError,
    ReplayStore,
    TelemetryStore,
)


logger = logging.getLogger(__name__)
STATIC_ROOT = Path(__file__).with_name("static")
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


@dataclass(slots=True)
class ServerMetrics:
    started_at_unix: float = field(default_factory=time.time)
    request_counts: Counter[tuple[str, int]] = field(default_factory=Counter)
    request_duration_seconds: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, route: str, status: int, duration_seconds: float) -> None:
        with self._lock:
            self.request_counts[(route, status)] += 1
            self.request_duration_seconds[route] = (
                self.request_duration_seconds.get(route, 0.0) + duration_seconds
            )

    def render(self, telemetry_cache_size: int, replay_cache_size: int) -> str:
        with self._lock:
            counts = sorted(self.request_counts.items())
            durations = sorted(self.request_duration_seconds.items())
        lines = [
            "# HELP kb_pente_monitor_uptime_seconds Monitor process uptime.",
            "# TYPE kb_pente_monitor_uptime_seconds gauge",
            f"kb_pente_monitor_uptime_seconds {time.time() - self.started_at_unix:.6f}",
            "# HELP kb_pente_monitor_http_requests_total HTTP requests handled.",
            "# TYPE kb_pente_monitor_http_requests_total counter",
        ]
        lines.extend(
            "kb_pente_monitor_http_requests_total"
            f'{{route="{_prometheus_escape(route)}",status="{status}"}} {count}'
            for (route, status), count in counts
        )
        lines.extend(
            (
                "# HELP kb_pente_monitor_http_request_duration_seconds_total "
                "Cumulative request handling time.",
                "# TYPE kb_pente_monitor_http_request_duration_seconds_total counter",
            )
        )
        lines.extend(
            "kb_pente_monitor_http_request_duration_seconds_total"
            f'{{route="{_prometheus_escape(route)}"}} {duration:.9f}'
            for route, duration in durations
        )
        lines.extend(
            (
                "# HELP kb_pente_monitor_cached_files Files held in parsed artifact caches.",
                "# TYPE kb_pente_monitor_cached_files gauge",
                f'kb_pente_monitor_cached_files{{kind="telemetry"}} {telemetry_cache_size}',
                f'kb_pente_monitor_cached_files{{kind="replay"}} {replay_cache_size}',
            )
        )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class MonitoringApplication:
    telemetry: TelemetryStore
    replays: ReplayStore
    metrics: ServerMetrics


class MonitoringHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: MonitoringApplication,
    ) -> None:
        self.application = application
        super().__init__(server_address, MonitoringRequestHandler)


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class MonitoringRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def application(self) -> MonitoringApplication:
        server = cast(MonitoringHTTPServer, self.server)
        return server.application

    def do_GET(self) -> None:
        started = time.perf_counter()
        route = "unknown"
        status: int = int(HTTPStatus.INTERNAL_SERVER_ERROR)
        try:
            route, status = self._dispatch_get()
        except ApiError as error:
            status = int(error.status)
            self._send_json(status, {"error": error.message})
        except ArtifactIdentifierError as error:
            status = int(HTTPStatus.BAD_REQUEST)
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except ArtifactNotFoundError as error:
            status = int(HTTPStatus.NOT_FOUND)
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
        except ValueError as error:
            status = int(HTTPStatus.BAD_REQUEST)
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            route = "disconnected"
            status = 499
        except Exception:
            logger.exception("Unhandled monitoring request failure")
            status = int(HTTPStatus.INTERNAL_SERVER_ERROR)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "Internal monitoring server error"},
            )
        finally:
            self.application.metrics.observe(
                route,
                int(status),
                time.perf_counter() - started,
            )

    def _dispatch_get(self) -> tuple[str, HTTPStatus]:
        request = urlsplit(self.path)
        path = request.path
        query = parse_qs(request.query, keep_blank_values=True)

        if path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "telemetry_root_available": self.application.telemetry.root.is_dir(),
                    "replay_root_available": self.application.replays.root.is_dir(),
                },
            )
            return "health", HTTPStatus.OK

        if path == "/api/runs":
            self._send_json(
                HTTPStatus.OK,
                {"runs": self.application.telemetry.list_runs()},
            )
            return "runs", HTTPStatus.OK

        if path.startswith("/api/runs/") and path.endswith("/summary"):
            run_id = _path_identifier(path, "/api/runs/", "/summary")
            self._send_json(
                HTTPStatus.OK,
                self.application.telemetry.summary(run_id),
            )
            return "run_summary", HTTPStatus.OK

        if path.startswith("/api/runs/") and path.endswith("/records"):
            run_id = _path_identifier(path, "/api/runs/", "/records")
            after = _query_integer(query, "after", default=0, minimum=0, maximum=100_000)
            limit = _query_integer(query, "limit", default=500, minimum=1, maximum=2_000)
            self._send_json(
                HTTPStatus.OK,
                self.application.telemetry.records(
                    run_id,
                    after=after,
                    limit=limit,
                ),
            )
            return "run_records", HTTPStatus.OK

        if path == "/api/replays":
            run_ids = query.get("run_id")
            replay_run_id = run_ids[0] if run_ids else None
            if replay_run_id == "":
                replay_run_id = None
            self._send_json(
                HTTPStatus.OK,
                self.application.replays.list_replays(run_id=replay_run_id),
            )
            return "replays", HTTPStatus.OK

        if path.startswith("/api/replays/"):
            replay_id = unquote(path.removeprefix("/api/replays/"))
            self._send_json(
                HTTPStatus.OK,
                self.application.replays.replay(replay_id),
            )
            return "replay", HTTPStatus.OK

        if path == "/metrics":
            metrics_body = self.application.metrics.render(
                self.application.telemetry.cache_size,
                self.application.replays.cache_size,
            )
            self._send_bytes(
                HTTPStatus.OK,
                metrics_body.encode("utf-8"),
                "text/plain; version=0.0.4; charset=utf-8",
                cache_control="no-store",
            )
            return "metrics", HTTPStatus.OK

        static_files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/pico.min.css": ("pico.min.css", "text/css; charset=utf-8"),
            "/app.css": ("app.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        static_file = static_files.get(path)
        if static_file is not None:
            name, content_type = static_file
            try:
                static_body = (STATIC_ROOT / name).read_bytes()
            except OSError as error:
                raise ApiError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "Dashboard asset is unavailable",
                ) from error
            self._send_bytes(
                HTTPStatus.OK,
                static_body,
                content_type,
                cache_control="no-cache",
            )
            return "static", HTTPStatus.OK

        raise ApiError(HTTPStatus.NOT_FOUND, "Route not found")

    def _send_json(self, status: HTTPStatus | int, payload: object) -> None:
        body = json.dumps(
            payload,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(
            status,
            body,
            "application/json; charset=utf-8",
            cache_control="no-store",
        )

    def _send_bytes(
        self,
        status: HTTPStatus | int,
        body: bytes,
        content_type: str,
        *,
        cache_control: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("monitoring request: " + format, *args)


def build_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    metrics_root: str | Path = "metrics",
    replay_root: str | Path = "replays",
    activity_window_seconds: float = 120.0,
    max_file_bytes: int = 64 * 1024 * 1024,
) -> MonitoringHTTPServer:
    if not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    application = MonitoringApplication(
        telemetry=TelemetryStore(
            metrics_root,
            activity_window_seconds=activity_window_seconds,
            max_file_bytes=max_file_bytes,
        ),
        replays=ReplayStore(replay_root, max_file_bytes=max_file_bytes),
        metrics=ServerMetrics(),
    )
    return MonitoringHTTPServer((host, port), application)


def _path_identifier(path: str, prefix: str, suffix: str) -> str:
    encoded = path[len(prefix) : -len(suffix)]
    identifier = unquote(encoded).strip("/")
    if not identifier:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Run identifier is required")
    return identifier


def _query_integer(
    query: dict[str, list[str]],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_values = query.get(name)
    if not raw_values:
        return default
    try:
        value = int(raw_values[0])
    except ValueError as error:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"{name} must be an integer") from error
    if not minimum <= value <= maximum:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"{name} must be between {minimum} and {maximum}",
        )
    return value


def _prometheus_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
