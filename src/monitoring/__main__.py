from __future__ import annotations

import argparse
import ipaddress
import logging

from src.monitoring.server import build_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the local kb-pente-ai telemetry dashboard",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--metrics-root", default="metrics")
    parser.add_argument("--replay-root", default="replays")
    parser.add_argument("--activity-seconds", type=float, default=120.0)
    parser.add_argument("--max-file-mib", type=int, default=64)
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow binding to a non-loopback address without authentication",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.allow_remote and not _is_loopback(args.host):
        raise SystemExit(
            "Refusing a non-loopback bind without --allow-remote; "
            "the monitoring server has no authentication"
        )
    if args.max_file_mib < 1:
        raise SystemExit("--max-file-mib must be positive")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    server = build_server(
        host=args.host,
        port=args.port,
        metrics_root=args.metrics_root,
        replay_root=args.replay_root,
        activity_window_seconds=args.activity_seconds,
        max_file_bytes=args.max_file_mib * 1024 * 1024,
    )
    print(
        f"kb-pente-ai monitor listening on http://{args.host}:{server.server_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
