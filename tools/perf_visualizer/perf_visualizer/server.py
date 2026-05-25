from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .collector import collect_effectiveness_snapshot, collect_proxy_snapshot
from .metrics import append_snapshot, load_history


STATIC_ROOT = Path(__file__).resolve().parent / "static"


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


class PerfState:
    def __init__(
        self,
        *,
        history_file: Path,
        effectiveness_exe: Path | None,
        repo_root: Path,
        proxy_stats_url: str | None,
        proxy_token: str | None,
        command_timeout_seconds: float,
    ) -> None:
        self.history_file = history_file
        self.effectiveness_exe = effectiveness_exe
        self.repo_root = repo_root
        self.proxy_stats_url = proxy_stats_url
        self.proxy_token = proxy_token
        self.command_timeout_seconds = command_timeout_seconds
        self._lock = threading.Lock()
        self._history = load_history(history_file)
        self._last_proxy_raw: dict[str, Any] | None = None
        self.last_error = ""

    def history(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._history)
        if limit is not None and limit >= 0:
            return rows[-limit:]
        return rows

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def append(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._history.append(snapshot)
            append_snapshot(self.history_file, snapshot)

    def run_effectiveness_once(self) -> dict[str, Any]:
        if self.effectiveness_exe is None:
            raise RuntimeError("no effectiveness runner configured")
        snapshot = collect_effectiveness_snapshot(
            self.effectiveness_exe,
            cwd=self.repo_root,
            timeout_seconds=self.command_timeout_seconds,
        )
        self.append(snapshot)
        return snapshot

    def fetch_proxy_once(self) -> dict[str, Any]:
        if not self.proxy_stats_url:
            raise RuntimeError("no proxy stats URL configured")
        previous = self._last_proxy_raw
        snapshot, raw = collect_proxy_snapshot(
            self.proxy_stats_url,
            previous=previous,
            proxy_token=self.proxy_token,
        )
        self._last_proxy_raw = raw
        self.append(snapshot)
        return snapshot


def make_handler(state: PerfState):
    class PerfHandler(BaseHTTPRequestHandler):
        server_version = "LLMPreprocessorPerfVisualizer/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            state.last_error = message
            self._json({"error": message}, status)

        def _serve_static(self, path: str) -> None:
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            target = (STATIC_ROOT / relative).resolve()
            root = STATIC_ROOT.resolve()
            if root not in target.parents and target != root:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            if not target.exists() or not target.is_file():
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            body = target.read_bytes()
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                self._json({"ok": True, "last_error": state.last_error})
                return
            if parsed.path == "/api/config":
                self._json({
                    "effectiveness_configured": state.effectiveness_exe is not None,
                    "proxy_configured": bool(state.proxy_stats_url),
                    "history_file": str(state.history_file),
                })
                return
            if parsed.path == "/api/history":
                query = parse_qs(parsed.query)
                limit = int(query.get("limit", ["500"])[0])
                self._json({"snapshots": state.history(limit=limit)})
                return
            if parsed.path == "/api/latest":
                self._json({"snapshot": state.latest()})
                return
            if parsed.path == "/api/proxy-stats":
                try:
                    self._json({"snapshot": state.fetch_proxy_once()})
                except Exception as exc:
                    self._error(HTTPStatus.BAD_GATEWAY, str(exc))
                return
            self._serve_static(parsed.path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/run-effectiveness":
                try:
                    self._json({"snapshot": state.run_effectiveness_once()})
                except Exception as exc:
                    self._error(HTTPStatus.BAD_GATEWAY, str(exc))
                return
            self._error(HTTPStatus.NOT_FOUND, "not found")

    return PerfHandler


def _collector_loop(
    *,
    name: str,
    interval_seconds: float,
    stop_event: threading.Event,
    action,
    state: PerfState,
) -> None:
    while not stop_event.is_set():
        try:
            action()
            state.last_error = ""
        except Exception as exc:
            state.last_error = f"{name}: {exc}"
        stop_event.wait(interval_seconds)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the llm-preprocessor performance dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard bind host.")
    parser.add_argument("--port", type=int, default=8787, help="Dashboard bind port.")
    parser.add_argument("--repo-root", default=".", help="Repository root used as runner cwd.")
    parser.add_argument("--history-file", default="benchmarks/results/perf_visualizer.ndjson")
    parser.add_argument("--effectiveness-exe", default="build/effectiveness_runner.exe")
    parser.add_argument("--no-effectiveness", action="store_true", help="Disable effectiveness runner integration.")
    parser.add_argument("--run-effectiveness-on-start", action="store_true")
    parser.add_argument("--effectiveness-interval-sec", type=float, default=0.0)
    parser.add_argument("--proxy-stats-url", default="")
    parser.add_argument("--proxy-token", default="")
    parser.add_argument("--proxy-interval-sec", type=float, default=0.0)
    parser.add_argument("--command-timeout-sec", type=float, default=120.0)
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser.")
    parser.add_argument(
        "--allow-unsafe-remote-dashboard",
        action="store_true",
        help="Allow binding the dashboard to a non-loopback host.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not _is_loopback(args.host) and not args.allow_unsafe_remote_dashboard:
        raise SystemExit(
            "Refusing to bind dashboard to a non-loopback host without "
            "--allow-unsafe-remote-dashboard."
        )

    repo_root = Path(args.repo_root).resolve()
    effectiveness_exe = None
    if not args.no_effectiveness:
        candidate = (repo_root / args.effectiveness_exe).resolve()
        if candidate.exists():
            effectiveness_exe = candidate

    state = PerfState(
        history_file=(repo_root / args.history_file).resolve(),
        effectiveness_exe=effectiveness_exe,
        repo_root=repo_root,
        proxy_stats_url=args.proxy_stats_url or None,
        proxy_token=args.proxy_token or None,
        command_timeout_seconds=args.command_timeout_sec,
    )

    if args.run_effectiveness_on_start and effectiveness_exe is not None:
        try:
            state.run_effectiveness_once()
        except Exception as exc:
            state.last_error = f"startup effectiveness run: {exc}"

    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    if args.effectiveness_interval_sec > 0 and effectiveness_exe is not None:
        threads.append(threading.Thread(
            target=_collector_loop,
            kwargs={
                "name": "effectiveness",
                "interval_seconds": args.effectiveness_interval_sec,
                "stop_event": stop_event,
                "action": state.run_effectiveness_once,
                "state": state,
            },
            daemon=True,
        ))
    if args.proxy_interval_sec > 0 and args.proxy_stats_url:
        threads.append(threading.Thread(
            target=_collector_loop,
            kwargs={
                "name": "proxy",
                "interval_seconds": args.proxy_interval_sec,
                "stop_event": stop_event,
                "action": state.fetch_proxy_once,
                "state": state,
            },
            daemon=True,
        ))
    for thread in threads:
        thread.start()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}/"
    print(f"Performance dashboard: {url}")
    print(f"History file: {state.history_file}")
    if args.open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        httpd.server_close()


if __name__ == "__main__":
    main()
