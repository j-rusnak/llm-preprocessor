from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .metrics import normalize_effectiveness_report, normalize_proxy_stats


def run_effectiveness_report(
    executable: str | Path,
    *,
    cwd: str | Path | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    command = [str(executable)]
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "effectiveness runner failed with exit "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("effectiveness runner did not write JSON to stdout") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("effectiveness runner JSON must be an object")
    return parsed


def collect_effectiveness_snapshot(
    executable: str | Path,
    *,
    cwd: str | Path | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    return normalize_effectiveness_report(
        run_effectiveness_report(executable, cwd=cwd, timeout_seconds=timeout_seconds)
    )


def fetch_proxy_stats(
    stats_url: str,
    *,
    proxy_token: str | None = None,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if proxy_token:
        value = proxy_token if proxy_token.startswith("Bearer ") else f"Bearer {proxy_token}"
        headers["X-Preprocessor-Authorization"] = value
    request = urllib.request.Request(stats_url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to read proxy stats from {stats_url}: {exc}") from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError("proxy stats JSON must be an object")
    return parsed


def collect_proxy_snapshot(
    stats_url: str,
    *,
    previous: dict[str, Any] | None = None,
    proxy_token: str | None = None,
    timeout_seconds: float = 2.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = fetch_proxy_stats(
        stats_url,
        proxy_token=proxy_token,
        timeout_seconds=timeout_seconds,
    )
    return normalize_proxy_stats(raw, previous=previous), raw
