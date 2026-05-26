from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, timeout_seconds: float) -> None:
    import urllib.request

    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/api/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if payload.get("ok"):
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError(f"dashboard did not become healthy at {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the perf visualizer dashboard with Playwright.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--effectiveness-exe", default="build/effectiveness_runner.exe")
    parser.add_argument("--screenshot-dir", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    effectiveness_exe = (repo_root / args.effectiveness_exe).resolve()
    if not effectiveness_exe.exists():
        raise SystemExit(f"missing effectiveness runner: {effectiveness_exe}")

    port = _free_port()
    screenshot_root = Path(args.screenshot_dir).resolve() if args.screenshot_dir else Path(tempfile.mkdtemp())
    screenshot_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        history_file = Path(tmp) / "dashboard-history.ndjson"
        server = subprocess.Popen(
            [
                sys.executable,
                str(repo_root / "tools" / "perf_visualizer" / "run_dashboard.py"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--repo-root",
                str(repo_root),
                "--effectiveness-exe",
                str(effectiveness_exe),
                "--history-file",
                str(history_file),
                "--run-effectiveness-on-start",
            ],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_for_health(port, 30)
            url = f"http://127.0.0.1:{port}/"
            console_events: list[dict[str, str]] = []
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                page.on(
                    "console",
                    lambda msg: console_events.append({"type": msg.type, "text": msg.text})
                    if msg.type in {"error", "warning"}
                    else None,
                )
                page.on("pageerror", lambda exc: console_events.append({"type": "pageerror", "text": str(exc)}))
                page.goto(url, wait_until="networkidle")

                expect(page.get_by_role("heading", name="Performance Console")).to_be_visible()
                expect(page.get_by_role("heading", name="Baseline Comparison")).to_be_visible()
                expect(page.get_by_role("heading", name="Retrieval Diagnostics")).to_be_visible()
                expect(page.get_by_role("heading", name="Slowest Queries")).to_be_visible()
                expect(page.locator("#languageRows .insight-row").first).to_be_visible()
                expect(page.locator("#diagnosticRows .insight-row").first).to_be_visible()
                expect(page.locator("#nearMissRows")).to_contain_text("No near misses")

                before = page.evaluate(
                    "() => Number(((document.querySelector('#sampleCount')?.textContent || '0').match(/\\d+/) || ['0'])[0])"
                )
                page.get_by_role("button", name="Run effectiveness").click()
                page.wait_for_function(
                    """before => {
                      const count = Number(((document.querySelector('#sampleCount')?.textContent || '0').match(/\\d+/) || ['0'])[0]);
                      const button = document.querySelector('#runEffectiveness');
                      return count > before && button && !button.disabled;
                    }""",
                    arg=before,
                    timeout=30000,
                )
                expect(page.locator("#baselineRows .insight-row").first).to_be_visible()

                canvas_state = page.evaluate(
                    """() => [...document.querySelectorAll('canvas')].map((canvas) => {
                      const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
                      let painted = 0;
                      for (let i = 3; i < data.length; i += 64) if (data[i] !== 0) painted += 1;
                      return { id: canvas.id, painted };
                    })"""
                )
                if len(canvas_state) != 4 or any(item["painted"] <= 0 for item in canvas_state):
                    raise AssertionError(f"expected four nonblank canvases, got {canvas_state}")

                page.screenshot(path=str(screenshot_root / "dashboard-desktop.png"), full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                page.wait_for_timeout(300)
                overflow = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
                if overflow > 2:
                    raise AssertionError(f"mobile layout has horizontal overflow: {overflow}")
                page.screenshot(path=str(screenshot_root / "dashboard-mobile.png"), full_page=True)
                browser.close()

            relevant_console = [event for event in console_events if "favicon" not in event["text"].lower()]
            if relevant_console:
                raise AssertionError(f"browser console errors: {relevant_console}")
            print(json.dumps({"ok": True, "url": url, "screenshots": str(screenshot_root)}, indent=2))
            return 0
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
