# Performance Visualizer

Standalone local dashboard for watching `llm-preprocessor` performance in real
time. Runtime serving is intentionally outside the C++ build and uses only the
Python standard library plus browser-native HTML/CSS/JavaScript. Browser smoke
testing is optional and uses Python Playwright when it is installed.

## What It Shows

- Context packing: included/omitted/deduped chunks and budget usage.
- Memory reuse: prompt-cache and embedding-cache speedup.
- Retrieval quality: BM25 and multi-language fixture top-3 accuracy.
- Retrieval diagnostics: per-language accuracy, expected rank, slowest fixture
  queries, near misses, and graph-expansion counts.
- Baseline comparison: latest effectiveness sample compared with the first
  matching sample in the current history file.
- LLM efficiency: prompt rewriting token reduction, diff wire savings, live
  cache hit rate, upstream avoidance, and token-savings counters from `/stats`.
- Streaming health: live stream cancellation and error counters when connected
  to a running proxy.

## Start With Offline Benchmarks

Build the project first so `effectiveness_runner.exe` exists:

```powershell
cmake --build build
```

Start the dashboard and run one effectiveness sample at startup:

```powershell
python tools\perf_visualizer\run_dashboard.py `
  --effectiveness-exe build\effectiveness_runner.exe `
  --history-file benchmarks\results\perf_visualizer.ndjson `
  --run-effectiveness-on-start `
  --open
```

Then open:

```text
http://127.0.0.1:8787/
```

Click `Run effectiveness` whenever you want a fresh sample. The dashboard polls
its local API once per second, so new samples appear without a page refresh.

## Add Live Proxy Stats

Start the proxy in another terminal:

```powershell
.\build\preprocessor_app.exe --serve config.example.json
```

Then start the dashboard with `/stats` polling. Use the token configured in
`config.example.json` or your own local config.

```powershell
python tools\perf_visualizer\run_dashboard.py `
  --effectiveness-exe build\effectiveness_runner.exe `
  --proxy-stats-url http://127.0.0.1:8088/stats `
  --proxy-token replace-with-local-proxy-token `
  --proxy-interval-sec 1 `
  --history-file benchmarks\results\perf_visualizer.ndjson `
  --open
```

The dashboard binds to `127.0.0.1` by default. It refuses non-loopback hosts
unless `--allow-unsafe-remote-dashboard` is supplied.

## Useful API Endpoints

```text
GET  /api/health
GET  /api/config
GET  /api/latest
GET  /api/agent-summary
GET  /api/history?limit=500
GET  /api/proxy-stats
POST /api/run-effectiveness
```

`/api/agent-summary` returns a compact status, key metrics, retrieval breakdown,
and recommendations object for coding agents or automation that need diagnostics
without scraping the HTML dashboard. History is written as NDJSON to the
configured `--history-file`. The default path lives under `benchmarks/results/`,
which is ignored by git.

## Test The Visualizer

```powershell
python -m unittest discover tools\perf_visualizer\tests
```

If Python Playwright is installed, run the rendered dashboard smoke test:

```powershell
python tools\perf_visualizer\tests\validate_dashboard.py `
  --effectiveness-exe build\effectiveness_runner.exe `
  --screenshot-dir benchmarks\results\dashboard-validation
```

The Playwright smoke starts a temporary dashboard, runs an effectiveness sample,
checks baseline and retrieval diagnostic panels, clicks `Run effectiveness`,
verifies four nonblank charts, checks mobile overflow, and writes screenshots to
the provided directory.

## Test The Whole Program

From the repository root:

```powershell
python tools\release_smoke.py
```

Use `python tools\release_smoke.py --skip-playwright` when Python Playwright is
not installed on the machine running the release gate.

For a local dashboard smoke test:

```powershell
python tools\perf_visualizer\run_dashboard.py `
  --effectiveness-exe build\effectiveness_runner.exe `
  --run-effectiveness-on-start
```

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8787/api/health
Invoke-RestMethod http://127.0.0.1:8787/api/history?limit=1
```
