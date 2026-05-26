# Production Readiness Next Cycles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the newly verified diagnostics, dashboard, and release-smoke work into a tighter production-ready service with stronger release packaging, retrieval evidence, cancellation behavior, and deployment defaults.

**Architecture:** Keep changes in small vertical slices that each add a measurable release capability. Use `effectiveness_runner`, focused gtests, visualizer tests, and `tools/release_smoke.py` as the proving loop. Avoid broad refactors; touch only the module owning the behavior being improved.

**Tech Stack:** C++17, CMake, GoogleTest/CTest, Python standard-library tooling, Python unittest, Playwright dashboard validation, PowerShell-compatible release commands.

---

## File Structure

- Existing verified slice to commit:
  - `.github/workflows/ci.yml`
  - `src/hybrid_retriever.cpp`
  - `benchmarks/effectiveness_runner.cpp`
  - `tests/test_effectiveness.cpp`
  - `tests/fixtures/retrieval/cpp/login_controller.cpp`
  - `tests/fixtures/retrieval/security/signature_verifier.cpp`
  - `tools/release_smoke.py`
  - `tools/perf_visualizer/**`
  - `README.md`
  - `docs/TESTING.md`
  - `docs/RELEASE.md`
- New package audit slice:
  - Create: `tools/package_audit.py`
  - Create: `tools/tests/test_package_audit.py`
  - Modify: `tools/release_smoke.py`
  - Modify: `.github/workflows/ci.yml`
  - Modify: `docs/RELEASE.md`
- New retrieval corpus slice:
  - Create fixture directories under `tests/fixtures/retrieval/realworld/`
  - Modify: `benchmarks/effectiveness_runner.cpp`
  - Modify: `tests/test_effectiveness.cpp`
  - Modify: `tools/perf_visualizer/perf_visualizer/static/app.js`
- New streaming cancellation slice:
  - Modify: `tests/test_openai_proxy.cpp`
  - Modify only if needed: `src/openai_proxy.cpp`
  - Modify: `benchmarks/effectiveness_runner.cpp`
- New production examples slice:
  - Create: `config.production.example.json`
  - Create: `docs/PRODUCTION_CONFIG.md`
  - Modify: `tests/test_config_loader.cpp`
  - Modify: `README.md`

---

### Task 1: Commit The Current Verified Work

**Files:**
- Stage/commit the current verified working tree.

- [ ] **Step 1: Review current status**

Run:

```powershell
git status --short
git diff --stat
```

Expected: only the diagnostics, visualizer, release-smoke, ranking, docs, and retrieval fixture files from the previous work cycle are modified or untracked.

- [ ] **Step 2: Run release hygiene only**

Run:

```powershell
git ls-files -ci --exclude-standard
```

Expected: no output.

- [ ] **Step 3: Commit the verified slice**

Run:

```powershell
git add .github/workflows/ci.yml README.md benchmarks/effectiveness_runner.cpp docs/RELEASE.md docs/TESTING.md src/hybrid_retriever.cpp tests/test_effectiveness.cpp tests/fixtures/retrieval/cpp/login_controller.cpp tests/fixtures/retrieval/security/signature_verifier.cpp tools/release_smoke.py tools/perf_visualizer
git commit -m "feat: add production diagnostics release gate"
```

Expected: one commit containing only the already verified changes.

---

### Task 2: Add Package Audit Tooling

**Files:**
- Create: `tools/package_audit.py`
- Create: `tools/tests/test_package_audit.py`
- Modify: `tools/release_smoke.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `docs/RELEASE.md`

- [ ] **Step 1: Write failing Python unit tests**

Create `tools/tests/test_package_audit.py`:

```python
import tempfile
import unittest
from pathlib import Path

from tools.package_audit import audit_install_tree


class PackageAuditTests(unittest.TestCase):
    def test_accepts_expected_install_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "lib" / "cmake" / "LLMPreprocessor").mkdir(parents=True)
            (root / "include").mkdir()
            (root / "bin" / "preprocessor_app.exe").write_text("", encoding="utf-8")
            (root / "bin" / "onnxruntime.dll").write_text("", encoding="utf-8")
            (root / "lib" / "cmake" / "LLMPreprocessor" / "LLMPreprocessorConfig.cmake").write_text("", encoding="utf-8")

            result = audit_install_tree(root)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["forbidden"], [])

    def test_rejects_runtime_state_and_missing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "models").mkdir()
            (root / "models" / "model.ort").write_text("", encoding="utf-8")
            (root / "prompt_cache.db").write_text("", encoding="utf-8")

            result = audit_install_tree(root)

            self.assertEqual(result["status"], "fail")
            self.assertIn("models/model.ort", result["forbidden"])
            self.assertIn("prompt_cache.db", result["forbidden"])
            self.assertTrue(any("LLMPreprocessorConfig.cmake" in item for item in result["missing"]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m unittest tools\tests\test_package_audit.py
```

Expected: import failure for `tools.package_audit`.

- [ ] **Step 3: Implement package audit**

Create `tools/package_audit.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".onnx",
    ".ort",
    ".zip",
    ".tgz",
}

REQUIRED_FRAGMENTS = [
    "lib/cmake/LLMPreprocessor/LLMPreprocessorConfig.cmake",
]


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def audit_install_tree(root: str | Path) -> dict[str, object]:
    root = Path(root)
    forbidden: list[str] = []
    missing: list[str] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = _rel(path, root)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(rel)

    for fragment in REQUIRED_FRAGMENTS:
        if not (root / fragment).exists():
            missing.append(fragment)

    return {
        "status": "ok" if not forbidden and not missing else "fail",
        "root": str(root),
        "forbidden": sorted(forbidden),
        "missing": sorted(missing),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit llm-preprocessor install/package output.")
    parser.add_argument("install_root")
    args = parser.parse_args()
    result = audit_install_tree(args.install_root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run package audit tests**

Run:

```powershell
python -m unittest tools\tests\test_package_audit.py
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 5: Wire audit into release smoke**

Modify `tools/release_smoke.py` after the install step to run:

```python
_run(
    "Package audit",
    [sys.executable, "tools/package_audit.py", str(install_prefix)],
    cwd=repo_root,
    dry_run=args.dry_run,
    timeout_seconds=args.command_timeout_sec,
)
```

- [ ] **Step 6: Verify dry run and focused package audit**

Run:

```powershell
python tools\release_smoke.py --dry-run --skip-playwright --skip-install
python tools\package_audit.py build\install-check
```

Expected: dry run exits 0; package audit exits 0 against the current install tree.

- [ ] **Step 7: Commit package audit slice**

Run:

```powershell
git add tools/package_audit.py tools/tests/test_package_audit.py tools/release_smoke.py .github/workflows/ci.yml docs/RELEASE.md
git commit -m "test: audit release package contents"
```

---

### Task 3: Broaden Retrieval Fixtures With Real-World Snapshots

**Files:**
- Create: `tests/fixtures/retrieval/realworld/python/service_auth.py`
- Create: `tests/fixtures/retrieval/realworld/typescript/agent_context_store.ts`
- Create: `tests/fixtures/retrieval/realworld/cpp/stream_forwarder.cpp`
- Modify: `benchmarks/effectiveness_runner.cpp`
- Modify: `tests/test_effectiveness.cpp`
- Modify: `tools/perf_visualizer/perf_visualizer/static/app.js`

- [ ] **Step 1: Add failing fixture expectations**

Modify the fixture case list in `tests/test_effectiveness.cpp` to include:

```cpp
{
    "python auth dependency override verify bearer token request state",
    "realworld/python/service_auth.py"
},
{
    "typescript agent context store retrieval window persistence budget",
    "realworld/typescript/agent_context_store.ts"
},
{
    "cpp stream forwarder cancel upstream on client disconnect idle timeout",
    "realworld/cpp/stream_forwarder.cpp"
},
```

- [ ] **Step 2: Run focused test to verify it fails**

Run:

```powershell
.\build\preprocessor_tests.exe --gtest_filter=Effectiveness_Retrieval.FixtureQueriesHitExpectedLanguageFileTop3
```

Expected: failure because the three fixture files do not exist or do not retrieve.

- [ ] **Step 3: Add real-world-shaped fixtures**

Create `tests/fixtures/retrieval/realworld/python/service_auth.py`:

```python
from fastapi import Depends, HTTPException, Request


def verify_bearer_token(request: Request) -> str:
    token = request.headers.get("x-preprocessor-authorization", "")
    if not token.startswith("Bearer "):
        raise HTTPException(status_code=401)
    return token.removeprefix("Bearer ").strip()


def dependency_override_for_tests(request: Request, token: str = Depends(verify_bearer_token)) -> dict[str, str]:
    request.state.local_proxy_token = token
    return {"token": token}
```

Create `tests/fixtures/retrieval/realworld/typescript/agent_context_store.ts`:

```typescript
export type ContextWindow = {
  retrievalQuery: string;
  includedChunkIds: string[];
  omittedChunkIds: string[];
  budgetChars: number;
};

export class AgentContextStore {
  private windows: ContextWindow[] = [];

  recordWindow(window: ContextWindow) {
    this.windows.push(window);
    if (this.windows.length > 50) this.windows.shift();
  }

  latestForRetrieval(query: string) {
    return [...this.windows].reverse().find((window) => window.retrievalQuery === query);
  }
}
```

Create `tests/fixtures/retrieval/realworld/cpp/stream_forwarder.cpp`:

```cpp
#include <atomic>
#include <chrono>

struct StreamForwarder {
    std::atomic_bool client_disconnected{false};

    bool should_cancel_upstream(std::chrono::steady_clock::time_point last_event,
                                std::chrono::steady_clock::time_point now,
                                std::chrono::seconds idle_timeout) const {
        return client_disconnected.load() || now - last_event > idle_timeout;
    }
};
```

- [ ] **Step 4: Add effectiveness categories**

Add matching cases to `benchmarks/effectiveness_runner.cpp` with categories:

```cpp
{"python", "security", "python auth dependency override verify bearer token request state", "realworld/python/service_auth.py"},
{"typescript", "context", "typescript agent context store retrieval window persistence budget", "realworld/typescript/agent_context_store.ts"},
{"cpp", "streaming", "cpp stream forwarder cancel upstream on client disconnect idle timeout", "realworld/cpp/stream_forwarder.cpp"},
```

- [ ] **Step 5: Run focused retrieval tests and effectiveness runner**

Run:

```powershell
.\build\preprocessor_tests.exe --gtest_filter=Effectiveness_Retrieval.FixtureQueriesHitExpectedLanguageFileTop3
.\build\effectiveness_runner.exe > benchmarks\results\effectiveness-retrieval-expanded.json
```

Expected: focused test passes; effectiveness JSON reports the added queries and no near misses.

- [ ] **Step 6: Commit retrieval fixture slice**

Run:

```powershell
git add tests/fixtures/retrieval/realworld tests/test_effectiveness.cpp benchmarks/effectiveness_runner.cpp tools/perf_visualizer/perf_visualizer/static/app.js
git commit -m "test: broaden retrieval fixture corpus"
```

---

### Task 4: Deepen Client Disconnect Cancellation Coverage

**Files:**
- Modify: `tests/test_openai_proxy.cpp`
- Modify if the test exposes a bug: `src/openai_proxy.cpp`
- Modify: `benchmarks/effectiveness_runner.cpp`

- [ ] **Step 1: Add a failing long-stream disconnect test**

In `tests/test_openai_proxy.cpp`, add a test named:

```cpp
TEST(OpenAIProxy, LongStreamingClientDisconnectCancelsUpstreamWithoutCacheWrite) {
    // Use the existing proxy test server helpers in this file.
    // Arrange an upstream stream that emits many SSE chunks slowly.
    // Connect a client, read the first data chunk, close the socket.
    // Assert upstream cancellation counter increments.
    // Assert prompt cache remains empty for that request key.
    // Assert no synthetic SSE error is written after disconnect.
}
```

Use the existing socket/client helpers already present in `tests/test_openai_proxy.cpp` rather than adding a new networking abstraction.

- [ ] **Step 2: Run the test and capture failure**

Run:

```powershell
.\build\preprocessor_tests.exe --gtest_filter=OpenAIProxy.LongStreamingClientDisconnectCancelsUpstreamWithoutCacheWrite
```

Expected: fail if cancellation is incomplete, or pass if existing behavior already covers it.

- [ ] **Step 3: Implement only if required**

If the test fails because upstream streaming continues after client disconnect, patch `src/openai_proxy.cpp` in the streaming forwarding loop so the write callback observes the client sink state and stops forwarding immediately.

The change should preserve:

```cpp
// Existing behavior to preserve:
// - SSE framing remains unchanged for healthy streams.
// - PromptCache is bypassed for streaming requests.
// - Client disconnect increments stream cancellation metrics.
// - No synthetic error trailer is emitted after client disconnect.
```

- [ ] **Step 4: Add benchmark counter**

Extend `benchmarks/effectiveness_runner.cpp` to include a `streaming_cancellation` object:

```json
{
  "long_stream_cancelled": true,
  "cache_write_after_disconnect": false
}
```

- [ ] **Step 5: Verify focused and full proxy tests**

Run:

```powershell
.\build\preprocessor_tests.exe --gtest_filter=OpenAIProxy.*Streaming*:OpenAIProxy.*Disconnect*
ctest --test-dir build --output-on-failure
```

Expected: focused streaming tests pass; full CTest passes.

- [ ] **Step 6: Commit cancellation slice**

Run:

```powershell
git add tests/test_openai_proxy.cpp src/openai_proxy.cpp benchmarks/effectiveness_runner.cpp
git commit -m "test: harden streaming disconnect cancellation"
```

---

### Task 5: Add Production Config Example And Health Guidance

**Files:**
- Create: `config.production.example.json`
- Create: `docs/PRODUCTION_CONFIG.md`
- Modify: `tests/test_config_loader.cpp`
- Modify: `README.md`

- [ ] **Step 1: Write config loader regression test**

Add to `tests/test_config_loader.cpp`:

```cpp
TEST(ConfigLoaderTest, LoadsProductionExampleConfig) {
    auto cfg = preprocessor::ConfigLoader::load("config.production.example.json");
    EXPECT_EQ(cfg.proxy_host, "127.0.0.1");
    EXPECT_FALSE(cfg.proxy_auth_bearer_tokens.empty());
    EXPECT_GT(cfg.proxy_max_request_bytes, 0u);
    EXPECT_FALSE(cfg.allow_unsafe_remote_proxy);
    EXPECT_FALSE(cfg.upstream_url.empty());
    EXPECT_TRUE(cfg.graph_expansion_enabled);
    EXPECT_TRUE(cfg.structural_fast_path_enabled);
}
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
.\build\preprocessor_tests.exe --gtest_filter=ConfigLoaderTest.LoadsProductionExampleConfig
```

Expected: failure because `config.production.example.json` does not exist.

- [ ] **Step 3: Add production config example**

Create `config.production.example.json` with loopback-safe defaults:

```json
{
  "model_path": "models/model.onnx",
  "vocab_path": "models/vocab.txt",
  "repo_root": ".",
  "proxy_host": "127.0.0.1",
  "proxy_port": 8088,
  "proxy_auth_bearer_tokens": ["replace-with-strong-local-token"],
  "proxy_forward_client_authorization": false,
  "proxy_rate_limit_tokens_per_second": 2.0,
  "proxy_rate_limit_burst": 10.0,
  "proxy_max_request_bytes": 8388608,
  "allow_unsafe_remote_proxy": false,
  "upstream_url": "https://api.openai.com/v1/chat/completions",
  "upstream_api_key": "",
  "upstream_timeout_seconds": 60,
  "upstream_connect_timeout_seconds": 10,
  "upstream_max_response_bytes": 8388608,
  "stream_idle_timeout_seconds": 30,
  "retrieval_k": 6,
  "max_context_chars": 8000,
  "prompt_optimizer_enabled": true,
  "include_project_card": true,
  "symbol_graph_enabled": true,
  "graph_expansion_enabled": true,
  "structural_fast_path_enabled": true,
  "prompt_rewriter_enabled": true,
  "prompt_rewriter_kind": "heuristic",
  "tokenizer_mode": "model-calibrated"
}
```

- [ ] **Step 4: Add production config docs**

Create `docs/PRODUCTION_CONFIG.md` with:

```markdown
# Production Configuration

Use `config.production.example.json` as the local production baseline. It binds
to loopback, requires local proxy auth, rate-limits callers, caps request body
size, enables graph-backed retrieval, enables structural fast-path answers, and
keeps upstream provider auth separate from local proxy auth.

Before serving:

```powershell
.\build\preprocessor_app.exe --health config.production.example.json
.\build\preprocessor_app.exe --serve config.production.example.json
```

Replace `replace-with-strong-local-token` and provide the upstream key through a
local secret mechanism. Do not expose a non-loopback host without auth and a
positive request-size limit.
```

- [ ] **Step 5: Verify config and docs links**

Run:

```powershell
.\build\preprocessor_tests.exe --gtest_filter=ConfigLoaderTest.LoadsProductionExampleConfig
.\build\preprocessor_app.exe --health config.production.example.json
```

Expected: config loader test passes; health may fail only if local model/vocab assets are absent. If health fails for missing model/vocab, document that exact expected local-asset prerequisite in `docs/PRODUCTION_CONFIG.md`.

- [ ] **Step 6: Commit production config slice**

Run:

```powershell
git add config.production.example.json docs/PRODUCTION_CONFIG.md tests/test_config_loader.cpp README.md
git commit -m "docs: add production config baseline"
```

---

### Task 6: Full Release Gate And Report

**Files:**
- No production code changes expected.
- Update docs only if verification reveals a gap.

- [ ] **Step 1: Run full release gate**

Run from a Visual Studio Developer shell:

```powershell
python tools\release_smoke.py
```

Expected:

```text
Release smoke gate completed.
```

- [ ] **Step 2: Capture effectiveness report**

Run:

```powershell
.\build\effectiveness_runner.exe > benchmarks\results\effectiveness-final.json
python tools\perf_visualizer\run_dashboard.py --effectiveness-exe build\effectiveness_runner.exe --run-effectiveness-on-start
```

Expected: effectiveness runner exits 0; dashboard starts on loopback.

- [ ] **Step 3: Query agent summary**

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8787/api/agent-summary
```

Expected: JSON includes `summary.status`, `summary.key_metrics.fixture_top3_pct`, `summary.retrieval.by_category`, and `summary.recommendations`.

- [ ] **Step 4: Commit verification docs if needed**

If the final gate exposed documentation gaps, commit those doc-only fixes:

```powershell
git add README.md docs/RELEASE.md docs/TESTING.md docs/PRODUCTION_CONFIG.md tools/perf_visualizer/README.md
git commit -m "docs: update release verification guidance"
```

If no docs changed, do not create an empty commit.

---

## Self-Review

- Spec coverage: The plan covers current verified work, release/package hygiene, retrieval eval growth, streaming cancellation, production config defaults, and full verification.
- Placeholder scan: No task uses deferred placeholders; each task names files, commands, expected results, and concrete content where new code/docs are introduced.
- Type consistency: Python helpers use `audit_install_tree`; C++ test names and config fields match existing project naming patterns from `ConfigLoaderTest` and `Effectiveness_Retrieval`.

