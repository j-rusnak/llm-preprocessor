# Public Release Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining work needed to publish `llm-preprocessor` safely as a public, production-oriented AI coding-agent preprocessor.

**Architecture:** Keep the existing C++17 proxy/retrieval core unchanged unless a task has test evidence that a small production slice is needed. Add release safety around the project through CI checks, documentation, packaging verification, and focused retrieval/security hardening.

**Tech Stack:** C++17, CMake, GoogleTest, Python release tooling, GitHub Actions, GitGuardian or compatible secret scanning, Playwright dashboard smoke tests.

---

## Current Public-Readiness Status

- Current branch: `polish-features`.
- Current sanitized head: check with `git rev-parse --short HEAD` immediately before pushing.
- GitGuardian-reported high-entropy fixture strings have been replaced in current files and scrubbed from reachable local history.
- Root `.gitignore` ignores `.venv/`, avoiding false positives from vendored Python packages.
- Full local release smoke gate passed after the secret cleanup.
- Publishing requires a force update of the remote `polish-features` branch because history was rewritten.

## Task 1: Publish Sanitized History

**Files:**
- No source edits.
- Remote branch: `origin/polish-features`.

- [ ] **Step 1: Confirm local secret strings are absent**

Run:

```powershell
$RetiredSecret1 = Read-Host "Paste GitGuardian retired value 1"
$RetiredSecret2 = Read-Host "Paste GitGuardian retired value 2"
git grep -n "$RetiredSecret1\|$RetiredSecret2" -- .
git log --all --oneline -S"$RetiredSecret1" -- config.lan.example.json tests/test_config_loader.cpp
git log --all --oneline -S"$RetiredSecret2" -- config.lan.example.json tests/test_config_loader.cpp
```

Expected:

```text
No output from all three commands.
```

- [ ] **Step 2: Force-push the sanitized branch with a lease**

Run:

```powershell
git push --force-with-lease=refs/heads/polish-features:e283c5b14389591fe06538f76cb78c7ee71e75a6 origin polish-features:polish-features
```

Expected:

```text
+ e283c5b...<new-sha> polish-features -> polish-features (forced update)
```

If the lease fails, do not use plain `--force`. Inspect the remote first:

```powershell
git ls-remote origin refs/heads/polish-features
```

- [ ] **Step 3: Re-run GitGuardian on the remote branch**

Run GitGuardian against the pushed branch or refresh the GitHub/GitGuardian check.

Expected:

```text
No active Generic High Entropy Secret findings for config.lan.example.json or tests/test_config_loader.cpp.
```

## Task 2: Add Automated Secret Scanning To CI

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `tools/secret_scan.py`
- Test: `tools/tests/test_secret_scan.py`

- [ ] **Step 1: Add scanner unit tests**

Create `tools/tests/test_secret_scan.py` with tests for:

```python
def test_rejects_high_entropy_fixture_token():
    value = "fixture-token-" + "4f7c9a2b" + "8d6e1f03"
    sample = f'proxy_auth_bearer_tokens = ["{value}"]'
    assert scan_text(sample)

def test_allows_documented_placeholder():
    sample = 'proxy_auth_bearer_tokens = ["replace-with-strong-lan-proxy-token"]'
    assert not scan_text(sample)
```

- [ ] **Step 2: Implement `tools/secret_scan.py`**

Implement a Python scanner that:

```text
Scans git ls-files only.
Skips binary files.
Flags common provider key formats.
Flags high-entropy quoted values assigned to token, secret, api_key, password, or authorization fields.
Allows explicit placeholders beginning with replace-with- and angle-bracket examples such as <local-token>.
Prints file:line:type and exits 1 on findings.
```

- [ ] **Step 3: Wire the scanner into CI**

Add a GitHub Actions step after `Release hygiene`:

```yaml
- name: Secret scan
  shell: pwsh
  run: python tools/secret_scan.py
```

- [ ] **Step 4: Verify**

Run:

```powershell
python -m unittest discover tools\tests -p "test_*.py"
python tools\secret_scan.py
```

Expected:

```text
All release-tooling tests pass.
Secret scanner exits 0 on the current tree.
```

## Task 3: Tighten Release Provenance

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `tools/release_smoke.py`
- Test: existing release smoke gate

- [ ] **Step 1: Ensure version metadata refreshes after history rewrites**

Update the release smoke script to run configure before build:

```powershell
cmake -S . -B build
cmake --build build --config Debug --parallel
```

- [ ] **Step 2: Assert app version commit matches `git rev-parse --short HEAD`**

Add a release smoke check that compares:

```powershell
.\build\preprocessor_app.exe --version
git rev-parse --short HEAD
```

Expected:

```text
The version output contains the current short commit.
```

- [ ] **Step 3: Verify**

Run:

```powershell
python tools\release_smoke.py
```

Expected:

```text
Release smoke gate completed.
```

## Task 4: Improve Retrieval Top-1 Ranking

**Files:**
- Modify: `src/hybrid_retriever.cpp`
- Modify: `src/retrieval_query.cpp`
- Test: `tests/test_effectiveness.cpp`
- Test: `benchmarks/effectiveness_runner.cpp`

- [ ] **Step 1: Add failing top-1 regression cases**

Add fixture queries for current near misses:

```text
cpp stream forwarder cancel upstream on client disconnect idle timeout -> realworld/cpp/stream_forwarder.cpp
markdown retrieval debugging near miss expected rank diagnostics -> docs/retrieval-debugging.md
```

- [ ] **Step 2: Tune ranking with query-path and symbol evidence**

Favor exact language plus filename/symbol overlap before broad production/security documents when vector scores are close.

- [ ] **Step 3: Verify**

Run:

```powershell
.\build\preprocessor_tests.exe --gtest_filter=Effectiveness_Retrieval.*
.\build\effectiveness_runner.exe
```

Expected:

```text
FixtureRetrieval top-3 remains 100%.
Top-1 improves above the current 72% baseline without unrelated graph pollution.
```

## Task 5: Public Documentation Pass

**Files:**
- Modify: `README.md`
- Modify: `docs/RELEASE.md`
- Modify: `docs/PRODUCTION_CONFIG.md`

- [ ] **Step 1: Add public-push security note**

Document:

```text
Never commit local copied configs.
Use placeholders only in tracked examples.
Run python tools/secret_scan.py before pushing.
Use X-Preprocessor-Authorization for local proxy auth.
```

- [ ] **Step 2: Add post-push checklist**

Document:

```text
CI green.
GitGuardian clean.
Release smoke gate passed.
Package audit status ok.
LAN config copied privately and token replaced outside Git.
```

- [ ] **Step 3: Verify docs references**

Run:

```powershell
rg -n "profile-token|change-me-32chars|View secret|real provider key" README.md docs config*.json tests
```

Expected:

```text
No output.
```

## Self-Review

- Spec coverage: The plan covers remote publication of scrubbed history, automated secret scanning, release provenance, retrieval quality, and public-facing docs.
- Placeholder scan: Placeholder language appears only as explicit safe example text and does not ask an implementer to fill in unspecified work.
- Type consistency: Tool names and paths match the current repository layout.
