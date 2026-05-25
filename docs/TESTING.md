# Testing & Feature Usage Guide

End-to-end walkthrough for verifying every feature of the LLM Preprocessor and
for driving each module from the command line, a config file, or as a library.

- [1. One-command quick start](#1-one-command-quick-start)
- [2. Build environment](#2-build-environment)
- [3. The four test surfaces](#3-the-four-test-surfaces)
- [4. Effectiveness runner — what it measures and how to read it](#4-effectiveness-runner)
- [5. Feature-by-feature usage](#5-feature-by-feature-usage)
- [6. Production deployment recipes](#6-production-deployment-recipes)
- [7. Troubleshooting](#7-troubleshooting)

---

## 1. One-command quick start

From a Visual Studio Developer PowerShell at the repo root:

```powershell
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
.\build\smoke_runner.exe
.\build\effectiveness_runner.exe > benchmarks\results\effectiveness.json
```

Expected:

| Surface | Pass criterion |
|---|---|
| `ctest` | `100% tests passed, 0 tests failed` |
| `smoke_runner.exe` | `Summary: 54 passed, 0 failed.` |
| `effectiveness_runner.exe` | Exit 0, summary table on stderr, JSON on stdout |

---

## 2. Build environment

`cl.exe` is invoked directly by Ninja, so the shell **must** have MSVC env vars
(`INCLUDE`, `LIB`, `PATH`). VS Code's default integrated PowerShell is NOT a
developer shell.

### Option A — Start menu (easiest)

Launch **"x64 Native Tools Command Prompt for VS 2022"** *or* **"Developer
PowerShell for VS 2022"**, then `cd` to the repo.

### Option B — Activate from PowerShell

```powershell
$vsdev = & "C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -property installationPath
Import-Module "$vsdev\Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell -VsInstallPath $vsdev -SkipAutomaticLocation `
    -DevCmdArguments "-arch=x64 -host_arch=x64"
```

### Option C — Activate from cmd.exe

```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
```

Adjust the edition (`Enterprise` / `Professional` / `Community`) and year
(`2022`) to match your install.

**Symptom of a missing env:** `fatal error C1083: Cannot open include file:
'string'` / `'cstddef'` / `'cstdint'` from gtest.h or any project header.
If you see this, you skipped one of the steps above. PowerShell cmdlets like
`$vsdev = ...` / `Enter-VsDevShell` do NOT work inside `cmd.exe` — use
Option C there.

---

## 3. The four test surfaces

### 3.1 Unit tests (Google Test)

```powershell
ctest --test-dir build --output-on-failure
```

Filter to one module:

```powershell
.\build\preprocessor_tests.exe --gtest_filter=VectorStoreTest.*
.\build\preprocessor_tests.exe --gtest_filter=Effectiveness_*
.\build\preprocessor_tests.exe --gtest_filter=PromptCacheTest.*:DiffPatcherTest.*
```

List every test case:

```powershell
.\build\preprocessor_tests.exe --gtest_list_tests
```

### 3.2 Smoke runner (54 stages)

End-to-end, in-memory rehearsal of every Phase 0-12 module. No ONNX model
required (deterministic fake embedder).

```powershell
.\build\smoke_runner.exe            # summary only
.\build\smoke_runner.exe --verbose  # per-stage PASS/FAIL detail
```

Exit code is `0` on full pass — safe for CI.

### 3.3 Benchmark runner (router latency / accuracy)

Targets the **legacy semantic router** path with 31 prompts:

```powershell
.\build\benchmark_runner.exe > benchmarks\results\benchmark_data.json
python benchmarks\visualize.py
```

Outputs PNG charts into `benchmarks\results\`. See
[Benchmarks & Visualizations](../README.md#benchmarks--visualizations).

### 3.4 Effectiveness runner (Phase 5-12 quality gates)

See section 4.

---

## 4. Effectiveness runner

### 4.1 What it measures

| Module | Metric |
|---|---|
| `PromptCache` | hit-rate, warm-vs-cold latency, speedup multiplier |
| `HeuristicCompressionRewriter` | char reduction %, token reduction %, original vs rewritten size |
| `StreamingCompactor` | turns rolled, turns kept, char-budget compliance |
| `EmbeddingCache` | round-trip latency, cold-vs-warm speedup |
| `DiffPatcher` | bytes saved vs full-file transport, apply success |
| `BM25Index` | top-1 / top-3 accuracy across hand-built queries, microseconds per query |
| `ModelRouter` | routing accuracy on bucket+char-length test cases, microseconds per route |
| `AbHarness` | sticky-assignment determinism (100/100), weighted balance (chi-squared) |
| `AuthMiddleware` | HMAC verifies/second, microseconds per op |
| `RateLimiter` | tokens granted in burst, refill behaviour, independent-key isolation |

### 4.2 How to run

```powershell
.\build\effectiveness_runner.exe > benchmarks\results\effectiveness.json
```

Stderr prints a human-readable summary table:

```
=== LLM Preprocessor Effectiveness Report ===

[PromptCache]          hit rate 100.0% | speedup 1.0x | warm 20.70 us vs cold 20.00 us
[PromptRewriter]       chars 1204 -> 782 (35.0%) | tokens 237 -> 150 (36.7%)
[StreamingCompactor]   40 turns -> 9 kept + 31 rolled | 8690 -> 4913 chars (43.5%)
[EmbeddingCache]       500 vectors | speedup 9.0x (cold 188.39 us -> warm 20.95 us)
[DiffPatcher]          5988 B full vs 176 B diff (97.1% saved) | applied=yes
[BM25Index]            17 docs / 8 queries | top-1 100.0% | top-3 100.0% | 37.69 us/query
[ModelRouter]          4/4 correct (100.0%) | 0.57 us/route
[AbHarness]            10000 assigns | A=2496 B=2502 C=5002 | chi^2=0.01 (crit 9.21) | sticky=100/100
[AuthMiddleware]       50000 HMAC verifies | 11.14 us/op | 89802/s
[RateLimiter]          phase1 allowed 10/50 (expected 10) | phase2 allowed 0 | independent-key=yes
```

### 4.3 JSON schema (stdout)

```jsonc
{
  "prompt_cache":        { "hit_rate": 1.0, "speedup": 9.1, "warm_us": 20.7, "cold_us": 188.4 },
  "prompt_rewriter":     { "chars_in": 1204, "chars_out": 782, "char_pct": 0.350,
                           "tokens_in": 237, "tokens_out": 150, "token_pct": 0.367 },
  "streaming_compactor": { "turns_in": 40, "kept": 9, "rolled": 31,
                           "chars_in": 8690, "chars_out": 4913 },
  "embedding_cache":     { "vectors": 500, "cold_us": 188.4, "warm_us": 20.9, "speedup": 9.0 },
  "diff_patcher":        { "full_bytes": 5988, "diff_bytes": 176, "saved_pct": 0.971, "applied": true },
  "bm25":                { "docs": 17, "queries": 8, "top1": 1.0, "top3": 1.0, "us_per_query": 37.7 },
  "model_router":        { "cases": 4, "correct": 4, "accuracy": 1.0, "us_per_route": 0.57 },
  "ab_harness":          { "assigns": 10000, "counts": {"A":2496,"B":2502,"C":5002},
                           "chi_squared": 0.01, "sticky_ok": 100 },
  "auth_middleware":     { "verifies": 50000, "us_per_op": 11.1, "ops_per_sec": 89802 },
  "rate_limiter":        { "phase1_allowed": 10, "phase2_allowed": 0,
                           "expected_burst": 10, "independent_key": true }
}
```

### 4.4 Regression thresholds

The `Effectiveness_*` gtest cases lock in conservative floors:

| Test | Floor |
|---|---|
| `Effectiveness_PromptRewriter.ReducesCharCount` | ≥15% char reduction |
| `Effectiveness_PromptRewriter.ReducesTokenCount` | rewritten < original |
| `Effectiveness_StreamingCompactor.StaysUnderHardCap` | kept window ≤ `max_total_chars` and `rolled > 0` |
| `Effectiveness_PromptCache.HitsAreFasterThanFreshComputeBy3x` | cold/warm ≥ 3x |
| `Effectiveness_EmbeddingCache.RoundTripsVectorsByModelId` | per-model isolation |
| `Effectiveness_DiffPatcher.DiffIsSmallerThanFullFile` | diff < full content |
| `Effectiveness_BM25.RetrievesCorrectDocInTop3` | top-1 correct on 2 queries |
| `Effectiveness_ModelRouter.PicksCheapForSmallEdits` | bucket + char-length routing |
| `Effectiveness_AbHarness.IsSticky` | identical key → identical variant |
| `Effectiveness_AbHarness.RoughlyBalancesWeights` | 50/50 within ±5% over 5000 |
| `Effectiveness_AuthMiddleware.AcceptsValidAndRejectsInvalid` | valid sig OK; bad sig + stale ts rejected |
| `Effectiveness_RateLimiter.BurstThenRefill` | burst=5 exact, refill 0<p2≤10 |

---

## 5. Feature-by-feature usage

All examples assume `preprocessor::` namespace.

### 5.1 `OpenAIProxy` (Phase 1) — drop-in HTTP server

```powershell
.\build\preprocessor_app.exe --serve config.json
```

Then point any OpenAI-compatible client at `http://127.0.0.1:<port>`:

```powershell
curl -s http://127.0.0.1:8080/healthz
curl -s http://127.0.0.1:8080/stats
curl -s http://127.0.0.1:8080/v1/chat/completions `
    -H "Content-Type: application/json" `
    -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"explain main()"}]}'
```

Streaming requests preserve SSE framing and bypass `PromptCache`:

```powershell
curl -N http://127.0.0.1:8080/v1/chat/completions `
    -H "Content-Type: application/json" `
    -d '{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"explain main()"}]}'
```

### 5.2 `McpServer` (Phase 4) — MCP over stdio

```powershell
.\build\preprocessor_app.exe --mcp config.json
```

Exposes three tools (`search_repo`, `structural_query`, `get_chunk`) and two
resources (`repo://card`, `repo://stats`). Wire it into VS Code via
[`vscode-extension/`](../vscode-extension/README.md).

### 5.3 `--health` (Phase 12) — preflight check

```powershell
.\build\preprocessor_app.exe --health config.json
echo "exit=$LASTEXITCODE"
```

Validates the config, ONNX model file, vocab, and any referenced template
overrides. Non-zero exit means *do not start the server*.

### 5.4 `PromptCache`

```cpp
preprocessor::PromptCache cache("cache.sqlite", /*ttl_seconds=*/86400);
auto key = preprocessor::PromptCache::make_key(model, compiled_upstream_request, chunk_ids);
if (auto hit = cache.get(key)) {
    return *hit;            // skip upstream
}
cache.put(key, upstream_response);
```

The key includes the model id and the full compiled upstream request, so changes
to parameters such as `temperature`, tools, or injected context do not collide.

### 5.5 `HeuristicCompressionRewriter` (Phase 5)

```cpp
preprocessor::HeuristicCompressionRewriter r;
std::string compact = r.rewrite(big_system_context, /*max_chars=*/4000);
```

Strips `//` and `/* … */` (but preserves strings and `#preproc` lines),
collapses blank-line runs, dedupes adjacent lines, optionally truncates with a
`... [truncated]` marker. Wire it into the proxy with
`OpenAIProxy::set_prompt_rewriter`.

Config:

```json
{
  "prompt_rewriter_enabled": true,
  "prompt_rewriter_kind": "heuristic",
  "prompt_rewriter_max_chars": 8000
}
```

### 5.6 `StreamingCompactor` (Phase 11)

```cpp
preprocessor::StreamingCompactor::Config cfg;
cfg.max_total_chars = 6000;
cfg.summary_chars_per_turn = 160;
cfg.keep_recent = 4;
preprocessor::StreamingCompactor sc(cfg);

auto r = sc.compact(chat_history);
// r.rolled_summary -> inject as a leading system message
// r.kept           -> append verbatim
```

### 5.7 `EmbeddingCache` (Phase 7)

```cpp
preprocessor::EmbeddingCache cache("embeddings.sqlite", "all-MiniLM-L6-v2");
if (auto v = cache.get(chunk_text)) return *v;
auto v = embedder.embed(chunk_text);
cache.put(chunk_text, v);
```

Cross-process, cross-repo safe — the model id is part of the key so a model
swap automatically invalidates.

### 5.8 `DiffPatcher` (Phase 6)

```cpp
preprocessor::DiffPatcher p;
std::unordered_map<std::string, std::string> files{{"src/x.cpp", original}};
auto r = p.apply(unified_diff, files);
if (r.ok && r.files[0].applied) {
    write_to_disk("src/x.cpp", r.files[0].patched_content);
}
```

Permissive (`apply` / `parse` — skips malformed hunks) or strict
(`parse_strict` — throws). `a/` / `b/` path prefixes are stripped automatically.

### 5.9 `BM25Index` + `HybridRetriever` (Phase 1)

```cpp
preprocessor::BM25Index idx;
idx.add(1, "configuration loader reads JSON files");
idx.add(2, "vector store HNSW nearest neighbour search");
auto hits = idx.search("HNSW nearest neighbour", /*k=*/3);
// hits[0].id == 2
```

`HybridRetriever` fuses ANN + BM25 hits via Reciprocal Rank Fusion (`k=60`).

### 5.10 `ModelRouter` (Phase 8)

```cpp
preprocessor::ModelRouter r;
r.add_tier({"cheap",    "https://api.x/v1", "gpt-4o-mini", "$KEY1", 16000});
r.add_tier({"frontier", "https://api.x/v1", "gpt-4o",      "$KEY2", 128000});
r.add_route({preprocessor::PromptBucket::CodeEdit,     0,    1000, "cheap"});
r.add_route({preprocessor::PromptBucket::CodeGenerate, 4000, 0,    "frontier"});

auto tier = r.route(bucket, request_chars);   // const ModelTier* or nullptr
```

### 5.11 `AbHarness` (Phase 9)

```cpp
preprocessor::AbHarness ab;
ab.define({"system-prompt-v3", {{"control",1},{"compact",1},{"verbose",2}}});
std::string variant = ab.assign("system-prompt-v3", user_id);
// ... use variant ...
ab.record_hit("system-prompt-v3", variant);
auto counts = ab.hit_counts();
```

Sticky on `xxhash64(experiment_id + '\0' + user_id)` — same user always gets
the same variant.

### 5.12 `SyncEndpoint` (Phase 10)

```cpp
preprocessor::SyncBundle b;
b.cache.push_back({key, body});
b.vectors.push_back({chunk_id, vec, "src/x.cpp"});
nlohmann::json payload = preprocessor::SyncEndpoint::to_json(b);
// POST payload to a teammate's /sync endpoint
auto applied = preprocessor::SyncEndpoint::apply_to_cache(b, &cache);
```

Transport-agnostic — wrap it in any HTTP/gRPC layer you already have.

### 5.13 `AuthMiddleware` (Phase 12)

```cpp
preprocessor::AuthMiddleware::Config cfg;
cfg.allowed_bearer_tokens = {"team-token-1", "team-token-2"};
cfg.hmac_secret           = "shared-secret";
cfg.max_clock_skew        = 300;          // seconds
preprocessor::AuthMiddleware mw(cfg);

const std::int64_t ts = std::time(nullptr);
auto sig = preprocessor::AuthMiddleware::sign(cfg.hmac_secret, ts, body);
bool ok = mw.verify(bearer_header, sig, std::to_string(ts), body, ts);
```

### 5.14 `RateLimiter` (Phase 12)

```cpp
preprocessor::RateLimiter rl({/*tps=*/5.0, /*burst=*/10.0});
if (!rl.try_acquire(api_key_or_ip)) {
    return 429;
}
```

Pass an explicit `now_seconds` from tests; `0.0` means *use the real clock*.

### 5.15 `ProjectCard` + `PromptOptimizer` (Phase 2)

Toggle in config:

```json
{
  "prompt_optimizer_enabled": true,
  "include_project_card": true,
  "prompt_templates_path": "templates.json"
}
```

`templates.json` overrides built-in `inja` scaffolds per `PromptBucket`.

### 5.16 `SymbolGraph` + `StructuralQueryEngine` (Phase 3)

Toggle in config:

```json
{
  "symbol_graph_enabled": true,
  "graph_expansion_enabled": true,
  "structural_fast_path_enabled": true
}
```

When enabled, queries like *"where is `Foo` defined"* or *"what calls
`bar`"* are answered locally with **zero LLM forward**.

---

## 6. Production deployment recipes

### 6.1 Auth + rate-limited proxy

`AuthMiddleware` and `RateLimiter` are wired into `ConfigLoader` and
`OpenAIProxy` for `/v1/chat/completions` and `/stats`. `/healthz` remains a
public liveness check. Non-loopback serving requires proxy auth unless
`allow_unsafe_remote_proxy` is explicitly set.

```json
{
  "proxy_host": "0.0.0.0",
  "proxy_auth_bearer_tokens": ["local-proxy-token"],
  "proxy_rate_limit_tokens_per_second": 2.0,
  "proxy_rate_limit_burst": 10.0,
  "proxy_max_request_bytes": 8388608,
  "tokenizer_mode": "model-calibrated",
  "proxy_forward_client_authorization": false,
  "upstream_api_key": "provider-key"
}
```

Use `X-Preprocessor-Authorization: Bearer <token>` for local proxy auth when a
client also needs to send a provider key in `Authorization`.

`tokenizer_mode: "model-calibrated"` keeps request/compiled token estimates and
`/stats.tokens_by_model_family` aligned with routed model families such as
`gpt-4o`, `gpt-4.1`, Claude, and Gemini.

```powershell
.\build\preprocessor_app.exe --health config.prod.json
.\build\preprocessor_app.exe --serve  config.prod.json
```

### 6.2 Multi-tier routing

`ModelRouter` is wired into `--serve` via `model_tiers` and `model_routes`.
The proxy classifies the latest user message, applies the first matching route,
rewrites the outbound model, and uses tier-specific upstream URL/API-key/context
overrides before cache lookup and forwarding. Regression coverage lives in
`ConfigLoaderTest.LoadsModelRouterConfig` and
`OpenAIProxy.ModelRouterSelectsTierAndRewritesForwardedRequest`.

### 6.3 Team mode (shared cache + vectors)

`OpenAIProxy` exposes authenticated `GET/POST /sync/cache` and
`GET/POST /sync/vectors` routes around `SyncEndpoint`. These routes use the
same bearer/HMAC middleware, rate limiter, and request-size guard as the proxy
runtime. Vector bundles include embedding data plus chunk metadata so imports
can hydrate the local retrieval index.

### 6.4 MCP for VS Code

```powershell
.\build\preprocessor_app.exe --mcp config.json
```

Then configure `vscode-extension/` to spawn that command — see
[`vscode-extension/README.md`](../vscode-extension/README.md).

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ninja: no work to do` after editing `CMakeLists.txt` | Re-run `cmake -B build` to regenerate the build graph |
| `Failed to change working directory to .../build/build` | You're inside `build/`. `Set-Location` to the repo root before any `ctest --test-dir build` |
| `fatal error C1083: Cannot open include file: 'string'` | Shell missing MSVC env vars; see section 2 |
| `[FATAL] effectiveness runner failed: remove: file in use` | A `PromptCache` / `EmbeddingCache` is still alive when `fs::remove` runs. Scope the cache `{ ... }` first |
| `RateLimiter` always allows everything in a unit test | You passed `now=0.0`. That means *real clock*. Pass `100.0`, `101.0`, etc. |
| `--health` exits non-zero on a fresh checkout | The ONNX model isn't downloaded; see [Setting Up the Model](../README.md#setting-up-the-model) |
| MCP / `--serve` clash | They are mutually exclusive; pick one per process |
| Effectiveness JSON empty / runner exits 1 | Run `.\build\effectiveness_runner.exe` directly (not piped) to see stderr |
