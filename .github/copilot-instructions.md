# Role and Goal
You are an expert C++ developer building a **plug-and-play middleware for AI coding
assistants** (Cursor, Continue, Copilot, Claude Code, etc.). Your job is to write
production-grade, low-latency C++ that intercepts prompts/completions, retrieves
the smallest relevant slice of local code context, enforces token budgets, and
forwards an optimised payload to the upstream LLM. The legacy command-routing path
(local OS actions via semantic intent matching) is preserved as a side feature.

# Project Context & Architecture
- **Project Name:** LLM Preprocessor
- **Primary goals (in order):**
  1. Reduce input/output tokens sent to upstream LLMs (cost).
  2. Reduce end-to-end latency (speed).
  3. Act as a smart local context engine (chunk + embed + index + retrieve).
- **Core pipeline:** Intercept prompt -> Sanitize -> (optional) Intent route ->
  Chunk + embed repo on first run / on file change -> ANN retrieve top-k chunks
  -> Compile prompt within token budget -> Forward to LLM.
- **Non-goals:** training models, hosting LLMs, IDE UI, language-server
  features. We *consume* clangd/LSPs in later phases; we never replace them.

# Roadmap (kept here so multi-step work stays aligned)
- **Phase 0 (DONE):** Foundation fixes - real ANN (`hnswlib`), content-addressed
  chunks (`xxhash`), filesystem watching (`efsw`), batched ONNX inference,
  `MemoryEngine` split into `ChatHistoryStore` + `VectorStore`, downstream LLM
  tokenizer interface, `IChunker` interface + `LineWindowChunker` fallback,
  end-to-end smoke runner.
- **Phase 1 (DONE):** MVP RAG proxy - `BraceAwareChunker` (AST-ish chunker;
  tree-sitter slots in behind the same `IChunker` later), `OpenAIProxy`
  (cpp-httplib server with `POST /v1/chat/completions`, `GET /healthz`,
  `GET /stats`), `BM25Index` + `HybridRetriever` (RRF fusion, `k=60`),
  `PromptCache` (SQLite, xxhash64 of `(model, compiled upstream request,
  sorted_chunk_ids)`, optional TTL), `RepoIndex` wiring `FileWatcher` to incremental
  re-indexing, `ProxyMetrics` for telemetry on tokens saved, `--serve`
  mode in `main`. Dependency added: `cpp-httplib`.
- **Phase 2 (DONE):** Project card + per-bucket prompt templates -
  `ProjectCard` + `ProjectCardBuilder` (extension histogram, top symbols,
  README excerpt over `RepoIndex::snapshot_chunks`), `IIntentClassifier` +
  `HeuristicIntentClassifier` (CodeEdit / CodeExplain / CodeGenerate /
  MetaQuery / Freeform), `PromptTemplates` (built-in scaffolds rendered
  via `inja`, JSON-overridable), `PromptOptimizer` (toggleable system-
  message rewriter wired into `OpenAIProxy::set_prompt_optimizer`), new
  config keys `prompt_optimizer_enabled` / `prompt_templates_path` /
  `include_project_card`. Dependency added: `inja` (`pantor::inja`).
- **Phase 3 (DONE):** Symbol graph + zero-LLM fast path -
  `SymbolGraph` (thread-safe defs/refs store with `neighbors_of` 1-hop
  expansion), `ISymbolExtractor` + `RegexSymbolExtractor` (comment +
  string stripping, reserved-word filtering, multi-language patterns;
  tree-sitter slots in behind the same interface later), `RepoIndex`
  hooks (`attach_symbol_graph`, `try_get_chunk`, graph (re)population in
  `index_file_locked` / `forget_file_locked`), `expand_with_graph`
  retrieval expander (query-aware ranked neighbour scores), `StructuralQueryEngine`
  zero-LLM fast path (definition / callers / functions-in-file / repo
  stats), `OpenAIProxy` wiring (`set_symbol_graph`,
  `set_structural_query_engine`; fast-path runs before retrieval, graph
  expansion runs before cache-key computation). New config keys
  `symbol_graph_enabled`, `graph_expansion_enabled`,
  `structural_fast_path_enabled`. No new vcpkg deps.
- **Phase 4 (DONE):** MCP server mode + VS Code extension; single-binary
  distribution. `McpServer` (JSON-RPC 2.0 over newline-delimited stdio)
  exposes three tools (`search_repo`, `structural_query`, `get_chunk`)
  and two resources (`repo://card`, `repo://stats`), wiring `RepoIndex`,
  `SymbolGraph`, `StructuralQueryEngine`, and `ProjectCard` together
  with no LLM in the loop. `main.cpp` gains a `--mcp` flag (mutually
  exclusive with `--serve`) so the same binary speaks both the OpenAI
  HTTP protocol and MCP. `vscode-extension/` is a minimal TypeScript
  shim that registers the binary with VS Code's Language Model host
  (1.99+). No new vcpkg deps.
- **Phase 5 (DONE):** Prompt rewriter / context compressor -
  `IPromptRewriter` interface, `HeuristicCompressionRewriter`
  (always available, dependency-free: strips `//` / `#` line comments
  and `/* ... */` blocks while preserving string literals and C
  preprocessor directives, collapses blank-line runs, dedupes adjacent
  lines, trims trailing whitespace, optional hard char cap with
  `... [truncated]` marker), `LlamaCppRewriter` (stub gated behind
  CMake option `LLM_PREPROCESSOR_WITH_LLAMA_CPP`, default OFF;
  constructor throws when the flag is off). `OpenAIProxy` gains
  `set_prompt_rewriter` and applies the rewriter to the assembled
  system context immediately before the cache-key + upstream forward.
  New config keys: `prompt_rewriter_enabled`, `prompt_rewriter_kind`,
  `prompt_rewriter_max_chars`, `llama_model_path`. No new vcpkg deps;
  `llama.cpp` linkage is deferred to Phase 6 once the model story is
  finalised. Full ctest and smoke suites pass.
- **Phase 6 (DONE):** Diff-aware response patching - `DiffPatcher`
  permissive unified-diff parser + applier. Parses `diff --git` /
  `--- ` / `+++ ` headers and `@@` hunks, validates context against
  current file content via an injected `unordered_map<path,
  contents>`, applies in memory (caller decides when to flush).
  Strips `a/` / `b/` path prefixes; permissive `parse` skips malformed
  hunks while `parse_strict` throws. Both struct-vector and
  `string_view`-overload `apply` paths. No new vcpkg deps.
- **Phase 7 (DONE):** Persistent + cross-repo embedding cache -
  `EmbeddingCache` backed by SQLite (`embeddings(key INTEGER PRIMARY
  KEY, model TEXT, dim INTEGER, vec BLOB)`) keyed by xxhash64 of
  `model_id || '|' || content`. WAL + `synchronous=NORMAL` pragmas;
  thread-safe via `std::mutex`. Methods: `get` / `get_by_key` / `put`
  / `put_by_key` / `size` / `clear` / `key_for`. Eliminates
  re-embedding across process restarts and sibling repos. No new
  vcpkg deps.
- **Phase 8 (DONE):** Multi-tier model routing - `ModelTier{name,
  upstream_url, model_name, api_key, max_context}` + `ModelRoute{
  PromptBucket bucket, min/max_request_chars, tier}`. `ModelRouter`
  with `add_tier` / `add_route` / `route(bucket, chars)` /
  `tier(name)`. Config keys: `model_tiers` and ordered `model_routes`.
  `OpenAIProxy` applies the selected tier before cache lookup and forwarding.
  Thread-safe via `std::mutex`. No new vcpkg deps.
- **Phase 9 (DONE):** Telemetry-driven prompt evolution - `AbHarness`
  with `AbVariant{name, weight}` + `AbExperiment{id, variants}`.
  `define(experiment)` rejects empty / all-zero-weight inputs.
  `assign(experiment_id, sticky_key)` hashes
  `xxhash64(experiment_id + '\0' + sticky_key)` into the weighted
  variant range for deterministic, sticky assignment. `record_hit` /
  `hit_counts` keyed `"experiment::variant"` for offline analysis.
  No new vcpkg deps.
- **Phase 10 (DONE):** Team mode - `SyncEndpoint` transport-agnostic
  serializer for shared cache + vector bundles. `SyncBundle{cache,
  vectors}` with `SyncCacheEntry{key, body}` and `SyncVectorEntry{
  chunk_id, vec, source_path}`. `to_json` / `from_json` over
  `nlohmann::json`; `apply_to_cache(bundle, PromptCache*)` returns
  applied count and bumps `bundles_imported()`. `OpenAIProxy` has
  authenticated `GET/POST /sync/cache` and `GET/POST /sync/vectors`.
  Vector bundle entries also carry chunk text/line/symbol metadata so
  imports can hydrate `RepoIndex`. No new vcpkg deps.
- **Phase 11 (DONE):** Streaming-aware compaction -
  `StreamingCompactor` folds older `ChatTurn{role, content}` entries
  into a rolling summary string suitable for re-injection as a
  system message. `Config{max_total_chars=6000,
  summary_chars_per_turn=160, keep_recent=4}` keeps the most recent
  N turns verbatim, grows the kept window until budget, and caps the
  rolled summary plus kept turns to `max_total_chars` when possible.
  Dependency-free; pluggable behind the same surface as the
  Phase 5 rewriter. No new vcpkg deps.
- **Phase 12 (DONE):** Production hardening - `AuthMiddleware` with
  bearer-token allow-list + HMAC-SHA256 (inline RFC-6234
  implementation in `src/auth_middleware.cpp`; constant-time hex
  compare). `verify(authorization, signature, timestamp, body,
  now_unix=0)` honours `max_clock_skew`. `RateLimiter`
  token-bucket per caller key with `tokens_per_second` / `burst`
  knobs (0/0 = disabled). `ConfigLoader` and `OpenAIProxy` wire auth,
  rate limiting, max request bytes, and unsafe non-loopback guards into
  `--serve`; `main.cpp` also has a `--health` flag that validates config +
  ONNX assets and exits non-zero on missing files. No new vcpkg deps. Full
  ctest and smoke suites pass.
- **Runtime token budgeting (DONE):** `tokenizer_mode` selects
  `HeuristicLLMTokenizer` or `ModelCalibratedLLMTokenizer`. The calibrated
  mode buckets routed model names into coarse families (`gpt-4o`, `gpt-4.1`,
  `gpt-5`, OpenAI reasoning, Claude, Gemini, default) and `ProxyMetrics`
  exposes `/stats.tokens_by_model_family` with original/compiled/saved totals.
- **Symbol extraction upgrade (DONE):** `RegexSymbolExtractor` covers JS/TS
  exported functions/classes, arrow function assignments, exported variables,
  and aliases qualified C++ method definitions by simple name for structural
  lookup and graph expansion.
- **Graph retrieval ranking (DONE):** `expand_with_graph` accepts
  `GraphExpansionConfig::query_text`, ranks expansion candidates by query match,
  reference count, symbol kind, and deterministic file/id tie-breaks, and the
  effectiveness harness reports graph top-3 lift plus unrelated pollution.
- **Effectiveness harness (DONE):**
  `benchmarks/effectiveness_runner.cpp` standalone runner emits a
  JSON report (stdout) + human summary table (stderr) covering every
  Phase 5-12 module: cache hit rate / speedup, rewriter char+token
  reduction, streaming compactor budget compliance, embedding cache
  speedup, diff transport savings, BM25 top-1/top-3 accuracy, graph
  retrieval lift/pollution, model router accuracy, A/B sticky+balance,
  HMAC verifies/sec, rate-limit
  burst+refill. `tests/test_effectiveness.cpp` adds 14
  `Effectiveness_*` gtest cases that lock in conservative thresholds
  (cache ≥3x, rewriter ≥15% chars, BM25 100% on hand-built queries,
  A/B 50/50 ±5%, rate-limit burst exact). Full end-to-end testing &
  feature-usage guide in `docs/TESTING.md`. Full ctest and smoke suites pass.
  No new vcpkg deps.

# Tech Stack & Build System
- **Language Standard:** C++17 (strictly enforced).
- **Build System:** CMake 3.15+. Add every new `.cpp` to `preprocessor_lib` and
  every new test to `preprocessor_tests`.
- **Dependency Management:** vcpkg manifest (`vcpkg.json`).
- **Allowed Third-Party Libraries:**
  - `nlohmann/json`, `libcurl`, `sqlite3`, `gtest`
  - `onnxruntime` (pre-built binary, not vcpkg) - local embedding inference
  - `hnswlib` - HNSW ANN index over code embeddings
  - `xxhash` - content-addressed chunk IDs + cache keys
  - `efsw` - cross-platform file-system watcher
  - `cpp-httplib` - embedded HTTP server for the OpenAI-compatible proxy
  - `inja` - Jinja2-style template engine for per-bucket prompt scaffolds
  - Phase-3+ additions (when introduced): `tree-sitter` + grammars,
    a BPE tokenizer (e.g. `cpp-tiktoken`), `llama.cpp` (Phase 5)

# Directory Structure (Flat)
- `include/` - public headers (`.hpp`); `#pragma once`; minimal includes.
- `src/` - one `.cpp` per header.
- `tests/` - one `test_<module>.cpp` per module, plus `smoke_runner.cpp` for
  end-to-end integration.
- `CMakeLists.txt`, `vcpkg.json` at the root.

Do NOT introduce nested subfolders inside `include/` or `src/` without an
explicit request.

# C++ Coding Standards & Best Practices
1. `#pragma once` at the top of every header; forward-declare in headers, fully
   include in `.cpp`.
2. Local includes use `"..."`; standard/external use `<...>`.
3. RAII everywhere. No raw `new` / `delete`. Use `std::unique_ptr` /
   `std::shared_ptr`. Use the pimpl idiom when a header would otherwise pull
   in a heavy third-party header (see `FileWatcher::Impl`).
4. Throw `std::runtime_error` / `std::invalid_argument` on failure; never fail
   silently. Network code (libcurl) must set timeouts and check HTTP codes.
5. Pass complex objects by `const&`. Prefer `std::string_view` for read-only
   string params introduced in new code.
6. Wrap all project code in `namespace preprocessor`.
7. Anything that touches the upstream LLM must go through `ILLMTokenizer` for
   budget accounting; do not eyeball token counts.
8. Anything that stores or searches vectors must go through `VectorStore`; do
   not implement ad-hoc cosine loops.
9. New chunker implementations derive from `IChunker`. New embedder
   implementations from `IEmbeddingEngine`.

# Workflow Instructions for the AI
- For new features: generate the `.hpp` interface first, then the `.cpp`.
- When you add a new `.cpp`, add it to `preprocessor_lib` in `CMakeLists.txt`
  in the same change.
- ALWAYS add or update unit tests in `tests/` for new behaviour. If a feature
  is cross-module, also extend `tests/smoke_runner.cpp` with a stage.
- Significant changes (new module, new dep, architecture pivot, build/setup
  changes) MUST update `README.md` and this file.
- Streaming chat-completions requests (`"stream": true`) must preserve
  `text/event-stream` framing and must not read from or write to `PromptCache`.
- Keep `main.cpp` thin - it only wires modules together and runs the loop.
- Latency is a feature. Prefer batched / mmap / zero-copy paths over clever
  abstractions. Profile before optimising; benchmark via `benchmark_runner`.
