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
  `PromptCache` (SQLite, xxhash64 of `(model, prompt, sorted(chunk_ids))`,
  optional TTL), `RepoIndex` wiring `FileWatcher` to incremental
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
  retrieval expander (decayed neighbour scores), `StructuralQueryEngine`
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
  finalised. 188 ctest cases / 45 smoke stages pass.
- **Phase 6 (next):** Diff-aware response patching for the `CodeEdit`
  bucket. Request unified diffs from upstream (system prompt + bucket
  template tweak), validate hunks against current file content, and
  apply them locally via a new `DiffPatcher` module. Targets ~70%
  output-token reduction on large edits. Includes a permissive parser
  that falls back to raw replacement if validation fails. May land the
  real `llama.cpp` linkage opportunistically (small instruct model for
  diff repair) but only if it stays single-binary friendly.
- **Phase 7:** Persistent + cross-repo embedding cache. Promote the
  in-memory chunk -> embedding map to a content-hash keyed
  `EmbeddingCache` backed by SQLite + xxhash64, warm-loaded at boot and
  shareable across sibling repos. Eliminates re-embedding on cold
  start; cuts indexing time on large monorepos.
- **Phase 8:** Multi-tier model routing. New `ModelRouter` consumes the
  Phase 2 intent bucket + request size + structural-graph hints and
  picks `cheap` / `medium` / `frontier` upstream per turn. Config:
  `model_routes: { CodeExplain: cheap, CodeEdit: medium, ... }`. Plays
  nicely with the Phase 1 `PromptCache` (cache key already includes
  model id).
- **Phase 9:** Telemetry-driven prompt evolution. Offline analyser over
  `ProxyMetrics` + completion logs proposes per-bucket template and
  budget tweaks. `PromptTemplates` gains versioning + an A/B harness
  so changes can be rolled out per-request fraction.
- **Phase 10:** Team mode. Shared `PromptCache` + `VectorStore` via a
  new HTTP sync endpoint (`/sync/cache`, `/sync/vectors`) so a team's
  proxies can pool warm context. Auth lands in Phase 12; until then,
  loopback / private network only.
- **Phase 11:** Streaming-aware compaction. Token-streaming upstream
  proxy that compacts mid-stream for long chats by rolling completed
  turns into a summary stored in `ChatHistoryStore`. Keeps long
  sessions inside the model's effective context window.
- **Phase 12:** Production hardening. HMAC / bearer auth on the proxy +
  MCP surfaces, rate limiting, OpenTelemetry tracing, a slim Docker
  image, Helm chart for the team-mode sync service, signed release
  binaries, and a `--health` self-check command.

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
- Keep `main.cpp` thin - it only wires modules together and runs the loop.
- Latency is a feature. Prefer batched / mmap / zero-copy paths over clever
  abstractions. Profile before optimising; benchmark via `benchmark_runner`.
