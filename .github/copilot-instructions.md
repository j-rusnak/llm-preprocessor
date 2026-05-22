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
- **Phase 2 (next):** Project card + per-bucket prompt templates (`inja`),
  toggleable prompt optimiser.
- **Phase 3:** Symbol graph (tree-sitter + clangd) with graph-aware expansion;
  zero-LLM fast path for structural queries.
- **Phase 4:** MCP server mode + VS Code extension; single-binary distribution.
- **Phase 5:** Optional local small-LLM prompt rewriter via `llama.cpp`.

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
  - Phase-2+ additions (when introduced): `tree-sitter` + grammars, `inja`,
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
