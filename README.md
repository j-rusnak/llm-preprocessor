# LLM Preprocessor

A high-performance C++17 **middleware for AI coding assistants**. It sits between the
IDE/agent and the LLM API to (1) cut token spend, (2) reduce latency, and
(3) act as a smart, local code-context engine — chunking source files, embedding
them, and serving the smallest possible slice of context per prompt instead of
letting the agent re-read entire files.

A legacy command-routing path (semantic intent matching for OS actions) is
preserved as a side feature.

## Project Status

**Phase 0 (Foundation Fixes) - complete.** The codebase has been re-architected
around the new direction:

- `MemoryEngine` split into `ChatHistoryStore` (SQLite) and `VectorStore`
  (HNSW ANN index over code-chunk embeddings).
- Real ANN backend via [`hnswlib`](https://github.com/nmslib/hnswlib).
- Content-addressed chunking with [`xxhash`](https://github.com/Cyan4973/xxHash).
- Cross-platform filesystem watching via [`efsw`](https://github.com/SpartanJ/efsw).
- True batched ONNX inference; the legacy 15-window cap in `IntentRouter` is gone.
- Downstream-LLM token budgeting via `ILLMTokenizer` (heuristic backend now).
- AST-aware `IChunker` interface with a `LineWindowChunker` fallback.

**Phase 1 (MVP RAG proxy) - complete.** A drop-in OpenAI-compatible local
proxy now sits between your IDE/agent and the upstream LLM:

- `BraceAwareChunker` - language-agnostic AST-ish chunker that respects
  brace depth modulo comments/strings (C, C++, JS, Java, Rust, ...). The
  tree-sitter backend will slot in behind the same `IChunker` interface in a
  later iteration.
- `BM25Index` - Okapi BM25 ranker with identifier-aware tokenisation
  (`snake_case` + `camelCase` splitting).
- `HybridRetriever` - fuses `VectorStore` ANN hits with BM25 hits via
  Reciprocal Rank Fusion (RRF, `k=60`).
- `PromptCache` - SQLite-backed cache keyed by
  `xxhash64(model || compiled_upstream_request || sorted(chunk_ids))` with optional TTL.
- `ProxyMetrics` - lock-free atomic counters for requests, cache hits,
  upstream calls, denials, upstream errors, stream cancellations, and
  **tokens saved** (compiled vs original).
- `RepoIndex` - wires `BraceAwareChunker` + embedder + `VectorStore` +
  `BM25Index` + `HybridRetriever` and watches the repo via `FileWatcher`
  for incremental re-indexing.
- `OpenAIProxy` - cpp-httplib server exposing `POST /v1/chat/completions`,
  `GET /healthz`, and `GET /stats`. Forwards to the configured upstream
  with libcurl, injecting retrieved context as a system message before the
  last user message.
- `main --serve config.json` boots the full pipeline.

**Phase 2 (Project card + per-bucket templates) - complete.** The proxy
now optimises prompts before sending them upstream:

- `ProjectCard` - lightweight repository summary (extension histogram,
  top symbols, README excerpt) built from the live `RepoIndex`.
- `HeuristicIntentClassifier` - cheap, allocation-light router that
  buckets each user turn into `CodeEdit`, `CodeExplain`, `CodeGenerate`,
  `MetaQuery`, or `Freeform`.
- `PromptTemplates` - [`inja`](https://github.com/pantor/inja)-rendered,
  per-bucket prompt scaffolds with sensible built-ins and an optional
  JSON override file (`prompt_templates_path`).
- `PromptOptimizer` - toggleable (`prompt_optimizer_enabled`) pipeline
  that classifies the turn, builds the context block to a char budget,
  optionally injects the project card, and renders the bucket's template
  as the system message. Disabled = exact Phase 1 behaviour.

**Phase 3 (Symbol graph + zero-LLM fast path) - complete.** The proxy now
builds a lightweight symbol graph as it indexes the repo, and can answer
purely structural questions without forwarding to the upstream LLM:

- `SymbolGraph` - thread-safe definitions / references store keyed by
  chunk id and file path, with one-hop neighbour expansion.
- `ISymbolExtractor` + `RegexSymbolExtractor` - pluggable extractor
  interface (tree-sitter slots in behind this in a future phase) plus a
  regex-based default covering C/C++/Java/JS/Python/Rust/Go and C macros,
  with comment/string stripping and reserved-word filtering.
- `GraphAwareRetriever::expand_with_graph` - ranks graph-reachable neighbour
  chunks by query-symbol match, reference count, symbol kind, and deterministic
  tie-breaks before context assembly.
- `StructuralQueryEngine::try_answer` - zero-LLM fast path for queries
  like "where is `Foo`", "what calls `bar`", "functions in `file.cpp`",
  and "repo stats"; on a hit the proxy synthesises an OpenAI-compatible
  completion locally and never touches the upstream.

**Phase 4 (MCP server + VS Code extension) - complete.** The same binary
now speaks both the OpenAI HTTP protocol (`--serve`) and the Model
Context Protocol over stdio (`--mcp`), so editors and agents can consume
the RAG stack directly:

- `McpServer` - JSON-RPC 2.0 over newline-delimited stdio. Surfaces
  three tools (`search_repo`, `structural_query`, `get_chunk`) and two
  resources (`repo://card`, `repo://stats`). Wires `RepoIndex`,
  `SymbolGraph`, `StructuralQueryEngine`, and `ProjectCard` together
  with no LLM in the loop.
- `preprocessor_app --mcp` - single-binary distribution: pick `--serve`
  for the HTTP proxy or `--mcp` for the stdio MCP server at startup.
- [`vscode-extension/`](vscode-extension/) - minimal TypeScript shim
  that registers the binary as a local MCP server with VS Code's
  Language Model host (VS Code 1.99+).

**Phase 5 (prompt rewriter / context compressor) - complete.** The proxy
now post-processes the assembled system context immediately before
forwarding upstream:

- `IPromptRewriter` interface with two implementations:
  - `HeuristicCompressionRewriter` (always on, dependency-free) - strips
    `//` / `#` line comments and `/* ... */` block comments while
    preserving string literals **and** C preprocessor directives,
    collapses runs of blank lines, dedupes adjacent duplicates, trims
    trailing whitespace, and applies an optional hard char cap with a
    `... [truncated]` marker.
  - `LlamaCppRewriter` (stub) gated behind the CMake option
    `LLM_PREPROCESSOR_WITH_LLAMA_CPP` (default `OFF`). The interface,
    config, and `is_available()` probe ship in this phase; the actual
    `llama.cpp` linkage lands in Phase 6 once the model story is
    finalised.
- `OpenAIProxy::set_prompt_rewriter` runs after retrieval / template
  rendering and before cache-key computation, so compressed context
  participates in caching too. Failures are non-fatal (keeps the
  uncompressed block).
- New config keys: `prompt_rewriter_enabled` (default `false`),
  `prompt_rewriter_kind` (`"heuristic"` or `"llama-cpp"`),
  `prompt_rewriter_max_chars` (`0` = inherit `max_context_chars`),
  `llama_model_path`.

Upcoming phases (diff-aware response patching, persistent embedding
cache, multi-tier model routing, telemetry-driven prompt evolution, team
mode, streaming-aware compaction, production hardening) are tracked in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md).

**Phases 6-12 (advanced middleware) - complete.** Seven additional
modules round out the production stack:

- **Phase 6 - `DiffPatcher`.** Permissive unified-diff parser + applier
  for the `CodeEdit` bucket. Validates hunk context against current
  file contents and applies changes in memory; lets upstream return
  small diffs instead of whole files. Falls back gracefully on
  malformed input.
- **Phase 7 - `EmbeddingCache`.** SQLite + xxhash64 cache keyed by
  `(model_id, content)` so chunk embeddings survive process restarts
  and are shareable across sibling repos. Eliminates re-embedding on
  cold start.
- **Phase 8 - `ModelRouter`.** Picks a `cheap` / `medium` / `frontier`
  upstream per turn from intent bucket + request size. Tiers and
  routes can be configured for `--serve`; the proxy rewrites the upstream
  URL, model name, optional API key, and context cap before caching/forwarding.
- **Phase 9 - `AbHarness`.** Sticky-hash A/B variant assignment for
  prompt templates and rewriter knobs. Deterministic per
  `(experiment_id, sticky_key)` via xxhash64; tracks hit counts.
- **Phase 10 - `SyncEndpoint`.** Transport-agnostic serializer for
  cache + vector bundles so a team's proxies can pool warm context.
  HTTP sync wiring is a thin wrapper around `to_json` / `from_json`
  and `apply_to_cache`.
- **Phase 11 - `StreamingCompactor`.** Folds older completed chat
  turns into a rolling summary so long sessions stay under the
  model's effective context window. Pluggable via the same
  `IPromptRewriter` style surface.
- **Phase 12 - `AuthMiddleware` + `RateLimiter`.** Bearer-token
  allow-list and HMAC-SHA256 (timestamp + body) auth, with optional
  token-bucket rate limiting per caller key. Self-check via new
  `--health` flag.

## Architecture

```
IDE / Agent prompt
        |
        v
TextSanitizer ---> Tokenizer ---> EmbeddingEngine (ONNX Runtime, batched)
        |                                  |
        |                                  v
        |                            IntentRouter (slash-commands /
        |                            meta queries; optional)
        |
        v
CodeChunker (line-window now, tree-sitter later)
        |
        v
VectorStore (HNSW + xxhash IDs)  <-- FileWatcher (efsw) keeps it fresh
        |
        v
PromptCompiler + ChatHistoryStore + ILLMTokenizer (budget enforcement)
        |
        v
JSON payload (OpenAI-compatible) for the upstream LLM
```

## Pipeline Modules

| Module | Header | Description |
|---|---|---|
| **ConfigLoader** | `config_loader.hpp` | Loads and validates JSON configuration. |
| **TextSanitizer** | `text_sanitizer.hpp` | Normalizes input - lowercases, collapses whitespace, trims. |
| **Tokenizer** | `tokenizer.hpp` | WordPiece tokenizer for BERT-class embedders. |
| **EmbeddingEngine** | `embedding_engine.hpp` | ONNX Runtime inference with **true batched** mean-pooled embeddings (`.onnx` / `.ort`). |
| **IntentRouter** | `intent_router.hpp` | Cosine-similarity routing for slash-commands / OS actions. Batched sub-phrase search (no 15-window cap). |
| **ContextGatherer** | `context_gatherer.hpp` | URL fetch via libcurl (HTTP/HTTPS, 10 MB cap). |
| **ChatHistoryStore** | `chat_history_store.hpp` | SQLite chat-turn store. Replaces the old `MemoryEngine`. |
| **VectorStore** | `vector_store.hpp` | Persistent HNSW ANN index keyed by 64-bit chunk IDs (xxhash). |
| **CodeChunker** | `code_chunker.hpp` | `IChunker` interface + `LineWindowChunker` and `BraceAwareChunker`. |
| **FileWatcher** | `file_watcher.hpp` | RAII wrapper around `efsw` for incremental re-indexing. |
| **LLMTokenizer** | `llm_tokenizer.hpp` | Downstream-LLM token budgeter (`HeuristicLLMTokenizer` and model-family calibrated mode). |
| **PromptCompiler** | `prompt_compiler.hpp` | Assembles the final OpenAI-style JSON payload. |
| **BM25Index** | `bm25_index.hpp` | Okapi BM25 ranker with identifier-aware tokenisation. |
| **RetrievalQuery** | `retrieval_query.hpp` | Shared query normalizer for terms, identifiers, paths, and language hints. |
| **HybridRetriever** | `hybrid_retriever.hpp` | RRF fusion of `VectorStore` + `BM25Index` hits. |
| **ContextPacker** | `context_packer.hpp` | Shared retrieved-context formatter with budget metadata for included and omitted chunks. |
| **PromptCache** | `prompt_cache.hpp` | SQLite-backed cache of upstream responses, keyed by `(model, compiled request, chunk_ids)`. |
| **ProxyMetrics** | `proxy_metrics.hpp` | Atomic counters for requests, cache hits, upstream calls, stream cancellations, tokens saved, and per-model-family token totals. |
| **RepoIndex** | `repo_index.hpp` | End-to-end chunk + embed + index over a repo, kept fresh by `FileWatcher`. |
| **OpenAIProxy** | `openai_proxy.hpp` | cpp-httplib server, OpenAI-compatible chat completions with RAG context injection. |
| **ProjectCard** | `project_card.hpp` | Repository summary (extensions, top symbols, README excerpt) derived from `RepoIndex`. |
| **IntentClassifier** | `intent_classifier.hpp` | `IIntentClassifier` interface + `HeuristicIntentClassifier` for bucket routing. |
| **PromptTemplates** | `prompt_templates.hpp` | `inja`-rendered, per-bucket prompt scaffolds; JSON-overridable. |
| **PromptOptimizer** | `prompt_optimizer.hpp` | Toggleable prompt rewriter wiring classifier + templates + project card. |
| **SymbolGraph** | `symbol_graph.hpp` | Defs/refs store + `ISymbolExtractor` + `RegexSymbolExtractor`; one-hop neighbour expansion. |
| **GraphAwareRetriever** | `graph_aware_retriever.hpp` | `expand_with_graph` adds ranked graph-reachable neighbour chunks to retrieval results. |
| **StructuralQueryEngine** | `structural_query_engine.hpp` | Zero-LLM fast path for definition / caller / file-symbols / repo-stats queries. |
| **McpServer** | `mcp_server.hpp` | JSON-RPC 2.0 MCP server over stdio; exposes RAG + structural surfaces as tools/resources. |
| **PromptRewriter** | `prompt_rewriter.hpp` | `IPromptRewriter` + `HeuristicCompressionRewriter` (always on) and `LlamaCppRewriter` (stub; enabled by `LLM_PREPROCESSOR_WITH_LLAMA_CPP`). |
| **DiffPatcher** | `diff_patcher.hpp` | Phase 6 permissive unified-diff parser + applier (context-validated). |
| **EmbeddingCache** | `embedding_cache.hpp` | Phase 7 persistent SQLite + xxhash64 chunk-embedding cache. |
| **ModelRouter** | `model_router.hpp` | Phase 8 multi-tier upstream selector keyed on intent bucket + request size. |
| **AbHarness** | `ab_harness.hpp` | Phase 9 sticky-hash A/B variant assignment for telemetry-driven prompt evolution. |
| **SyncEndpoint** | `sync_endpoint.hpp` | Phase 10 transport-agnostic serializer for cache + vector bundles (team mode). |
| **StreamingCompactor** | `streaming_compactor.hpp` | Phase 11 rolling chat-history summarizer; keeps long sessions under the context window. |
| **AuthMiddleware** | `auth_middleware.hpp` | Phase 12 bearer + HMAC-SHA256 request authentication. |
| **RateLimiter** | `rate_limiter.hpp` | Phase 12 per-key token-bucket rate limiter. |
| **EffectivenessRunner** | `benchmarks/effectiveness_runner.cpp` | Standalone harness that measures cache speedup, rewriter compression, BM25/graph retrieval quality, A/B determinism, HMAC throughput, and rate-limit burst behaviour. Emits JSON for CI dashboards. |

## Tech Stack

- **C++17** (strictly enforced)
- **CMake 3.15+** with **vcpkg** manifest mode
- **ONNX Runtime 1.23.2** - local embedding inference (pre-built binary)
- **libcurl**, **SQLite3**, **nlohmann/json**
- **hnswlib** - ANN index
- **xxHash** - content-addressed chunk IDs
- **efsw** - cross-platform file watching
- **cpp-httplib** - embedded HTTP server for the OpenAI-compatible proxy
- **inja** - Jinja2-style template engine for per-bucket prompt scaffolds
- **Google Test** - unit testing

## Project Structure

```
├── CMakeLists.txt
├── vcpkg.json
├── config.json
├── include/
│   ├── chat_history_store.hpp
│   ├── code_chunker.hpp
│   ├── config_loader.hpp
│   ├── context_gatherer.hpp
│   ├── embedding_engine.hpp
│   ├── file_watcher.hpp
│   ├── i_embedding_engine.hpp
│   ├── intent_router.hpp
│   ├── llm_tokenizer.hpp
│   ├── prompt_compiler.hpp
│   ├── text_sanitizer.hpp
│   ├── tokenizer.hpp
│   └── vector_store.hpp
├── src/
│   ├── main.cpp
│   └── (one .cpp per header above)
├── tests/
│   ├── smoke_runner.cpp           (Phase 0 integration framework)
│   ├── test_chat_history_store.cpp
│   ├── test_code_chunker.cpp
│   ├── test_config_loader.cpp
│   ├── test_context_gatherer.cpp
│   ├── test_file_watcher.cpp
│   ├── test_intent_router.cpp
│   ├── test_llm_tokenizer.cpp
│   ├── test_prompt_compiler.cpp
│   ├── test_text_sanitizer.cpp
│   ├── test_tokenizer.cpp
│   └── test_vector_store.cpp
├── benchmarks/
├── models/
└── onnxruntime-win-x64-1.23.2/
```

## Prerequisites

- A C++17 compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
- [CMake 3.15+](https://cmake.org/)
- [vcpkg](https://vcpkg.io/) — package manager
- **ONNX Runtime 1.23.2** — download the [official pre-built release](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2) and extract it to the project root (CMake auto-selects the platform-appropriate directory name)
- A **BERT-based ONNX embedding model** (e.g., `all-MiniLM-L6-v2`)

## Setting Up the Model

The preprocessor requires a sentence-embedding ONNX model and its WordPiece vocabulary.

```bash
# Install Python dependencies
pip install optimum[onnxruntime] sentence-transformers

# Export model to ONNX format
python -c "
from optimum.onnxruntime import ORTModelForFeatureExtraction
m = ORTModelForFeatureExtraction.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', export=True)
m.save_pretrained('models')
"

# Download the vocabulary file
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
t.save_vocabulary('models')
"
```

This creates `models/model.onnx` (~80 MB) and `models/vocab.txt` (~232 KB).

> **Tip:** Add `models/` to your `.gitignore` — don't commit large binary files.

Runtime assets such as `models/`, `onnxruntime-*/`, `*.zip`, and local SQLite
databases are ignored by git. Keep them local or provide them through your
release process instead of committing generated binaries.

For proxy deployments, start from [`config.example.json`](config.example.json)
and replace the local proxy token and upstream key placeholders before serving.

## Build

Building requires a **Visual Studio Developer Command Prompt** (or equivalent) so the MSVC environment variables are set.

```powershell
# Open a VS Developer PowerShell, then:
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

The build copies the ONNX Runtime runtime DLLs next to the generated
executables on Windows.

## Install / Package Smoke Check

The CMake install target installs the app, library, headers, CMake package
files, and ONNX Runtime redistributables into the chosen prefix:

```powershell
cmake --install build --prefix build\install
Get-ChildItem build\install\lib\cmake\LLMPreprocessor
Get-ChildItem build\install\bin\onnxruntime*.dll
```

Consumers can use the installed package with:

```cmake
find_package(LLMPreprocessor CONFIG REQUIRED)
target_link_libraries(my_tool PRIVATE LLMPreprocessor::preprocessor_lib)
```

## Testing

Three complementary surfaces:

### 1. Unit tests (Google Test)

Fast, hermetic per-module tests. Run via CTest:

```powershell
cd build
ctest --output-on-failure
```

Or the binary directly with filtering:

```powershell
.\build\preprocessor_tests.exe --gtest_filter=VectorStoreTest.*
```

### 2. Integration smoke runner (Phase 0 framework)

`tests/smoke_runner.cpp` exercises every new module end-to-end against a
synthetic in-memory "repo". Uses a deterministic hash-based fake embedder so
it does NOT require the ONNX model and stays sub-second.

```powershell
.\build\smoke_runner.exe            # PASS/FAIL summary
.\build\smoke_runner.exe --verbose  # per-stage detail
```

Exit code is `0` on full pass, non-zero on any failure - safe to wire into CI.

### 3. Benchmark runner

Latency / accuracy charts for the semantic router. See
[Benchmarks & Visualizations](#benchmarks--visualizations).

### 4. Effectiveness runner (Phase 5-12 quality gates)

`benchmarks/effectiveness_runner.cpp` measures real, end-to-end effectiveness of
every Phase 5-12 module without requiring the ONNX model. JSON to stdout,
human-readable summary table to stderr.

```powershell
.\build\effectiveness_runner.exe > benchmarks\results\effectiveness.json
```

The matching gtest suite (`Effectiveness_*` in `tests/test_effectiveness.cpp`)
locks in minimum thresholds (cache speedup, rewriter char/token reduction,
BM25 top-1 correctness, A/B sticky + balance, rate-limiter burst, ...) so any
regression breaks `ctest`.

A complete end-to-end testing & feature-usage walkthrough lives in
[docs/TESTING.md](docs/TESTING.md).

## Configuration

The preprocessor is driven by a `config.json` file:

```json
{
    "model_path": "models/model.ort",
    "vocab_path": "models/vocab.txt",
    "db_path": "history.db",
    "system_prompt": "You are a helpful AI assistant.",
    "similarity_threshold": 0.65,
    "history_limit": 10,
    "intents": [
        {
            "name": "ACTION_DECREASE_VOLUME",
            "examples": ["turn down the volume", "lower the volume", "make it quieter"]
        },
        {
            "name": "ACTION_OPEN_BROWSER",
            "examples": ["open the web browser", "launch a browser", "start the browser"]
        }
    ]
}
```

Each intent supports multiple synonym examples via the `"examples"` array. The router registers every example as a separate embedding — the best match across all examples determines the intent. A single `"example"` string is also accepted for backward compatibility.

| Key | Description | Default |
|---|---|---|
| `model_path` | Path to the ONNX/ORT embedding model | *(required)* |
| `vocab_path` | Path to the WordPiece vocab file | *(required)* |
| `db_path` | SQLite database file for conversation history | `"history.db"` |
| `system_prompt` | System message prepended to every LLM payload | `"You are a helpful assistant."` |
| `similarity_threshold` | Cosine similarity cutoff for intent matching (0.0–1.0) | `0.65` |
| `history_limit` | Max conversation turns to include in payload | `10` |
| `intents` | Array of `{name, examples}` objects for semantic routing | `[]` |
| `api_model` | *(Optional)* Model name for complete API payload (e.g., `"gpt-4"`) | — |
| `api_endpoint` | *(Optional)* API endpoint URL | — |
| `temperature` | *(Optional)* Sampling temperature (0.0–2.0) | — |
| `max_tokens` | *(Optional)* Max tokens in LLM response | — |

## Running

```powershell
.\build\preprocessor_app.exe                       # interactive REPL, uses config.json
.\build\preprocessor_app.exe my_config.json        # custom config path
.\build\preprocessor_app.exe --serve config.json   # start OpenAI-compatible RAG proxy
.\build\preprocessor_app.exe --help                # show usage
.\build\preprocessor_app.exe --version             # show version
```

### Running as an OpenAI-compatible RAG proxy

`--serve` indexes `repo_root` and starts an HTTP server on `proxy_host:proxy_port`.
Point any OpenAI-compatible client (Cursor, Continue, etc.) at it:

```powershell
.\build\preprocessor_app.exe --serve config.json
# then in your IDE set the OpenAI base URL to http://127.0.0.1:8088
```

Endpoints:

- `POST /v1/chat/completions` - drop-in OpenAI chat completions; the proxy
  retrieves top-k relevant code chunks, injects them as a system message,
  forwards to `upstream_url`, caches the response by
  `(model, compiled upstream request, chunk_ids)`.
  Requests with `"stream": true` are forwarded as `text/event-stream` and are
  not cached.
- `GET /healthz` - liveness check.
- `GET /stats` - JSON snapshot of `ProxyMetrics` (tokens saved, cache hits,
  upstream calls, auth/rate/request-size denials, upstream errors, stream
  cancellations, and per-model-family token totals). Protected by proxy auth
  when auth is configured.
- `GET /sync/cache` - export a `SyncBundle` containing recent cache entries.
  Protected by proxy auth when auth is configured.
- `POST /sync/cache` - import cache entries from a peer `SyncBundle`.
  Protected by proxy auth when auth is configured.
- `GET /sync/vectors` - export a `SyncBundle` containing indexed vector
  entries plus chunk metadata. Protected by proxy auth when auth is configured.
- `POST /sync/vectors` - import vector entries and hydrate them into the local
  retrieval index. Protected by proxy auth when auth is configured.

Phase 1 config keys (in addition to the Phase 0 ones):

| Key | Description | Default |
|---|---|---|
| `proxy_host` | Bind address for `--serve` | `"127.0.0.1"` |
| `proxy_port` | Bind port for `--serve` | `8088` |
| `repo_root` | Directory to chunk + index on startup | *(optional)* |
| `cache_db_path` | SQLite file backing `PromptCache` | `"prompt_cache.db"` |
| `retrieval_k` | Top-k chunks injected per request | `6` |
| `embedding_dim` | Must match the embedder | `384` |
| `max_context_chars` | Cap on injected context | `8000` |
| `upstream_url` | OpenAI-compatible URL to forward to | `https://api.openai.com/v1/chat/completions` |
| `upstream_api_key` | Fallback bearer token if the client did not send one | — |
| `upstream_timeout_seconds` | Overall upstream libcurl timeout | `60` |
| `upstream_connect_timeout_seconds` | Upstream connection timeout | `10` |
| `upstream_max_response_bytes` | Max buffered non-streaming upstream response size (`0` = disabled) | `0` |
| `stream_idle_timeout_seconds` | Streaming idle timeout (`0` = disabled) | `0` |
| `tokenizer_mode` | Token estimator for budgets and telemetry: `"heuristic"` or `"model-calibrated"` | `"heuristic"` |
| `model_tiers` | Optional array of `{name, upstream_url, model_name, api_key, max_context}` tier definitions | `[]` |
| `model_routes` | Optional ordered array of `{bucket, min_request_chars, max_request_chars, tier}` routing rules | `[]` |
| `proxy_auth_bearer_tokens` | Local proxy bearer-token allow-list | `[]` |
| `proxy_auth_hmac_secret` | Local HMAC-SHA256 shared secret | `""` |
| `proxy_auth_max_clock_skew_seconds` | Allowed HMAC timestamp skew | `300` |
| `proxy_rate_limit_tokens_per_second` | Per-caller proxy token refill rate (`0` = disabled) | `0` |
| `proxy_rate_limit_burst` | Per-caller proxy burst size (`0` = disabled) | `0` |
| `proxy_max_request_bytes` | Max chat-completions body size (`0` = disabled) | `8388608` |
| `sync_cache_export_limit` | Max cache entries returned by `GET /sync/cache` (`0` = unlimited) | `1000` |
| `sync_vector_export_limit` | Max vector entries returned by `GET /sync/vectors` (`0` = unlimited) | `1000` |
| `proxy_forward_client_authorization` | Forward client `Authorization` to upstream; defaults to `false` when local auth is configured unless set explicitly | `true` |
| `allow_unsafe_remote_proxy` | Permit non-loopback unauthenticated serving | `false` |
| `prompt_optimizer_enabled` | Enable Phase 2 per-bucket prompt rewriting | `false` |
| `prompt_templates_path` | Optional JSON file overriding bucket templates | *(empty)* |
| `include_project_card` | Inject the `ProjectCard` summary into the system prompt | `true` |
| `symbol_graph_enabled` | Build Phase 3 symbol graph during indexing | `false` |
| `graph_expansion_enabled` | Append graph-reachable neighbour chunks to retrieval (requires `symbol_graph_enabled`) | `true` |
| `structural_fast_path_enabled` | Answer structural queries locally without forwarding upstream (requires `symbol_graph_enabled`) | `true` |
| `prompt_rewriter_enabled` | Apply Phase 5 prompt rewriter to assembled context before forwarding | `false` |
| `prompt_rewriter_kind` | `"heuristic"` (always available) or `"llama-cpp"` (requires `LLM_PREPROCESSOR_WITH_LLAMA_CPP`) | `"heuristic"` |
| `prompt_rewriter_max_chars` | Soft char cap for the rewriter (`0` = inherit `max_context_chars`) | `0` |
| `llama_model_path` | Path to a `.gguf` model when `prompt_rewriter_kind == "llama-cpp"` | — |

By default, `proxy_host` is loopback-only. Binding to `0.0.0.0`, a LAN IP, or
another non-loopback address requires local proxy auth
(`proxy_auth_bearer_tokens` or `proxy_auth_hmac_secret`) unless the explicit
`allow_unsafe_remote_proxy=true` override is set. Non-loopback configs also
reject the example placeholder token and require a positive
`proxy_max_request_bytes` unless the unsafe override is set. Local bearer auth
accepts `X-Preprocessor-Authorization: Bearer <token>` or
`Authorization: Bearer <token>`. Prefer the `X-Preprocessor-*` headers when the
client also needs to send an upstream provider key in `Authorization`.

[`config.example.json`](config.example.json) shows a production-oriented local
proxy starter config with loopback binding, local bearer auth, rate limiting,
request-size limits, model-calibrated metrics, symbol graph retrieval, and the
heuristic prompt rewriter enabled. Do not expose a non-loopback proxy without
real local proxy auth unless you intentionally set
`allow_unsafe_remote_proxy=true`.

Multi-tier routing is ordered, first-match wins. Buckets are `code_edit`,
`code_explain`, `code_generate`, `meta_query`, and `freeform`; request-size
bounds are character counts from the incoming request body. Tier `upstream_url`,
`api_key`, and `max_context` are optional overrides; `model_name` is required.

```json
{
  "model_tiers": [
    {"name": "cheap", "model_name": "gpt-4o-mini"},
    {"name": "frontier", "model_name": "gpt-4.1", "api_key": "frontier-key"}
  ],
  "model_routes": [
    {"bucket": "code_explain", "max_request_chars": 12000, "tier": "cheap"},
    {"bucket": "code_edit", "tier": "frontier"}
  ]
}
```

When `api_model` is set in config, payloads are emitted as complete API request bodies (`{model, messages, temperature, max_tokens}`). Without it, the old messages-only format is used.

The interactive loop accepts free-text input:

```
LLM Preprocessor ready. Type your input (or 'quit' to exit).

> open the web browser
[ACTION] ACTION_OPEN_BROWSER

> what is the meaning of life?

=== LLM Payload ===
[
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "what is the meaning of life?"}
]

> quit
```

- Inputs matching a registered intent trigger a **local action** (no LLM call).
- Unmatched inputs produce a **JSON payload** for the host application to forward to an LLM.
- The terminal display shows a clean payload without conversation history to reduce clutter; the full history is still included when the payload is sent to the LLM.
- URLs in the input are automatically fetched and injected as RAG context.
- If model files are missing, semantic routing is gracefully disabled and only the payload path is active.

## Usage as a Library

```cpp
#include "config_loader.hpp"
#include "intent_router.hpp"
#include "embedding_engine.hpp"
#include "tokenizer.hpp"
#include "prompt_compiler.hpp"
#include "chat_history_store.hpp"
#include "context_gatherer.hpp"
#include "text_sanitizer.hpp"

// Load config
auto config = preprocessor::ConfigLoader::load("config.json");

// Initialize pipeline
auto tokenizer = std::make_shared<preprocessor::Tokenizer>(config.vocab_path);
auto engine = std::make_shared<preprocessor::EmbeddingEngine>(config.model_path, tokenizer);
preprocessor::IntentRouter router(config.similarity_threshold, engine);
preprocessor::ChatHistoryStore history_store(config.db_path);
preprocessor::PromptCompiler compiler(config.system_prompt);

// Register intents (multiple synonym examples per intent)
for (const auto& [name, examples] : config.intents) {
    for (const auto& example : examples) {
        router.add_intent(name, example);
    }
}

// Process input
std::string input = preprocessor::TextSanitizer::sanitize(raw_input);
auto matched = router.route(input);
if (matched) {
    // Handle locally — no LLM call needed
} else {
    auto urls = preprocessor::ContextGatherer::extract_urls(raw_input);
    std::string context;
    for (const auto& url : urls) {
        context += preprocessor::ContextGatherer::fetch_url(url);
    }
    auto history = history_store.get_recent_history(config.history_limit);
    std::string payload = compiler.build_payload(input, context, history);
    // Send payload to your LLM...
}
```

## Benchmarks & Visualizations

The project includes a full benchmarking and visualization pipeline for evaluating the semantic router's latency, accuracy, and similarity characteristics. This is useful for presentations, reports, and tuning.

### Overview

| Component | Location | Purpose |
|---|---|---|
| **Benchmark Runner** | `benchmarks/benchmark_runner.cpp` | C++ executable that runs 31 test prompts through the routing pipeline, collecting latency, accuracy, similarity scores, and embedding timing. Outputs structured JSON. |
| **Visualization Script** | `benchmarks/visualize.py` | Python script that reads the JSON output and generates 10 presentation-ready PNG charts. |
| **Orchestration Script** | `benchmarks/run_benchmarks.ps1` | PowerShell script that runs both steps end-to-end. |
| **Results Directory** | `benchmarks/results/` | Output directory for JSON data and PNG charts. |

### Prerequisites

Ensure the project is already built (see [Build](#build) above), then install the Python dependencies:

```bash
pip install matplotlib numpy
```

### Quick Start (All-In-One)

From the project root, run the PowerShell orchestration script:

```powershell
.\benchmarks\run_benchmarks.ps1
```

This will:
1. Verify `benchmark_runner.exe` exists in `build/`
2. Run the C++ benchmark → `benchmarks/results/benchmark_data.json`
3. Generate all 10 PNG charts → `benchmarks/results/`

### Step-by-Step (Manual)

#### 1. Build the benchmark executable

The benchmark target is included in the CMake build. If you haven't built yet:

```powershell
# From a VS Developer PowerShell:
cmake -B build -G Ninja `
    -DCMAKE_BUILD_TYPE=Debug `
    -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"

cmake --build build
```

Verify the executable exists:

```powershell
Test-Path .\build\benchmark_runner.exe   # should be True
```

#### 2. Run the benchmark

The benchmark must be run from the **project root** so it can find `config.json` and `models/`:

```powershell
# Create results directory
New-Item -ItemType Directory -Path benchmarks\results -Force | Out-Null

# Run and capture JSON output
.\build\benchmark_runner.exe > benchmarks\results\benchmark_data.json
```

The benchmark runs 5 phases:
1. **Routing Benchmarks** — 31 test inputs × 10 runs each, measuring latency and correctness
2. **Similarity Analysis** — per-input similarity scores against all 8 intents
3. **Intent Similarity Matrix** — 8×8 cosine similarity between intents
4. **Embedding Timing** — 9 different input lengths × 20 runs each
5. **Tokenization Timing** — encoding speed for the same 9 inputs

Progress is printed to `stderr`; JSON data goes to `stdout`.

#### 3. Generate visualizations

```powershell
python benchmarks\visualize.py benchmarks\results\benchmark_data.json benchmarks\results
```

Arguments:
- **Arg 1** (required): Path to the JSON data file
- **Arg 2** (optional): Output directory for PNGs (default: `benchmarks/results`)

### Generated Charts

| # | File | Chart Type | What It Shows |
|---|------|-----------|---------------|
| 1 | `01_latency_by_category.png` | Bar chart (± std) | Average routing latency across 5 input categories |
| 2 | `02_latency_vs_words.png` | Scatter + trend line | How routing latency scales with input word count |
| 3 | `03_accuracy_by_category.png` | Bar chart | Percentage of correctly routed inputs per category |
| 4 | `04_score_distribution.png` | Box plot | Distribution of best cosine similarity scores, with threshold overlay |
| 5 | `05_similarity_heatmap.png` | Heatmap (8×8) | Cosine similarity between all registered intents (with formula) |
| 6 | `06_threshold_curve.png` | Multi-line plot | Accuracy, Precision, Recall, and F1 across thresholds 0.40–0.95 |
| 7 | `07_embedding_timing.png` | Bar chart (± error) | ONNX embedding generation time vs input length |
| 8 | `08_api_comparison.png` | Log-scale bars | Local routing latency vs estimated LLM API round-trip times |
| 9 | `09_per_input_scores.png` | Heatmap (31×8) | Every test input's similarity score against every intent |
| 10 | `10_summary_dashboard.png` | Text dashboard | Configuration, accuracy, latency (P50/P95), throughput, optimal threshold |

### Test Categories

The 31 benchmark inputs span 5 complexity levels:

| Category | Count | Examples |
|---|---|---|
| **Direct Commands** | 8 | `"mute the sound"`, `"turn up the volume"`, `"open a file"` |
| **Noisy Commands** | 6 | `"hey bro can you mute that"`, `"yo dude turn up the volume please"` |
| **Complex Sentences** | 6 | `"i was wondering if you could perhaps mute the audio"` |
| **Non-Matching** | 6 | `"what is the meaning of life"`, `"tell me about quantum physics"` |
| **Edge Cases** | 5 | `"mute"` (1 word), `"volume"`, `"mute the sound and open the browser"` (multi-intent) |

### Benchmark JSON Schema

The JSON output contains these top-level sections:

```
{
  "config":                  { ... },  // threshold, num_intents, num_runs, etc.
  "routing_results":         [ ... ],  // 31 entries with latency, correctness, scores
  "similarity_analysis":     [ ... ],  // per-input scores against all intents
  "intent_similarity_matrix": { ... }, // 8×8 matrix with labels
  "embedding_timing":        [ ... ],  // 9 entries with avg/min/max/p50
  "tokenization_timing":     [ ... ]   // 9 entries with timing + token count
}
```

Each routing result includes: `category`, `input`, `expected_intent`, `matched_intent`, `score`, `correct`, `word_count`, and `latency` object with `avg_us`, `min_us`, `max_us`, `p50_us`, `p95_us`.

## Running the Test Suite

### Quick Run

```powershell
cd build
ctest --output-on-failure
```

### Verbose Output

```powershell
cd build
ctest --output-on-failure -V
```

### Run a Single Test Suite

```powershell
cd build
.\preprocessor_tests.exe --gtest_filter="TextSanitizerTest.*"
.\preprocessor_tests.exe --gtest_filter="IntentRouterTest.*"
.\preprocessor_tests.exe --gtest_filter="ChatHistoryStoreTest.*"
.\preprocessor_tests.exe --gtest_filter="PromptCompilerTest.*"
.\preprocessor_tests.exe --gtest_filter="UrlExtractionTest.*"
.\preprocessor_tests.exe --gtest_filter="ConfigLoaderTest.*"
.\preprocessor_tests.exe --gtest_filter="TokenizerTest.*"
.\preprocessor_tests.exe --gtest_filter="EmbeddingEngineTest.*"
```

### Run a Single Test

```powershell
.\preprocessor_tests.exe --gtest_filter="ConfigLoaderTest.LoadValidConfig"
```

### List All Tests

```powershell
.\preprocessor_tests.exe --gtest_list_tests
```

### Test Suites

| Suite | Tests | What It Covers |
|---|---|---|
| **TextSanitizerTest** | 5 | Whitespace collapsing, case normalization, trimming |
| **IntentRouterTest** | 5 | Cosine similarity routing, edge cases, empty/identical embeddings |
| **ChatHistoryStoreTest** | 10 | SQLite CRUD, ordering, history limits, move semantics, update, clear, prune |
| **PromptCompilerTest** | 7 | JSON payload construction, `build_payload_json`, API params |
| **UrlExtractionTest** | 6 | URL detection in text (http/https, mixed content) |
| **ConfigLoaderTest** | 29 | Config validation, defaults, multi-example parsing, proxy safety, packaging example, model routing |
| **TokenizerTest** | 12 | WordPiece encoding, special tokens, truncation, subwords |
| **EmbeddingEngineTest** | 13 | Shape, normalization, similarity, multi-example routing, sliding-window, stop-words |

Use `ctest --test-dir build --output-on-failure` or
`.\build\preprocessor_tests.exe --gtest_list_tests` for the authoritative
current test list.

## Complete Workflow Reference

A full build-test-benchmark-visualize cycle from a clean state:

```powershell
# 0. Open a VS Developer PowerShell (MSVC environment)
& "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

# 1. Configure
cmake -B build -G Ninja `
    -DCMAKE_BUILD_TYPE=Debug `
    -DCMAKE_TOOLCHAIN_FILE="C:/Users/you/vcpkg/scripts/buildsystems/vcpkg.cmake"

# 2. Build everything (app + tests + benchmark)
cmake --build build

# 3. Run the test suite
cd build
ctest --output-on-failure
cd ..

# 4. Run an install/package smoke check
cmake --install build --prefix build\install

# 5. Run the benchmark
New-Item -ItemType Directory -Path benchmarks\results -Force | Out-Null
.\build\benchmark_runner.exe > benchmarks\results\benchmark_data.json

# 6. Generate visualizations
pip install matplotlib numpy   # first time only
python benchmarks\visualize.py benchmarks\results\benchmark_data.json benchmarks\results

# 7. Run the application
.\build\preprocessor_app.exe config.json
```

## License

See [LICENSE](LICENSE) for details.
