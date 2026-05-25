# LLM Preprocessor

`llm-preprocessor` is a C++17 local middleware for AI coding agents. It sits
between an IDE/agent and an OpenAI-compatible upstream model, indexes the local
repo, retrieves focused code context, optimizes the prompt, enforces budgets,
and forwards the request with lower token cost and latency.

The product surface is intentionally narrow:

- OpenAI-compatible HTTP proxy via `--serve`.
- MCP context server via `--mcp`.
- Health and release checks via `--health`, tests, smoke, and effectiveness
  runners.
- No local OS command routing or action execution path.

## Current Features

| Area | What ships |
|---|---|
| Repository indexing | `RepoIndex` combines chunking, embeddings, HNSW vectors, BM25, file watching, and optional symbol graph population. |
| Retrieval | `HybridRetriever` fuses vector and metadata-aware keyword hits; `RepoIndex` returns distinct files before repeated chunks; `GraphAwareRetriever` expands relevant symbol neighbours. |
| Context packing | `ContextPacker` formats retrieved chunks with dedupe, diversity, omitted-chunk metadata, and stable cache keys. |
| Prompt optimization | `PromptOptimizer` classifies coding turns into buckets, injects a project card, and renders per-bucket templates. |
| Compression | `HeuristicCompressionRewriter` removes low-signal text and caps context before upstream forwarding. |
| Protocols | `OpenAIProxy` exposes `/v1/chat/completions`, `/healthz`, `/stats`, and sync endpoints; `McpServer` exposes repo-search tools/resources over stdio. |
| Production controls | Bearer/HMAC auth, rate limiting, request-size limits, upstream timeouts, stream idle timeout, non-loopback safety checks. |
| Team mode | `SyncEndpoint` exports/imports prompt-cache entries and vector bundles. |
| Measurement | `smoke_runner`, `effectiveness_runner`, and `Effectiveness_*` tests cover end-to-end behaviour and quality floors. |
| Visualization | `tools/perf_visualizer` serves a standalone local dashboard for effectiveness samples and live proxy `/stats`. |

## Architecture

```text
AI coding agent / IDE
        |
        v
OpenAI-compatible request or MCP tool call
        |
        v
OpenAIProxy / McpServer
        |
        +--> RepoIndex
        |       +--> CodeChunker
        |       +--> EmbeddingEngine + VectorStore
        |       +--> BM25Index + HybridRetriever
        |       +--> SymbolGraph + StructuralQueryEngine
        |
        +--> PromptOptimizer + ContextPacker + PromptRewriter
        |
        +--> PromptCache / ProxyMetrics / AuthMiddleware / RateLimiter
        |
        v
OpenAI-compatible upstream model
```

Structural queries such as "where is `Foo` defined" can be answered locally by
`StructuralQueryEngine`; normal chat-completion requests are enriched with repo
context and forwarded upstream.

## Core Modules

| Module | Header | Purpose |
|---|---|---|
| `ConfigLoader` | `config_loader.hpp` | Loads production proxy/MCP config and rejects removed command-router keys. |
| `Tokenizer` / `EmbeddingEngine` | `tokenizer.hpp`, `embedding_engine.hpp` | WordPiece tokenization and ONNX embedding inference. |
| `VectorStore` | `vector_store.hpp` | HNSW ANN index keyed by content-addressed chunk ids. |
| `CodeChunker` | `code_chunker.hpp` | Line-window and brace-aware source chunkers. |
| `RepoIndex` | `repo_index.hpp` | End-to-end repo indexing, metadata-aware search, distinct-file ranking, and incremental updates. |
| `BM25Index` / `HybridRetriever` | `bm25_index.hpp`, `hybrid_retriever.hpp` | Keyword ranking over code/path/language/symbol metadata and weighted RRF fusion with vector hits. |
| `ContextPacker` | `context_packer.hpp` | Budget-aware retrieved-context formatting and telemetry. |
| `PromptCache` | `prompt_cache.hpp` | SQLite cache of upstream responses keyed by model, request, and included chunks. |
| `PromptOptimizer` | `prompt_optimizer.hpp` | Coding-prompt bucket classification, project-card injection, and template rendering. |
| `SymbolGraph` | `symbol_graph.hpp` | Definitions/references store with regex extraction and neighbour expansion. |
| `StructuralQueryEngine` | `structural_query_engine.hpp` | Zero-upstream answers for definitions, callers, file symbols, and repo stats. |
| `McpServer` | `mcp_server.hpp` | JSON-RPC MCP server exposing repo context tools/resources. |
| `OpenAIProxy` | `openai_proxy.hpp` | HTTP proxy that injects context and forwards OpenAI-compatible requests. |
| `PromptRewriter` | `prompt_rewriter.hpp` | Heuristic context compressor plus optional llama.cpp stub surface. |
| `DiffPatcher` | `diff_patcher.hpp` | Unified-diff parser/applier for diff-aware coding-agent responses. |
| `EmbeddingCache` | `embedding_cache.hpp` | Persistent per-model embedding cache. |
| `ModelRouter` | `model_router.hpp` | Ordered bucket/request-size routing to configured upstream tiers. |
| `SyncEndpoint` | `sync_endpoint.hpp` | Cache/vector bundle serialization for team mode. |
| `StreamingCompactor` | `streaming_compactor.hpp` | Long chat-history compaction primitive. |
| `AuthMiddleware` / `RateLimiter` | `auth_middleware.hpp`, `rate_limiter.hpp` | Local proxy auth and per-caller token bucket enforcement. |

## Prerequisites

- C++17 compiler.
- CMake 3.15+.
- vcpkg manifest mode.
- ONNX Runtime 1.23.2 extracted at the repo root using the platform-specific
  directory name expected by `CMakeLists.txt`.
- A BERT-style ONNX sentence-embedding model and `vocab.txt`.

Example model export:

```bash
pip install optimum[onnxruntime] sentence-transformers
python -c "
from optimum.onnxruntime import ORTModelForFeatureExtraction
m = ORTModelForFeatureExtraction.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', export=True)
m.save_pretrained('models')
"
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
t.save_vocabulary('models')
"
```

Runtime assets such as `models/`, `onnxruntime-*/`, archives, and local SQLite
caches are ignored by git and should be supplied through release packaging.

## Build

Use a Visual Studio Developer shell on Windows so MSVC environment variables are
available:

```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>\scripts\buildsystems\vcpkg.cmake
cmake --build build
```

## Running

The binary requires an explicit runtime mode. Running without `--serve`,
`--mcp`, or `--health` prints usage and exits without starting any interactive
command loop.

```powershell
.\build\preprocessor_app.exe --health config.example.json
.\build\preprocessor_app.exe --serve  config.example.json
.\build\preprocessor_app.exe --mcp    config.example.json
.\build\preprocessor_app.exe --version
```

### HTTP Proxy

`--serve` indexes `repo_root` and starts an OpenAI-compatible local proxy:

```powershell
.\build\preprocessor_app.exe --serve config.json
```

Point a coding agent at `http://127.0.0.1:<proxy_port>` and send normal
OpenAI-compatible chat-completion requests:

```powershell
curl -s http://127.0.0.1:8088/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "X-Preprocessor-Authorization: Bearer <local-token>" `
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"explain RepoIndex::index_path"}]}'
```

Streaming requests preserve SSE framing and bypass `PromptCache`. Client
disconnects cancel forwarding without writing a synthetic error trailer.

### MCP Server

`--mcp` exposes repo context over stdio:

```powershell
.\build\preprocessor_app.exe --mcp config.json
```

Tools:

- `search_repo`
- `structural_query`
- `get_chunk`

Resources:

- `repo://card`
- `repo://stats`

See [`vscode-extension/`](vscode-extension/) for the local MCP integration shim.

## Configuration

Start from [`config.example.json`](config.example.json). The tracked example is
loopback-only, auth-enabled, rate-limited, and tuned for a local coding-agent
proxy.

| Key | Default | Purpose |
|---|---|---|
| `model_path` | `models/model.onnx` | ONNX/ORT embedding model. |
| `vocab_path` | `models/vocab.txt` | WordPiece vocabulary. |
| `proxy_host` | `127.0.0.1` | Bind address for `--serve`. |
| `proxy_port` | `8088` | Bind port for `--serve`. |
| `repo_root` | empty | Repo to index on startup. |
| `cache_db_path` | `prompt_cache.db` | SQLite prompt-cache path. |
| `retrieval_k` | `6` | Candidate chunks retrieved per request. |
| `embedding_dim` | `384` | Embedding vector dimension. |
| `max_context_chars` | `8000` | Cap for injected context. |
| `upstream_url` | OpenAI chat-completions URL | Upstream OpenAI-compatible endpoint. |
| `upstream_api_key` | empty | Fallback upstream bearer token. |
| `upstream_timeout_seconds` | `60` | Overall upstream timeout. |
| `upstream_connect_timeout_seconds` | `10` | Upstream connect timeout. |
| `upstream_max_response_bytes` | `0` | Max buffered non-streaming response (`0` disables). |
| `stream_idle_timeout_seconds` | `0` | Abort stalled upstream streams (`0` disables). |
| `tokenizer_mode` | `heuristic` | `heuristic` or `model-calibrated`. |
| `model_tiers` / `model_routes` | `[]` | Optional ordered model routing rules. |
| `proxy_auth_bearer_tokens` | `[]` | Local proxy bearer-token allow-list. |
| `proxy_auth_hmac_secret` | empty | HMAC-SHA256 shared secret. |
| `proxy_rate_limit_tokens_per_second` | `0` | Per-caller refill rate (`0` disables). |
| `proxy_rate_limit_burst` | `0` | Per-caller burst (`0` disables). |
| `proxy_max_request_bytes` | `8388608` | Max chat-completions body size. |
| `sync_cache_export_limit` | `1000` | Max cache entries exported by sync. |
| `sync_vector_export_limit` | `1000` | Max vector entries exported by sync. |
| `proxy_forward_client_authorization` | `true` | Forward client `Authorization` upstream; defaults false when local proxy auth is configured unless explicit. |
| `allow_unsafe_remote_proxy` | `false` | Explicit override for unauthenticated non-loopback serving. |
| `prompt_optimizer_enabled` | `false` | Enable project-card and bucket-template optimization. |
| `prompt_templates_path` | empty | JSON template override path. |
| `include_project_card` | `true` | Include repository summary in optimized prompts. |
| `symbol_graph_enabled` | `false` | Build symbol graph during indexing. |
| `graph_expansion_enabled` | `true` | Expand retrieval through graph neighbours. |
| `structural_fast_path_enabled` | `true` | Answer structural queries locally. |
| `prompt_rewriter_enabled` | `false` | Compress assembled context before forwarding. |
| `prompt_rewriter_kind` | `heuristic` | `heuristic` or `llama-cpp`. |
| `prompt_rewriter_max_chars` | `0` | Rewriter char cap (`0` inherits context cap). |
| `llama_model_path` | empty | `.gguf` path for future llama.cpp rewriter wiring. |

Removed command-router keys such as `intents`, `similarity_threshold`,
`history_limit`, and `system_prompt` are rejected by `ConfigLoader`.

Security defaults:

- `/healthz` stays public.
- `/stats` and sync routes are protected when proxy auth is configured.
- Non-loopback serving requires local proxy auth unless
  `allow_unsafe_remote_proxy=true`.
- Placeholder example tokens are rejected on non-loopback hosts.
- Request-size limits are enforced before JSON parsing and forwarding.

## Testing

Primary verification:

```powershell
cmake --build build
.\build\preprocessor_tests.exe --gtest_filter=ConfigLoaderTest.*
ctest --test-dir build --output-on-failure
.\build\smoke_runner.exe
.\build\effectiveness_runner.exe
```

The smoke runner exercises the local middleware pipeline without requiring the
ONNX model. The effectiveness runner emits JSON to stdout and a summary table
to stderr for cache speedup, context packing, compression, retrieval quality,
model routing, auth throughput, and rate limiting.

To visualize those gains over time, run the standalone dashboard:

```powershell
python tools\perf_visualizer\run_dashboard.py `
  --effectiveness-exe build\effectiveness_runner.exe `
  --run-effectiveness-on-start `
  --open
```

See [tools/perf_visualizer/README.md](tools/perf_visualizer/README.md) for live
proxy `/stats` polling and full-program test commands.

See [docs/TESTING.md](docs/TESTING.md) for the full feature usage guide and
[docs/RELEASE.md](docs/RELEASE.md) for the release checklist.

## Install / Package Check

```powershell
cmake --install build --prefix build\install
Get-ChildItem build\install\lib\cmake\LLMPreprocessor
Get-ChildItem build\install\bin\onnxruntime*.dll
```

Consumers can link the installed package with:

```cmake
find_package(LLMPreprocessor CONFIG REQUIRED)
target_link_libraries(my_tool PRIVATE LLMPreprocessor::preprocessor_lib)
```

## License

See [LICENSE](LICENSE).
