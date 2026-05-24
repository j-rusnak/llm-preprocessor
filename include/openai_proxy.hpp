#pragma once

#include "auth_middleware.hpp"
#include "rate_limiter.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace httplib {
class Server;
} // namespace httplib

namespace preprocessor {

class RepoIndex;
class PromptCache;
class ProxyMetrics;
class ILLMTokenizer;
class PromptOptimizer;
class IPromptRewriter;
class SymbolGraph;
class StructuralQueryEngine;

/// Configuration for `OpenAIProxy`.
struct OpenAIProxyConfig {
    /// Fully-qualified upstream chat-completions endpoint, e.g.
    /// "https://api.openai.com/v1/chat/completions".
    std::string upstream_url = "https://api.openai.com/v1/chat/completions";

    /// Optional bearer token forwarded as `Authorization: Bearer ...` when the
    /// incoming request does not carry its own Authorization header.
    std::string upstream_api_key;

    /// Number of retrieved chunks to inject as additional context per request.
    std::size_t retrieval_k = 6;

    /// Cap on the combined character length of injected chunks. Prevents one
    /// huge file from blowing the prompt budget. Set to 0 to disable.
    std::size_t max_context_chars = 8000;

    /// libcurl upstream timeout (seconds).
    long upstream_timeout_seconds = 60;

    /// Local proxy authentication. Disabled when no bearer tokens or HMAC
    /// secret are configured.
    AuthMiddleware::Config auth;

    /// Local per-caller token bucket. Disabled when both values are zero.
    RateLimiter::Config rate_limit;

    /// Maximum accepted HTTP request body size for chat completions. Set to 0
    /// to disable the guard. The default leaves room for large coding-agent
    /// prompts without accepting unbounded payloads.
    std::size_t max_request_bytes = 8 * 1024 * 1024;

    /// Forward the client's Authorization header to the upstream provider.
    /// Disable this when Authorization is used as a local proxy bearer token.
    bool forward_client_authorization = true;
};

/// OpenAI-compatible HTTP proxy.
///
/// Listens on a loopback port and exposes:
///   POST /v1/chat/completions  - intercepts the request, augments the
///                                latest user message with retrieved chunks
///                                from `RepoIndex`, consults `PromptCache`
///                                (keyed by prompt + model + chunk ids),
///                                forwards to the upstream LLM if needed,
///                                and returns the upstream JSON verbatim.
///   GET  /stats                - JSON snapshot of `ProxyMetrics`.
///   GET  /healthz              - liveness probe.
///
/// The proxy is intended to be a drop-in middleware for tools like Cursor or
/// Continue: point their OpenAI base URL at `http://127.0.0.1:<port>` and the
/// preprocessor handles retrieval + caching + telemetry transparently.
///
/// Thread-safety: `listen()` blocks until `stop()` is called from another
/// thread. The internal handler is reentrant; it relies on RepoIndex and
/// PromptCache providing their own thread-safety.
class OpenAIProxy {
public:
    OpenAIProxy(RepoIndex& index,
                PromptCache& cache,
                ProxyMetrics& metrics,
                ILLMTokenizer& tokenizer,
                OpenAIProxyConfig config = {});
    ~OpenAIProxy();

    OpenAIProxy(const OpenAIProxy&) = delete;
    OpenAIProxy& operator=(const OpenAIProxy&) = delete;

    /// Install a Phase 2 prompt optimiser. Ownership stays with the caller;
    /// the pointer must outlive the proxy. Pass `nullptr` to revert to the
    /// Phase 1 plain-context behaviour.
    void set_prompt_optimizer(PromptOptimizer* optimiser) noexcept;

    /// Install a Phase 3 symbol graph. When attached, retrieval results are
    /// expanded with one hop of graph-aware neighbours before being injected
    /// as context. Caller owns; pass `nullptr` to detach.
    void set_symbol_graph(SymbolGraph* graph) noexcept;

    /// Install a Phase 3 structural query engine. When attached, the proxy
    /// consults it before doing retrieval; on a successful answer the request
    /// is served as a synthetic completion without forwarding upstream
    /// (zero-LLM fast path). Caller owns; pass `nullptr` to detach.
    void set_structural_query_engine(StructuralQueryEngine* engine) noexcept;

    /// Install a Phase 5 prompt rewriter. Applied to the assembled system
    /// context block immediately before it is attached to the upstream
    /// payload. Caller owns; pass `nullptr` to disable.
    void set_prompt_rewriter(IPromptRewriter* rewriter) noexcept;

    /// Bind to `host:port` without blocking. Returns the actually-bound port
    /// (useful when `port == 0` to let the OS pick one). Throws if bind fails.
    int bind_to_port(const std::string& host, int port);

    /// Begin serving on the bound port. Blocks until `stop()` is called.
    /// Must be preceded by `bind_to_port`.
    void listen_after_bind();

    /// Convenience: bind + listen. Blocks. Returns when `stop()` is called.
    void listen(const std::string& host, int port);

    /// Signal the running server to stop. Safe to call from any thread.
    void stop();

private:
    void install_routes();

    RepoIndex& index_;
    PromptCache& cache_;
    ProxyMetrics& metrics_;
    ILLMTokenizer& tokenizer_;
    OpenAIProxyConfig config_;
    PromptOptimizer* optimiser_ = nullptr;
    SymbolGraph* symbol_graph_ = nullptr;
    StructuralQueryEngine* structural_engine_ = nullptr;
    IPromptRewriter* rewriter_ = nullptr;
    AuthMiddleware auth_;
    RateLimiter rate_limiter_;
    std::unique_ptr<httplib::Server> server_;
};

} // namespace preprocessor
