#pragma once

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
    std::unique_ptr<httplib::Server> server_;
};

} // namespace preprocessor
