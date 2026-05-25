#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <unordered_map>

namespace preprocessor {

/// Thread-safe counters for the OpenAI-compatible proxy. All counters are
/// monotonic; reset only by re-launching the process. Read via `snapshot()`
/// which returns a JSON object suitable for a `/stats` endpoint.
///
/// Field semantics:
///   - requests_total       : every /v1/chat/completions hit
///   - cache_hits           : served from PromptCache
///   - upstream_calls       : forwarded to upstream LLM
///   - errors_total         : transport / parsing / upstream failures
///   - auth_failures_total  : rejected local proxy auth attempts
///   - rate_limit_denials_total : rejected local proxy rate-limit attempts
///   - request_too_large_denials_total : rejected oversized chat requests
///   - upstream_errors_total : upstream transport or response-limit failures
///   - stream_cancellations_total : downstream client stream disconnects
///   - tokens_in_original   : token count of raw user prompt(s)
///   - tokens_in_compiled   : token count of payload actually sent upstream
///   - tokens_saved         : max(0, original - compiled). Approximates the
///                            cost reduction delivered by sanitiser +
///                            chunk-selection vs naive forwarding.
///   - context_chunks_included_total / omitted_total : retrieved chunks that
///                            did or did not fit into injected context.
///   - context_chars_injected_total : context-system-message chars sent
///                            upstream after optional rewriting.
///   - context_truncations_total : requests where packing omitted chunks.
class ProxyMetrics {
public:
    struct TokenTotals {
        std::uint64_t original = 0;
        std::uint64_t compiled = 0;
        std::uint64_t saved = 0;
    };

    void on_request() { requests_total.fetch_add(1, std::memory_order_relaxed); }
    void on_cache_hit() { cache_hits.fetch_add(1, std::memory_order_relaxed); }
    void on_upstream_call() { upstream_calls.fetch_add(1, std::memory_order_relaxed); }
    void on_error() { errors_total.fetch_add(1, std::memory_order_relaxed); }
    void on_auth_failure() {
        auth_failures_total.fetch_add(1, std::memory_order_relaxed);
    }
    void on_rate_limit_denial() {
        rate_limit_denials_total.fetch_add(1, std::memory_order_relaxed);
    }
    void on_request_too_large_denial() {
        request_too_large_denials_total.fetch_add(1, std::memory_order_relaxed);
    }
    void on_upstream_error() {
        upstream_errors_total.fetch_add(1, std::memory_order_relaxed);
    }
    void on_stream_cancellation() {
        stream_cancellations_total.fetch_add(1, std::memory_order_relaxed);
    }

    void observe_tokens(std::uint64_t original, std::uint64_t compiled) {
        tokens_in_original.fetch_add(original, std::memory_order_relaxed);
        tokens_in_compiled.fetch_add(compiled, std::memory_order_relaxed);
        const std::uint64_t saved = original > compiled ? original - compiled : 0;
        tokens_saved.fetch_add(saved, std::memory_order_relaxed);
    }

    void observe_tokens(const std::string& model_family,
                        std::uint64_t original,
                        std::uint64_t compiled);

    void observe_context_pack(std::uint64_t included_chunks,
                              std::uint64_t omitted_chunks,
                              std::uint64_t injected_chars,
                              bool truncated) {
        context_chunks_included_total.fetch_add(included_chunks,
                                                std::memory_order_relaxed);
        context_chunks_omitted_total.fetch_add(omitted_chunks,
                                               std::memory_order_relaxed);
        context_chars_injected_total.fetch_add(injected_chars,
                                              std::memory_order_relaxed);
        if (truncated) {
            context_truncations_total.fetch_add(1, std::memory_order_relaxed);
        }
    }

    nlohmann::json snapshot() const;

    std::atomic<std::uint64_t> requests_total{0};
    std::atomic<std::uint64_t> cache_hits{0};
    std::atomic<std::uint64_t> upstream_calls{0};
    std::atomic<std::uint64_t> errors_total{0};
    std::atomic<std::uint64_t> auth_failures_total{0};
    std::atomic<std::uint64_t> rate_limit_denials_total{0};
    std::atomic<std::uint64_t> request_too_large_denials_total{0};
    std::atomic<std::uint64_t> upstream_errors_total{0};
    std::atomic<std::uint64_t> stream_cancellations_total{0};
    std::atomic<std::uint64_t> tokens_in_original{0};
    std::atomic<std::uint64_t> tokens_in_compiled{0};
    std::atomic<std::uint64_t> tokens_saved{0};
    std::atomic<std::uint64_t> context_chunks_included_total{0};
    std::atomic<std::uint64_t> context_chunks_omitted_total{0};
    std::atomic<std::uint64_t> context_chars_injected_total{0};
    std::atomic<std::uint64_t> context_truncations_total{0};

private:
    mutable std::mutex token_family_mutex_;
    std::unordered_map<std::string, TokenTotals> tokens_by_model_family_;
};

} // namespace preprocessor
