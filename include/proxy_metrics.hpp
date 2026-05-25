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
///   - tokens_in_original   : token count of raw user prompt(s)
///   - tokens_in_compiled   : token count of payload actually sent upstream
///   - tokens_saved         : max(0, original - compiled). Approximates the
///                            cost reduction delivered by sanitiser +
///                            chunk-selection vs naive forwarding.
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

    void observe_tokens(std::uint64_t original, std::uint64_t compiled) {
        tokens_in_original.fetch_add(original, std::memory_order_relaxed);
        tokens_in_compiled.fetch_add(compiled, std::memory_order_relaxed);
        const std::uint64_t saved = original > compiled ? original - compiled : 0;
        tokens_saved.fetch_add(saved, std::memory_order_relaxed);
    }

    void observe_tokens(const std::string& model_family,
                        std::uint64_t original,
                        std::uint64_t compiled);

    nlohmann::json snapshot() const;

    std::atomic<std::uint64_t> requests_total{0};
    std::atomic<std::uint64_t> cache_hits{0};
    std::atomic<std::uint64_t> upstream_calls{0};
    std::atomic<std::uint64_t> errors_total{0};
    std::atomic<std::uint64_t> tokens_in_original{0};
    std::atomic<std::uint64_t> tokens_in_compiled{0};
    std::atomic<std::uint64_t> tokens_saved{0};

private:
    mutable std::mutex token_family_mutex_;
    std::unordered_map<std::string, TokenTotals> tokens_by_model_family_;
};

} // namespace preprocessor
