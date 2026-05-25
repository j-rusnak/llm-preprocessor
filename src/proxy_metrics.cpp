#include "proxy_metrics.hpp"

#include <nlohmann/json.hpp>

namespace preprocessor {

void ProxyMetrics::observe_tokens(const std::string& model_family,
                                  std::uint64_t original,
                                  std::uint64_t compiled) {
    observe_tokens(original, compiled);

    const std::uint64_t saved = original > compiled ? original - compiled : 0;
    const std::string family = model_family.empty() ? "default" : model_family;
    std::lock_guard<std::mutex> lock(token_family_mutex_);
    auto& totals = tokens_by_model_family_[family];
    totals.original += original;
    totals.compiled += compiled;
    totals.saved += saved;
}

nlohmann::json ProxyMetrics::snapshot() const {
    nlohmann::json by_family = nlohmann::json::object();
    {
        std::lock_guard<std::mutex> lock(token_family_mutex_);
        for (const auto& [family, totals] : tokens_by_model_family_) {
            by_family[family] = {
                {"original", totals.original},
                {"compiled", totals.compiled},
                {"saved", totals.saved},
            };
        }
    }

    return {
        {"requests_total",     requests_total.load(std::memory_order_relaxed)},
        {"cache_hits",         cache_hits.load(std::memory_order_relaxed)},
        {"upstream_calls",     upstream_calls.load(std::memory_order_relaxed)},
        {"errors_total",       errors_total.load(std::memory_order_relaxed)},
        {"auth_failures_total",
            auth_failures_total.load(std::memory_order_relaxed)},
        {"rate_limit_denials_total",
            rate_limit_denials_total.load(std::memory_order_relaxed)},
        {"request_too_large_denials_total",
            request_too_large_denials_total.load(std::memory_order_relaxed)},
        {"upstream_errors_total",
            upstream_errors_total.load(std::memory_order_relaxed)},
        {"stream_cancellations_total",
            stream_cancellations_total.load(std::memory_order_relaxed)},
        {"tokens_in_original", tokens_in_original.load(std::memory_order_relaxed)},
        {"tokens_in_compiled", tokens_in_compiled.load(std::memory_order_relaxed)},
        {"tokens_saved",       tokens_saved.load(std::memory_order_relaxed)},
        {"context_chunks_included_total",
            context_chunks_included_total.load(std::memory_order_relaxed)},
        {"context_chunks_omitted_total",
            context_chunks_omitted_total.load(std::memory_order_relaxed)},
        {"context_chars_injected_total",
            context_chars_injected_total.load(std::memory_order_relaxed)},
        {"context_truncations_total",
            context_truncations_total.load(std::memory_order_relaxed)},
        {"tokens_by_model_family", by_family},
    };
}

} // namespace preprocessor
