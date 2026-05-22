#include "proxy_metrics.hpp"

#include <nlohmann/json.hpp>

namespace preprocessor {

nlohmann::json ProxyMetrics::snapshot() const {
    return {
        {"requests_total",     requests_total.load(std::memory_order_relaxed)},
        {"cache_hits",         cache_hits.load(std::memory_order_relaxed)},
        {"upstream_calls",     upstream_calls.load(std::memory_order_relaxed)},
        {"errors_total",       errors_total.load(std::memory_order_relaxed)},
        {"tokens_in_original", tokens_in_original.load(std::memory_order_relaxed)},
        {"tokens_in_compiled", tokens_in_compiled.load(std::memory_order_relaxed)},
        {"tokens_saved",       tokens_saved.load(std::memory_order_relaxed)},
    };
}

} // namespace preprocessor
