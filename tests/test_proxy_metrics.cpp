#include "proxy_metrics.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using preprocessor::ProxyMetrics;

TEST(ProxyMetrics, CountersStartAtZero) {
    ProxyMetrics m;
    auto j = m.snapshot();
    EXPECT_EQ(j["requests_total"], 0u);
    EXPECT_EQ(j["tokens_saved"], 0u);
}

TEST(ProxyMetrics, IncrementsAccumulate) {
    ProxyMetrics m;
    m.on_request();
    m.on_request();
    m.on_cache_hit();
    m.on_upstream_call();
    m.on_error();
    auto j = m.snapshot();
    EXPECT_EQ(j["requests_total"], 2u);
    EXPECT_EQ(j["cache_hits"], 1u);
    EXPECT_EQ(j["upstream_calls"], 1u);
    EXPECT_EQ(j["errors_total"], 1u);
}

TEST(ProxyMetrics, TokensSavedClampedToZero) {
    ProxyMetrics m;
    m.observe_tokens(100, 60);  // saved 40
    m.observe_tokens(50, 80);   // saved 0 (compiled exceeded original)
    auto j = m.snapshot();
    EXPECT_EQ(j["tokens_in_original"], 150u);
    EXPECT_EQ(j["tokens_in_compiled"], 140u);
    EXPECT_EQ(j["tokens_saved"], 40u);
}
