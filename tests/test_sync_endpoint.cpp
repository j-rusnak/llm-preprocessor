#include "sync_endpoint.hpp"
#include "prompt_cache.hpp"

#include <gtest/gtest.h>
#include <filesystem>

using preprocessor::SyncBundle;
using preprocessor::SyncCacheEntry;
using preprocessor::SyncEndpoint;
using preprocessor::SyncVectorEntry;
using preprocessor::PromptCache;

namespace fs = std::filesystem;

TEST(SyncEndpoint, RoundtripsJson) {
    SyncEndpoint s;
    SyncBundle b;
    b.cache.push_back({"key1", "body1"});
    b.cache.push_back({"key2", "body2"});
    SyncVectorEntry v;
    v.chunk_id = 42;
    v.vec = {0.1f, 0.2f, 0.3f};
    v.source_path = "src/foo.cpp";
    b.vectors.push_back(v);

    auto j = s.to_json(b);
    auto round = s.from_json(j);
    ASSERT_EQ(round.cache.size(), 2u);
    EXPECT_EQ(round.cache[0].key, "key1");
    ASSERT_EQ(round.vectors.size(), 1u);
    EXPECT_EQ(round.vectors[0].chunk_id, 42u);
    EXPECT_EQ(round.vectors[0].vec.size(), 3u);
}

TEST(SyncEndpoint, AppliesBundleToPromptCache) {
    auto p = fs::temp_directory_path() / "sync_pc.db";
    std::error_code ec; fs::remove(p, ec);
    PromptCache cache(p.string());
    SyncEndpoint s;
    SyncBundle b;
    b.cache.push_back({"k", "v"});
    auto n = s.apply_to_cache(b, &cache);
    EXPECT_EQ(n, 1u);
    EXPECT_EQ(cache.get("k").value_or(""), "v");
    EXPECT_EQ(s.bundles_imported(), 1u);
}

TEST(SyncEndpoint, HandlesEmptyBundle) {
    SyncEndpoint s;
    EXPECT_EQ(s.apply_to_cache(SyncBundle{}, nullptr), 0u);
    EXPECT_EQ(s.bundles_imported(), 1u);
}

TEST(SyncEndpoint, FromJsonRejectsMalformed) {
    SyncEndpoint s;
    auto b = s.from_json("not json");
    EXPECT_TRUE(b.cache.empty());
}
