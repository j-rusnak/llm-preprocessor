#include "embedding_cache.hpp"

#include <gtest/gtest.h>
#include <filesystem>

using preprocessor::EmbeddingCache;

namespace fs = std::filesystem;

namespace {
fs::path tmp_db(const char* name) {
    auto p = fs::temp_directory_path() / (std::string("emb_cache_") + name + ".db");
    std::error_code ec; fs::remove(p, ec);
    return p;
}
}

TEST(EmbeddingCache, PutGetRoundtrip) {
    auto p = tmp_db("rt");
    EmbeddingCache c(p.string(), "model-A");
    std::vector<float> v{1.0f, 2.0f, 3.5f, -7.25f};
    c.put("hello", v);
    auto got = c.get("hello");
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, v);
}

TEST(EmbeddingCache, MissReturnsNullopt) {
    auto p = tmp_db("miss");
    EmbeddingCache c(p.string(), "model-A");
    EXPECT_FALSE(c.get("unknown").has_value());
}

TEST(EmbeddingCache, KeyChangesWithModelId) {
    auto p = tmp_db("kbd");
    EmbeddingCache a(p.string(), "model-A");
    EmbeddingCache b(p.string(), "model-B");
    EXPECT_NE(a.key_for("same"), b.key_for("same"));
}

TEST(EmbeddingCache, SurvivesReopen) {
    auto p = tmp_db("reopen");
    {
        EmbeddingCache c(p.string(), "m");
        c.put("k", std::vector<float>{0.1f, 0.2f});
    }
    EmbeddingCache c(p.string(), "m");
    auto got = c.get("k");
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->size(), 2u);
}

TEST(EmbeddingCache, ClearEmpties) {
    auto p = tmp_db("clr");
    EmbeddingCache c(p.string(), "m");
    c.put("a", {1.f});
    c.put("b", {2.f});
    EXPECT_EQ(c.size(), 2u);
    c.clear();
    EXPECT_EQ(c.size(), 0u);
}
