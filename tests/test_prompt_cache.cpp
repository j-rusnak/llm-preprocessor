#include "prompt_cache.hpp"

#include <gtest/gtest.h>

#include <thread>
#include <chrono>

using preprocessor::PromptCache;

TEST(PromptCache, PutGetRoundTrip) {
    PromptCache c(":memory:");
    const auto k = PromptCache::make_key("gpt-4", "hello", {1, 2, 3});
    EXPECT_FALSE(c.get(k).has_value());
    c.put(k, R"({"answer":"hi"})");
    auto got = c.get(k);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(*got, R"({"answer":"hi"})");
}

TEST(PromptCache, MakeKeyIsOrderInvariantForChunkIds) {
    auto a = PromptCache::make_key("m", "p", {1, 2, 3});
    auto b = PromptCache::make_key("m", "p", {3, 1, 2});
    EXPECT_EQ(a, b);
    auto c = PromptCache::make_key("m", "p", {1, 2});
    EXPECT_NE(a, c);
}

TEST(PromptCache, MakeKeySensitiveToModelAndPrompt) {
    auto a = PromptCache::make_key("m1", "p", {});
    auto b = PromptCache::make_key("m2", "p", {});
    auto c = PromptCache::make_key("m1", "q", {});
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
}

TEST(PromptCache, OverwriteReplacesPayload) {
    PromptCache c(":memory:");
    const std::string k = "k1";
    c.put(k, "v1");
    c.put(k, "v2");
    EXPECT_EQ(*c.get(k), "v2");
    EXPECT_EQ(c.size(), 1u);
}

TEST(PromptCache, EraseAndClear) {
    PromptCache c(":memory:");
    c.put("a", "1");
    c.put("b", "2");
    c.erase("a");
    EXPECT_FALSE(c.get("a").has_value());
    EXPECT_EQ(c.size(), 1u);
    c.clear();
    EXPECT_EQ(c.size(), 0u);
}

TEST(PromptCache, TTLExpires) {
    PromptCache c(":memory:", /*ttl_seconds=*/1);
    c.put("k", "v");
    EXPECT_TRUE(c.get("k").has_value());
    std::this_thread::sleep_for(std::chrono::seconds(2));
    EXPECT_FALSE(c.get("k").has_value());
}
