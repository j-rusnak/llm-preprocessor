#include "model_router.hpp"

#include <gtest/gtest.h>

using preprocessor::ModelRouter;
using preprocessor::ModelTier;
using preprocessor::ModelRoute;
using preprocessor::PromptBucket;

TEST(ModelRouter, RoutesByBucket) {
    ModelRouter r;
    r.add_tier({"cheap", "http://x", "gpt-cheap", "", 0});
    r.add_tier({"frontier", "http://y", "gpt-pro", "", 0});
    r.add_route({PromptBucket::CodeExplain, 0, 0, "cheap"});
    r.add_route({PromptBucket::CodeEdit,    0, 0, "frontier"});

    auto* a = r.route(PromptBucket::CodeExplain, 100);
    auto* b = r.route(PromptBucket::CodeEdit, 5000);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(a->name, "cheap");
    EXPECT_EQ(b->name, "frontier");
}

TEST(ModelRouter, RoutesByRequestSize) {
    ModelRouter r;
    r.add_tier({"small", "u1", "m1", "", 0});
    r.add_tier({"big",   "u2", "m2", "", 0});
    r.add_route({PromptBucket::CodeEdit, 0,    1000, "small"});
    r.add_route({PromptBucket::CodeEdit, 1001, 0,    "big"});

    EXPECT_EQ(r.route(PromptBucket::CodeEdit, 500)->name, "small");
    EXPECT_EQ(r.route(PromptBucket::CodeEdit, 50000)->name, "big");
}

TEST(ModelRouter, NoMatchReturnsNull) {
    ModelRouter r;
    EXPECT_EQ(r.route(PromptBucket::CodeExplain, 0), nullptr);
}

TEST(ModelRouter, UnknownTierIsSkipped) {
    ModelRouter r;
    r.add_route({PromptBucket::CodeExplain, 0, 0, "ghost"});
    EXPECT_EQ(r.route(PromptBucket::CodeExplain, 0), nullptr);
}

TEST(ModelRouter, CountsTrackInsertion) {
    ModelRouter r;
    r.add_tier({"a", "", "", "", 0});
    r.add_tier({"b", "", "", "", 0});
    r.add_route({PromptBucket::Freeform, 0, 0, "a"});
    EXPECT_EQ(r.tier_count(), 2u);
    EXPECT_EQ(r.route_count(), 1u);
}
