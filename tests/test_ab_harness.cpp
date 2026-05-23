#include "ab_harness.hpp"

#include <gtest/gtest.h>
#include <set>

using preprocessor::AbExperiment;
using preprocessor::AbHarness;
using preprocessor::AbVariant;

TEST(AbHarness, AssignmentIsDeterministic) {
    AbHarness h;
    h.define({"tpl", {{"A", 1.0}, {"B", 1.0}}});
    auto a1 = h.assign("tpl", "client-123");
    auto a2 = h.assign("tpl", "client-123");
    EXPECT_EQ(a1, a2);
}

TEST(AbHarness, DistributesAcrossKeys) {
    AbHarness h;
    h.define({"tpl", {{"A", 1.0}, {"B", 1.0}}});
    std::set<std::string> seen;
    for (int i = 0; i < 200; ++i) {
        seen.insert(h.assign("tpl", "k" + std::to_string(i)));
    }
    EXPECT_EQ(seen.size(), 2u);
}

TEST(AbHarness, RespectsWeights) {
    AbHarness h;
    h.define({"tpl", {{"A", 9.0}, {"B", 1.0}}});
    int a = 0, b = 0;
    for (int i = 0; i < 1000; ++i) {
        auto v = h.assign("tpl", std::to_string(i));
        if (v == "A") ++a; else ++b;
    }
    // 9:1 weighting; allow generous slack.
    EXPECT_GT(a, 4 * b);
}

TEST(AbHarness, UnknownExperimentReturnsEmpty) {
    AbHarness h;
    EXPECT_EQ(h.assign("nope", "k"), "");
}

TEST(AbHarness, DefineRejectsEmpty) {
    AbHarness h;
    EXPECT_THROW(h.define({"e", {}}), std::invalid_argument);
}

TEST(AbHarness, HitCounters) {
    AbHarness h;
    h.define({"tpl", {{"A", 1.0}}});
    h.record_hit("tpl", "A");
    h.record_hit("tpl", "A");
    auto c = h.hit_counts();
    EXPECT_EQ(c["tpl::A"], 2u);
}
