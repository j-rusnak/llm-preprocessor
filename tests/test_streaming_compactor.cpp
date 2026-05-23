#include "streaming_compactor.hpp"

#include <gtest/gtest.h>

using preprocessor::ChatTurn;
using preprocessor::StreamingCompactor;

TEST(StreamingCompactor, KeepsAllWhenUnderBudget) {
    StreamingCompactor::Config cfg;
    cfg.max_total_chars = 4000;
    cfg.keep_recent = 2;
    StreamingCompactor c(cfg);
    std::vector<ChatTurn> h{
        {"user", "hi"}, {"assistant", "hey"}, {"user", "more"}
    };
    auto r = c.compact(h);
    EXPECT_EQ(r.rolled, 0u);
    EXPECT_TRUE(r.rolled_summary.empty());
    EXPECT_EQ(r.kept.size(), 3u);
}

TEST(StreamingCompactor, RollsOldTurnsIntoSummary) {
    StreamingCompactor::Config cfg;
    cfg.max_total_chars = 100;  // tight
    cfg.keep_recent = 1;
    cfg.summary_chars_per_turn = 24;
    StreamingCompactor c(cfg);
    std::vector<ChatTurn> h;
    for (int i = 0; i < 10; ++i) {
        h.push_back({"user", std::string("message number ") + std::to_string(i)});
    }
    auto r = c.compact(h);
    EXPECT_GT(r.rolled, 0u);
    EXPECT_FALSE(r.rolled_summary.empty());
    EXPECT_GE(r.kept.size(), 1u);
    EXPECT_NE(r.rolled_summary.find("Summary of earlier turns"), std::string::npos);
}

TEST(StreamingCompactor, EmptyHistoryPassesThrough) {
    StreamingCompactor c;
    auto r = c.compact({});
    EXPECT_EQ(r.rolled, 0u);
    EXPECT_TRUE(r.kept.empty());
}

TEST(StreamingCompactor, KeepRecentRespected) {
    StreamingCompactor::Config cfg;
    cfg.max_total_chars = 10;  // would otherwise drop everything
    cfg.keep_recent = 3;
    StreamingCompactor c(cfg);
    std::vector<ChatTurn> h{
        {"a", "x"}, {"b", "y"}, {"c", "z"}, {"d", "w"}
    };
    auto r = c.compact(h);
    EXPECT_GE(r.kept.size(), 3u);
}
