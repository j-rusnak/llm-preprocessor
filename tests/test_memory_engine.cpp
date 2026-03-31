#include <gtest/gtest.h>
#include "memory_engine.hpp"

#include <cstdio>
#include <string>

// Helper to create a unique temp DB path per test.
class MemoryEngineTest : public ::testing::Test {
protected:
    std::string db_path_;

    void SetUp() override {
        db_path_ = ":memory:";
    }
};

TEST_F(MemoryEngineTest, AddAndRetrieveMessages) {
    preprocessor::MemoryEngine engine(db_path_);
    engine.add_message("user", "hello");
    engine.add_message("assistant", "hi there");

    auto history = engine.get_recent_history(10);
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[0].first, "user");
    EXPECT_EQ(history[0].second, "hello");
    EXPECT_EQ(history[1].first, "assistant");
    EXPECT_EQ(history[1].second, "hi there");
}

TEST_F(MemoryEngineTest, RespectsLimit) {
    preprocessor::MemoryEngine engine(db_path_);
    engine.add_message("user", "msg1");
    engine.add_message("assistant", "msg2");
    engine.add_message("user", "msg3");
    engine.add_message("assistant", "msg4");

    auto history = engine.get_recent_history(2);
    ASSERT_EQ(history.size(), 2u);
    // Should return the two most recent in chronological order.
    EXPECT_EQ(history[0].second, "msg3");
    EXPECT_EQ(history[1].second, "msg4");
}

TEST_F(MemoryEngineTest, EmptyHistoryReturnsEmpty) {
    preprocessor::MemoryEngine engine(db_path_);
    auto history = engine.get_recent_history(5);
    EXPECT_TRUE(history.empty());
}

TEST_F(MemoryEngineTest, ChronologicalOrder) {
    preprocessor::MemoryEngine engine(db_path_);
    engine.add_message("user", "first");
    engine.add_message("assistant", "second");
    engine.add_message("user", "third");

    auto history = engine.get_recent_history(10);
    ASSERT_EQ(history.size(), 3u);
    EXPECT_EQ(history[0].second, "first");
    EXPECT_EQ(history[1].second, "second");
    EXPECT_EQ(history[2].second, "third");
}

TEST_F(MemoryEngineTest, HandlesSpecialCharacters) {
    preprocessor::MemoryEngine engine(db_path_);
    engine.add_message("user", "Hello \"world\" it's <html> & stuff; DROP TABLE messages;");

    auto history = engine.get_recent_history(1);
    ASSERT_EQ(history.size(), 1u);
    EXPECT_EQ(history[0].second, "Hello \"world\" it's <html> & stuff; DROP TABLE messages;");
}

TEST_F(MemoryEngineTest, MoveConstructorTransfersOwnership) {
    preprocessor::MemoryEngine engine1(db_path_);
    engine1.add_message("user", "test message");

    preprocessor::MemoryEngine engine2(std::move(engine1));

    auto history = engine2.get_recent_history(10);
    ASSERT_EQ(history.size(), 1u);
    EXPECT_EQ(history[0].second, "test message");
}

TEST_F(MemoryEngineTest, MoveAssignmentTransfersOwnership) {
    preprocessor::MemoryEngine engine1(db_path_);
    engine1.add_message("user", "original");

    preprocessor::MemoryEngine engine2(":memory:");
    engine2 = std::move(engine1);

    auto history = engine2.get_recent_history(10);
    ASSERT_EQ(history.size(), 1u);
    EXPECT_EQ(history[0].second, "original");
}

TEST_F(MemoryEngineTest, UpdateLastMessage) {
    preprocessor::MemoryEngine engine(db_path_);
    engine.add_message("user", "hello");
    engine.add_message("assistant", "(placeholder)");

    engine.update_last_message("real response");

    auto history = engine.get_recent_history(10);
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[1].second, "real response");
}

TEST_F(MemoryEngineTest, ClearHistoryRemovesAll) {
    preprocessor::MemoryEngine engine(db_path_);
    engine.add_message("user", "msg1");
    engine.add_message("assistant", "msg2");

    engine.clear_history();
    auto history = engine.get_recent_history(10);
    EXPECT_TRUE(history.empty());
}

TEST_F(MemoryEngineTest, PruneKeepsRecentMessages) {
    preprocessor::MemoryEngine engine(db_path_);
    engine.add_message("user", "msg1");
    engine.add_message("assistant", "msg2");
    engine.add_message("user", "msg3");
    engine.add_message("assistant", "msg4");
    engine.add_message("user", "msg5");

    engine.prune(3);
    auto history = engine.get_recent_history(10);
    ASSERT_EQ(history.size(), 3u);
    EXPECT_EQ(history[0].second, "msg3");
    EXPECT_EQ(history[1].second, "msg4");
    EXPECT_EQ(history[2].second, "msg5");
}

TEST_F(MemoryEngineTest, UpdateLastMessageOnEmptyTableThrows) {
    preprocessor::MemoryEngine engine(db_path_);
    EXPECT_THROW(engine.update_last_message("content"), std::runtime_error);
}

TEST_F(MemoryEngineTest, PruneZeroRowsThrows) {
    preprocessor::MemoryEngine engine(db_path_);
    EXPECT_THROW(engine.prune(0), std::invalid_argument);
}

TEST_F(MemoryEngineTest, PruneNegativeRowsThrows) {
    preprocessor::MemoryEngine engine(db_path_);
    EXPECT_THROW(engine.prune(-5), std::invalid_argument);
}
