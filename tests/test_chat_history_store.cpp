#include <gtest/gtest.h>
#include "chat_history_store.hpp"

#include <string>

class ChatHistoryStoreTest : public ::testing::Test {
protected:
    std::string db_path_ = ":memory:";
};

TEST_F(ChatHistoryStoreTest, AddAndRetrieveMessages) {
    preprocessor::ChatHistoryStore store(db_path_);
    store.add_message("user", "hello");
    store.add_message("assistant", "hi there");

    auto history = store.get_recent_history(10);
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[0].first, "user");
    EXPECT_EQ(history[0].second, "hello");
    EXPECT_EQ(history[1].first, "assistant");
    EXPECT_EQ(history[1].second, "hi there");
}

TEST_F(ChatHistoryStoreTest, RespectsLimit) {
    preprocessor::ChatHistoryStore store(db_path_);
    store.add_message("user", "msg1");
    store.add_message("assistant", "msg2");
    store.add_message("user", "msg3");
    store.add_message("assistant", "msg4");

    auto history = store.get_recent_history(2);
    ASSERT_EQ(history.size(), 2u);
    EXPECT_EQ(history[0].second, "msg3");
    EXPECT_EQ(history[1].second, "msg4");
}

TEST_F(ChatHistoryStoreTest, EmptyHistoryReturnsEmpty) {
    preprocessor::ChatHistoryStore store(db_path_);
    EXPECT_TRUE(store.get_recent_history(5).empty());
}

TEST_F(ChatHistoryStoreTest, HandlesSpecialCharacters) {
    preprocessor::ChatHistoryStore store(db_path_);
    store.add_message("user", "Hello \"world\" it's <html> & stuff; DROP TABLE messages;");
    auto h = store.get_recent_history(1);
    ASSERT_EQ(h.size(), 1u);
    EXPECT_EQ(h[0].second, "Hello \"world\" it's <html> & stuff; DROP TABLE messages;");
}

TEST_F(ChatHistoryStoreTest, MoveTransfersOwnership) {
    preprocessor::ChatHistoryStore a(db_path_);
    a.add_message("user", "ping");
    preprocessor::ChatHistoryStore b(std::move(a));
    auto h = b.get_recent_history(10);
    ASSERT_EQ(h.size(), 1u);
    EXPECT_EQ(h[0].second, "ping");
}

TEST_F(ChatHistoryStoreTest, UpdateLastMessage) {
    preprocessor::ChatHistoryStore store(db_path_);
    store.add_message("user", "hello");
    store.add_message("assistant", "(placeholder)");
    store.update_last_message("real response");

    auto h = store.get_recent_history(10);
    EXPECT_EQ(h[1].second, "real response");
}

TEST_F(ChatHistoryStoreTest, ClearAndPrune) {
    preprocessor::ChatHistoryStore store(db_path_);
    for (int i = 0; i < 5; ++i) {
        store.add_message("user", "m" + std::to_string(i));
    }
    store.prune(3);
    auto h = store.get_recent_history(10);
    EXPECT_EQ(h.size(), 3u);
    EXPECT_EQ(h.front().second, "m2");

    store.clear_history();
    EXPECT_TRUE(store.get_recent_history(10).empty());
}

TEST_F(ChatHistoryStoreTest, UpdateLastOnEmptyThrows) {
    preprocessor::ChatHistoryStore store(db_path_);
    EXPECT_THROW(store.update_last_message("x"), std::runtime_error);
}

TEST_F(ChatHistoryStoreTest, PruneInvalidArg) {
    preprocessor::ChatHistoryStore store(db_path_);
    EXPECT_THROW(store.prune(0), std::invalid_argument);
    EXPECT_THROW(store.prune(-1), std::invalid_argument);
}
