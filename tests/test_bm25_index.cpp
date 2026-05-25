#include "bm25_index.hpp"

#include <gtest/gtest.h>

using preprocessor::BM25Index;

TEST(BM25Index, EmptyIndexReturnsNothing) {
    BM25Index idx;
    EXPECT_TRUE(idx.search("anything", 5).empty());
}

TEST(BM25Index, SingleTermMatch) {
    BM25Index idx;
    idx.add(1, "hello world");
    idx.add(2, "goodbye sky");
    auto hits = idx.search("hello", 5);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].id, 1u);
    EXPECT_GT(hits[0].score, 0.0f);
}

TEST(BM25Index, RanksMoreRelevantHigher) {
    BM25Index idx;
    idx.add(1, "buffer overflow buffer overflow buffer");
    idx.add(2, "buffer something else entirely");
    idx.add(3, "completely unrelated text");
    auto hits = idx.search("buffer overflow", 3);
    ASSERT_GE(hits.size(), 2u);
    EXPECT_EQ(hits[0].id, 1u);
    EXPECT_GE(hits[0].score, hits[1].score);
}

TEST(BM25Index, IdentifierSubtokenSplit) {
    BM25Index idx;
    idx.add(1, "void getUserName() { return name_; }");
    idx.add(2, "int main() { return 0; }");
    // "user" should match via camelCase split of getUserName.
    auto hits = idx.search("user", 5);
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits[0].id, 1u);
}

TEST(BM25Index, SnakeCaseSplit) {
    BM25Index idx;
    idx.add(1, "compute_hash_value");
    auto hits = idx.search("hash", 5);
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits[0].id, 1u);
}

TEST(BM25Index, RemoveDropsDoc) {
    BM25Index idx;
    idx.add(1, "alpha beta gamma");
    idx.add(2, "alpha delta");
    idx.remove(1);
    auto hits = idx.search("alpha", 5);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].id, 2u);
}

TEST(BM25Index, AddReplacesPreviousDoc) {
    BM25Index idx;
    idx.add(1, "original content");
    idx.add(1, "replacement content");
    auto h_orig = idx.search("original", 5);
    EXPECT_TRUE(h_orig.empty());
    auto h_rep = idx.search("replacement", 5);
    ASSERT_FALSE(h_rep.empty());
    EXPECT_EQ(h_rep[0].id, 1u);
}

TEST(BM25Index, ConstructorValidation) {
    EXPECT_THROW(BM25Index(-0.1f, 0.5f), std::invalid_argument);
    EXPECT_THROW(BM25Index(1.0f, 1.5f), std::invalid_argument);
    EXPECT_NO_THROW(BM25Index(1.0f, 0.0f));
}
