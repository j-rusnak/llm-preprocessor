#include "bm25_index.hpp"
#include "hybrid_retriever.hpp"
#include "vector_store.hpp"

#include <gtest/gtest.h>

#include <vector>

using preprocessor::BM25Index;
using preprocessor::HybridRetriever;
using preprocessor::VectorStore;

namespace {

// Build a 4-dim L2-normalized "embedding" by placing weight on one axis.
std::vector<float> axis_vec(int axis, std::size_t dim = 4) {
    std::vector<float> v(dim, 0.0f);
    v[axis % dim] = 1.0f;
    return v;
}

} // namespace

TEST(HybridRetriever, FusesVectorAndKeywordResults) {
    VectorStore vec(4, 64, "", "cosine");
    BM25Index bm;

    vec.add(1, axis_vec(0));
    bm.add(1, "alpha gamma");
    vec.add(2, axis_vec(1));
    bm.add(2, "beta gamma");
    vec.add(3, axis_vec(2));
    bm.add(3, "delta");

    HybridRetriever h(vec, bm, 60);

    // Query close to axis-1 (id=2) but containing the word "alpha" (id=1).
    // Both rankers contribute; ids 1 and 2 should out-rank id 3.
    auto hits = h.search("alpha", axis_vec(1), 3);
    ASSERT_FALSE(hits.empty());

    bool saw1 = false, saw2 = false;
    for (const auto& h : hits) {
        if (h.id == 1) saw1 = true;
        if (h.id == 2) saw2 = true;
    }
    EXPECT_TRUE(saw1);
    EXPECT_TRUE(saw2);
}

TEST(HybridRetriever, EmptyKResultEmpty) {
    VectorStore vec(4, 16);
    BM25Index bm;
    HybridRetriever h(vec, bm);
    EXPECT_TRUE(h.search("foo", axis_vec(0), 0).empty());
}

TEST(HybridRetriever, KeywordOnlyStillReturnsResults) {
    VectorStore vec(4, 16);
    BM25Index bm;
    bm.add(42, "lonely keyword document");
    HybridRetriever h(vec, bm);
    auto hits = h.search("keyword", axis_vec(0), 3);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].id, 42u);
}

TEST(HybridRetriever, KeywordSearchUsesNormalizedIdentifierTerms) {
    VectorStore vec(4, 16);
    BM25Index bm;
    bm.add(42, "handles http status responses");

    HybridRetriever h(vec, bm);
    auto hits = h.search("getHTTPStatus", axis_vec(0), 3);

    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].id, 42u);
    EXPECT_GT(hits[0].keyword_score, 0.0f);
}

TEST(HybridRetriever, RejectsZeroK) {
    VectorStore vec(4, 16);
    BM25Index bm;
    EXPECT_THROW(HybridRetriever(vec, bm, 0), std::invalid_argument);
}
