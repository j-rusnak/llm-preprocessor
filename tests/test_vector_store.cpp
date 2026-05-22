#include <gtest/gtest.h>
#include "vector_store.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace {

std::vector<float> normalize(std::vector<float> v) {
    float s = 0.0f;
    for (float x : v) s += x * x;
    s = std::sqrt(s);
    if (s > 0.0f) {
        for (auto& x : v) x /= s;
    }
    return v;
}

std::vector<float> random_unit(std::size_t dim, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = d(rng);
    return normalize(std::move(v));
}

} // namespace

TEST(VectorStoreTest, RoundTripsNearestNeighbour) {
    preprocessor::VectorStore store(8, /*max*/ 100);
    std::mt19937 rng(42);

    const auto a = normalize({1, 0, 0, 0, 0, 0, 0, 0});
    const auto b = normalize({0, 1, 0, 0, 0, 0, 0, 0});
    const auto c = normalize({0, 0, 1, 0, 0, 0, 0, 0});

    store.add(1, a);
    store.add(2, b);
    store.add(3, c);

    auto hits = store.search(a, 1);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].id, 1u);
}

TEST(VectorStoreTest, RejectsDimMismatch) {
    preprocessor::VectorStore store(4, 10);
    EXPECT_THROW(store.add(1, std::vector<float>{1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(store.search(std::vector<float>{1, 2, 3}, 1), std::invalid_argument);
}

TEST(VectorStoreTest, RemoveExcludesFromResults) {
    preprocessor::VectorStore store(4, 10);
    store.add(1, normalize({1, 0, 0, 0}));
    store.add(2, normalize({0.99f, 0.01f, 0, 0}));
    store.remove(1);

    auto hits = store.search(normalize({1, 0, 0, 0}), 5);
    for (const auto& h : hits) {
        EXPECT_NE(h.id, 1u);
    }
}

TEST(VectorStoreTest, EmptySearchReturnsEmpty) {
    preprocessor::VectorStore store(4, 10);
    EXPECT_TRUE(store.search(std::vector<float>(4, 0.0f), 5).empty());
}

TEST(VectorStoreTest, ConstructionValidates) {
    EXPECT_THROW(preprocessor::VectorStore(0, 10), std::invalid_argument);
    EXPECT_THROW(preprocessor::VectorStore(4, 0), std::invalid_argument);
    EXPECT_THROW(preprocessor::VectorStore(4, 10, "", "bogus-metric"), std::invalid_argument);
}

TEST(VectorStoreTest, OrdersResultsByDistanceAscending) {
    preprocessor::VectorStore store(4, 10);
    std::mt19937 rng(7);
    for (std::uint64_t i = 0; i < 10; ++i) {
        store.add(i, random_unit(4, rng));
    }
    auto hits = store.search(random_unit(4, rng), 5);
    for (std::size_t i = 1; i < hits.size(); ++i) {
        EXPECT_LE(hits[i - 1].distance, hits[i].distance);
    }
}
