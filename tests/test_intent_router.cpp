#include <gtest/gtest.h>
#include "intent_router.hpp"

#include <cmath>
#include <vector>

TEST(IntentRouterTest, IdenticalVectorsReturnOne) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    float score = preprocessor::IntentRouter::cosine_similarity(a, b);
    EXPECT_NEAR(score, 1.0f, 1e-5f);
}

TEST(IntentRouterTest, OrthogonalVectorsReturnZero) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    float score = preprocessor::IntentRouter::cosine_similarity(a, b);
    EXPECT_NEAR(score, 0.0f, 1e-5f);
}

TEST(IntentRouterTest, OppositeVectorsReturnNegativeOne) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {-1.0f, 0.0f};
    float score = preprocessor::IntentRouter::cosine_similarity(a, b);
    EXPECT_NEAR(score, -1.0f, 1e-5f);
}

TEST(IntentRouterTest, EmptyVectorsReturnZero) {
    std::vector<float> a;
    std::vector<float> b;
    float score = preprocessor::IntentRouter::cosine_similarity(a, b);
    EXPECT_FLOAT_EQ(score, 0.0f);
}

TEST(IntentRouterTest, MismatchedDimensionsThrow) {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {1.0f};
    EXPECT_THROW(preprocessor::IntentRouter::cosine_similarity(a, b), std::invalid_argument);
}
