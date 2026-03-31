#include <gtest/gtest.h>
#include "intent_router.hpp"
#include "i_embedding_engine.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal mock engine — returns a predetermined fixed embedding for any text.
// ---------------------------------------------------------------------------
class MockEmbeddingEngine : public preprocessor::IEmbeddingEngine {
public:
    explicit MockEmbeddingEngine(std::vector<float> fixed_embedding)
        : embedding_(std::move(fixed_embedding)) {}

    std::vector<float> generate_embedding(const std::string& /*text*/) override {
        return embedding_;
    }

    void set_embedding(std::vector<float> emb) { embedding_ = std::move(emb); }

private:
    std::vector<float> embedding_;
};

// ---------------------------------------------------------------------------
// Cosine similarity helpers (existing tests kept as-is)
// ---------------------------------------------------------------------------

TEST(IntentRouterTest, IdenticalVectorsReturnOne) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    float score = preprocessor::detail::cosine_similarity(a, b);
    EXPECT_NEAR(score, 1.0f, 1e-5f);
}

TEST(IntentRouterTest, OrthogonalVectorsReturnZero) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    float score = preprocessor::detail::cosine_similarity(a, b);
    EXPECT_NEAR(score, 0.0f, 1e-5f);
}

TEST(IntentRouterTest, OppositeVectorsReturnNegativeOne) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {-1.0f, 0.0f};
    float score = preprocessor::detail::cosine_similarity(a, b);
    EXPECT_NEAR(score, -1.0f, 1e-5f);
}

TEST(IntentRouterTest, EmptyVectorsReturnZero) {
    std::vector<float> a;
    std::vector<float> b;
    float score = preprocessor::detail::cosine_similarity(a, b);
    EXPECT_FLOAT_EQ(score, 0.0f);
}

TEST(IntentRouterTest, MismatchedDimensionsThrow) {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {1.0f};
    EXPECT_THROW(preprocessor::detail::cosine_similarity(a, b), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// IntentRouter structural behaviour tests (remove_intent / clear_intents /
// on_action) using the MockEmbeddingEngine.
// ---------------------------------------------------------------------------

// Helper: build a router + mock engine that always returns a unit vector.
static auto make_router(float threshold = 0.5f) {
    auto engine = std::make_shared<MockEmbeddingEngine>(std::vector<float>{1.0f, 0.0f, 0.0f});
    auto router = std::make_unique<preprocessor::IntentRouter>(threshold, engine);
    return std::make_pair(std::move(router), engine);
}

TEST(IntentRouterTest, RemoveIntentStopsRouting) {
    auto [router, engine] = make_router();

    router->add_intent("volume_down", "turn down the volume");
    // With one intent added and the mock always returning the same unit
    // vector, cosine similarity is 1.0 → above threshold.
    ASSERT_TRUE(router->route("anything").has_value());

    router->remove_intent("volume_down");
    // No intents remain → route() must return nullopt.
    EXPECT_FALSE(router->route("anything").has_value());
}

TEST(IntentRouterTest, RemoveIntentLeavesOthersIntact) {
    auto [router, engine] = make_router();

    router->add_intent("volume_down", "turn down the volume");
    router->add_intent("open_browser", "open the browser");

    router->remove_intent("volume_down");

    auto result = router->route("anything");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->intent_name, "open_browser");
}

TEST(IntentRouterTest, ClearIntentsDisablesRouting) {
    auto [router, engine] = make_router();

    router->add_intent("volume_down", "turn down volume");
    router->add_intent("open_file", "open the file");

    router->clear_intents();

    EXPECT_FALSE(router->route("anything").has_value());
}

TEST(IntentRouterTest, OnActionCallbackInvokedOnSuccessfulRoute) {
    auto [router, engine] = make_router();

    router->add_intent("volume_down", "turn down volume");

    bool callback_called = false;
    std::string received_intent;
    float received_score = -1.0f;

    router->on_action([&](const preprocessor::RouteResult& result, const std::string& /*input*/) {
        callback_called = true;
        received_intent = result.intent_name;
        received_score = result.score;
    });

    auto result = router->route("turn down the volume");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(callback_called);
    EXPECT_EQ(received_intent, "volume_down");
    EXPECT_GT(received_score, 0.5f);
}

TEST(IntentRouterTest, OnActionCallbackNotInvokedWhenNoMatch) {
    // Use threshold of 1.1 so nothing ever matches.
    auto engine = std::make_shared<MockEmbeddingEngine>(std::vector<float>{1.0f, 0.0f, 0.0f});
    preprocessor::IntentRouter router(1.0f, engine);

    // Provide a different intent embedding so cosine < 1.0 and no match.
    engine->set_embedding({0.0f, 1.0f, 0.0f}); // orthogonal to intent embedding
    router.add_intent("volume_down", "turn down volume");
    engine->set_embedding({1.0f, 0.0f, 0.0f}); // restore for route()

    bool callback_called = false;
    router.on_action([&](const preprocessor::RouteResult&, const std::string&) {
        callback_called = true;
    });

    // The intent was registered with {0,1,0}, but route() input embedding is
    // {1,0,0} → cosine similarity is 0.0 → below threshold 1.0 → no match.
    auto result = router.route("something");
    EXPECT_FALSE(result.has_value());
    EXPECT_FALSE(callback_called);
}

