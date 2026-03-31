#include <gtest/gtest.h>
#include "embedding_engine.hpp"
#include "intent_router.hpp"
#include "tokenizer.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <cmath>
#include <numeric>

TEST(EmbeddingEngineTest, RejectsNullTokenizer) {
    EXPECT_THROW(
        preprocessor::EmbeddingEngine("models/model.ort", nullptr),
        std::invalid_argument);
}

TEST(EmbeddingEngineTest, RejectsInvalidModelPath) {
    // Create a minimal tokenizer vocab for construction.
    {
        std::ofstream f("test_ee_vocab_tmp.txt");
        f << "[PAD]\n[UNK]\n[CLS]\n[SEP]\n";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("test_ee_vocab_tmp.txt");
    EXPECT_THROW(
        preprocessor::EmbeddingEngine("nonexistent_model.onnx", tok),
        std::runtime_error);

    std::remove("test_ee_vocab_tmp.txt");
}

// Integration tests — only run when the real model files are available.
class EmbeddingEngineIntegrationTest : public ::testing::Test {
protected:
    static bool model_available() {
        return std::filesystem::exists("models/model.ort") &&
               std::filesystem::exists("models/vocab.txt");
    }
};

TEST_F(EmbeddingEngineIntegrationTest, ProducesNormalizedEmbedding) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    preprocessor::EmbeddingEngine engine("models/model.ort", tok);

    auto embedding = engine.generate_embedding("hello world");

    // all-MiniLM-L6-v2 produces 384-dim embeddings
    EXPECT_EQ(embedding.size(), 384u);

    // Should be L2-normalized (magnitude ≈ 1.0)
    float magnitude = std::sqrt(
        std::inner_product(embedding.begin(), embedding.end(), embedding.begin(), 0.0f));
    EXPECT_NEAR(magnitude, 1.0f, 1e-3f);
}

TEST_F(EmbeddingEngineIntegrationTest, DifferentInputsProduceDifferentEmbeddings) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    preprocessor::EmbeddingEngine engine("models/model.ort", tok);

    auto emb1 = engine.generate_embedding("open the web browser");
    auto emb2 = engine.generate_embedding("the meaning of life is complex");

    ASSERT_EQ(emb1.size(), emb2.size());

    // Embeddings for semantically different inputs should not be identical.
    float dot = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);
    EXPECT_LT(dot, 0.95f); // Not near-identical
}

TEST_F(EmbeddingEngineIntegrationTest, SimilarInputsProduceHighSimilarity) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    preprocessor::EmbeddingEngine engine("models/model.ort", tok);

    auto emb1 = engine.generate_embedding("turn up the volume");
    auto emb2 = engine.generate_embedding("increase the volume");

    float dot = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);
    EXPECT_GT(dot, 0.75f); // Semantically similar
}

// Integration tests for multi-example intent routing.
TEST_F(EmbeddingEngineIntegrationTest, MultiExampleMuteMatches) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    auto engine = std::make_shared<preprocessor::EmbeddingEngine>("models/model.ort", tok);
    preprocessor::IntentRouter router(0.65f, engine);

    // Register multiple examples for mute
    router.add_intent("ACTION_MUTE", "mute the sound");
    router.add_intent("ACTION_MUTE", "silence the audio");
    router.add_intent("ACTION_MUTE", "please mute");
    router.add_intent("ACTION_MUTE", "turn off the sound");

    auto result = router.route("please mute");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "ACTION_MUTE");
}

TEST_F(EmbeddingEngineIntegrationTest, MultiExampleVolumeDownMatches) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    auto engine = std::make_shared<preprocessor::EmbeddingEngine>("models/model.ort", tok);
    preprocessor::IntentRouter router(0.65f, engine);

    router.add_intent("ACTION_DECREASE_VOLUME", "turn down the volume");
    router.add_intent("ACTION_DECREASE_VOLUME", "lower the volume");
    router.add_intent("ACTION_DECREASE_VOLUME", "make it quieter");
    router.add_intent("ACTION_DECREASE_VOLUME", "reduce the volume please");

    auto result = router.route("turn the volume down please");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "ACTION_DECREASE_VOLUME");
}

TEST_F(EmbeddingEngineIntegrationTest, MultiExampleDoesNotFalseMatch) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    auto engine = std::make_shared<preprocessor::EmbeddingEngine>("models/model.ort", tok);
    preprocessor::IntentRouter router(0.75f, engine);

    router.add_intent("ACTION_MUTE", "mute the sound");
    router.add_intent("ACTION_MUTE", "silence the audio");

    // Unrelated input should not match
    auto result = router.route("what is the weather forecast for tomorrow");
    EXPECT_FALSE(result.has_value());
}

// Sliding-window subphrase routing tests.
TEST_F(EmbeddingEngineIntegrationTest, NoisyMuteInputMatchesViaSubphrase) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    auto engine = std::make_shared<preprocessor::EmbeddingEngine>("models/model.ort", tok);
    preprocessor::IntentRouter router(0.75f, engine);

    router.add_intent("ACTION_MUTE", "mute the sound");
    router.add_intent("ACTION_MUTE", "silence the audio");
    router.add_intent("ACTION_MUTE", "please mute");
    router.add_intent("ACTION_MUTE", "turn off the sound");

    auto result = router.route("yo bro can you mute that");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "ACTION_MUTE");
}

TEST_F(EmbeddingEngineIntegrationTest, NoisyVolumeDownMatchesViaSubphrase) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    auto engine = std::make_shared<preprocessor::EmbeddingEngine>("models/model.ort", tok);
    preprocessor::IntentRouter router(0.65f, engine);

    router.add_intent("ACTION_DECREASE_VOLUME", "turn down the volume");
    router.add_intent("ACTION_DECREASE_VOLUME", "lower the volume");
    router.add_intent("ACTION_DECREASE_VOLUME", "make it quieter");
    router.add_intent("ACTION_DECREASE_VOLUME", "reduce the volume please");

    auto result = router.route("hey computer turn the volume down please");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "ACTION_DECREASE_VOLUME");
}

TEST_F(EmbeddingEngineIntegrationTest, SubphraseDoesNotFalseMatch) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    auto engine = std::make_shared<preprocessor::EmbeddingEngine>("models/model.ort", tok);
    preprocessor::IntentRouter router(0.75f, engine);

    router.add_intent("ACTION_MUTE", "mute the sound");
    router.add_intent("ACTION_OPEN_BROWSER", "open the web browser");

    // Long unrelated input — no subphrase should match
    auto result = router.route("hey dude what do you think about the economy these days");
    EXPECT_FALSE(result.has_value());
}

// Stop-word filtering tests.
TEST_F(EmbeddingEngineIntegrationTest, StopWordFilteredVolumeDown) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    auto engine = std::make_shared<preprocessor::EmbeddingEngine>("models/model.ort", tok);
    preprocessor::IntentRouter router(0.65f, engine);

    router.add_intent("ACTION_DECREASE_VOLUME", "turn down the volume");
    router.add_intent("ACTION_DECREASE_VOLUME", "lower the volume");
    router.add_intent("ACTION_DECREASE_VOLUME", "make it quieter");
    router.add_intent("ACTION_DECREASE_VOLUME", "reduce the volume please");

    // After stop-word removal: "turn volume down" — matches "turn down the volume"
    auto result = router.route("could you turn the volume down for me");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "ACTION_DECREASE_VOLUME");
}

TEST_F(EmbeddingEngineIntegrationTest, StopWordFilteredMute) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found — skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    auto engine = std::make_shared<preprocessor::EmbeddingEngine>("models/model.ort", tok);
    preprocessor::IntentRouter router(0.75f, engine);

    router.add_intent("ACTION_MUTE", "mute the sound");
    router.add_intent("ACTION_MUTE", "silence the audio");

    // After stop-word removal: "mute thing"
    auto result = router.route("could you please mute that thing");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "ACTION_MUTE");
}
