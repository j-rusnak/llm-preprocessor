#include <gtest/gtest.h>
#include "embedding_engine.hpp"
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
