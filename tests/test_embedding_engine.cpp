#include <gtest/gtest.h>

#include "embedding_engine.hpp"
#include "tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>

TEST(EmbeddingEngineTest, RejectsNullTokenizer) {
    EXPECT_THROW(
        preprocessor::EmbeddingEngine("models/model.ort", nullptr),
        std::invalid_argument);
}

TEST(EmbeddingEngineTest, RejectsInvalidModelPath) {
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

class EmbeddingEngineIntegrationTest : public ::testing::Test {
protected:
    static bool model_available() {
        return std::filesystem::exists("models/model.ort") &&
               std::filesystem::exists("models/vocab.txt");
    }
};

TEST_F(EmbeddingEngineIntegrationTest, ProducesNormalizedEmbedding) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found; skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    preprocessor::EmbeddingEngine engine("models/model.ort", tok);

    auto embedding = engine.generate_embedding("find the retry logic in openai_proxy.cpp");

    EXPECT_EQ(embedding.size(), 384u);

    const float magnitude = std::sqrt(
        std::inner_product(embedding.begin(), embedding.end(), embedding.begin(), 0.0f));
    EXPECT_NEAR(magnitude, 1.0f, 1e-3f);
}

TEST_F(EmbeddingEngineIntegrationTest, DifferentCodingQueriesProduceDifferentEmbeddings) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found; skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    preprocessor::EmbeddingEngine engine("models/model.ort", tok);

    auto emb1 = engine.generate_embedding("explain the request size guard in openai_proxy.cpp");
    auto emb2 = engine.generate_embedding("summarize the vector cache schema in embedding_cache.cpp");

    ASSERT_EQ(emb1.size(), emb2.size());

    const float dot = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);
    EXPECT_LT(dot, 0.95f);
}

TEST_F(EmbeddingEngineIntegrationTest, SimilarCodingQueriesProduceHighSimilarity) {
    if (!model_available()) {
        GTEST_SKIP() << "Model files not found; skipping integration test";
    }

    auto tok = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
    preprocessor::EmbeddingEngine engine("models/model.ort", tok);

    auto emb1 = engine.generate_embedding("fix the null dereference in repo_index.cpp");
    auto emb2 = engine.generate_embedding("repair the nullptr crash in repo_index.cpp");

    const float dot = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);
    EXPECT_GT(dot, 0.70f);
}
