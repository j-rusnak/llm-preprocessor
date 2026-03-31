#include <gtest/gtest.h>
#include "config_loader.hpp"

#include <fstream>
#include <cstdio>
#include <string>

// Helper to write a temp config file and clean it up.
class ConfigLoaderTest : public ::testing::Test {
protected:
    std::string temp_path_ = "test_config_tmp.json";

    void write_config(const std::string& content) {
        std::ofstream f(temp_path_);
        f << content;
    }

    void TearDown() override {
        std::remove(temp_path_.c_str());
    }
};

TEST_F(ConfigLoaderTest, LoadsDefaults) {
    write_config("{}");
    auto config = preprocessor::ConfigLoader::load(temp_path_);

    EXPECT_EQ(config.model_path, "models/model.onnx");
    EXPECT_EQ(config.vocab_path, "models/vocab.txt");
    EXPECT_EQ(config.db_path, "history.db");
    EXPECT_FLOAT_EQ(config.similarity_threshold, 0.65f);
    EXPECT_EQ(config.history_limit, 10);
    EXPECT_TRUE(config.intents.empty());
}

TEST_F(ConfigLoaderTest, LoadsCustomValues) {
    write_config(R"({
        "model_path": "custom/model.onnx",
        "similarity_threshold": 0.9,
        "history_limit": 20,
        "intents": [
            {"name": "VOLUME_UP", "example": "increase volume"}
        ]
    })");
    auto config = preprocessor::ConfigLoader::load(temp_path_);

    EXPECT_EQ(config.model_path, "custom/model.onnx");
    EXPECT_FLOAT_EQ(config.similarity_threshold, 0.9f);
    EXPECT_EQ(config.history_limit, 20);
    ASSERT_EQ(config.intents.size(), 1u);
    EXPECT_EQ(config.intents[0].first, "VOLUME_UP");
    ASSERT_EQ(config.intents[0].second.size(), 1u);
    EXPECT_EQ(config.intents[0].second[0], "increase volume");
}

TEST_F(ConfigLoaderTest, RejectsInvalidThreshold) {
    write_config(R"({"similarity_threshold": 1.5})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsInvalidHistoryLimit) {
    write_config(R"({"history_limit": 0})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsMissingFile) {
    EXPECT_THROW(preprocessor::ConfigLoader::load("nonexistent.json"), std::runtime_error);
}

TEST_F(ConfigLoaderTest, RejectsInvalidJson) {
    write_config("not json at all");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::runtime_error);
}

TEST_F(ConfigLoaderTest, RejectsIntentWithoutName) {
    write_config(R"({"intents": [{"example": "test"}]})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, LoadsMultipleExamples) {
    write_config(R"({
        "intents": [
            {"name": "MUTE", "examples": ["mute audio", "silence sound", "please mute"]}
        ]
    })");
    auto config = preprocessor::ConfigLoader::load(temp_path_);

    ASSERT_EQ(config.intents.size(), 1u);
    EXPECT_EQ(config.intents[0].first, "MUTE");
    ASSERT_EQ(config.intents[0].second.size(), 3u);
    EXPECT_EQ(config.intents[0].second[0], "mute audio");
    EXPECT_EQ(config.intents[0].second[1], "silence sound");
    EXPECT_EQ(config.intents[0].second[2], "please mute");
}

TEST_F(ConfigLoaderTest, BackwardCompatSingleExample) {
    write_config(R"({
        "intents": [
            {"name": "OPEN", "example": "open the file"}
        ]
    })");
    auto config = preprocessor::ConfigLoader::load(temp_path_);

    ASSERT_EQ(config.intents.size(), 1u);
    ASSERT_EQ(config.intents[0].second.size(), 1u);
    EXPECT_EQ(config.intents[0].second[0], "open the file");
}

TEST_F(ConfigLoaderTest, RejectsIntentWithNoExamples) {
    write_config(R"({"intents": [{"name": "EMPTY", "examples": []}]})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsIntentWithoutExampleOrExamples) {
    write_config(R"({"intents": [{"name": "NOEX"}]})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsEmptyModelPath) {
    write_config(R"({"model_path": ""})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsEmptyVocabPath) {
    write_config(R"({"vocab_path": ""})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsEmptyDbPath) {
    write_config(R"({"db_path": ""})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, LoadsOptionalApiParams) {
    write_config(R"({
        "api_model": "gpt-4",
        "api_endpoint": "https://api.openai.com/v1/chat/completions",
        "temperature": 0.7,
        "max_tokens": 2048
    })");
    auto config = preprocessor::ConfigLoader::load(temp_path_);

    ASSERT_TRUE(config.api_model.has_value());
    EXPECT_EQ(*config.api_model, "gpt-4");
    ASSERT_TRUE(config.api_endpoint.has_value());
    EXPECT_EQ(*config.api_endpoint, "https://api.openai.com/v1/chat/completions");
    ASSERT_TRUE(config.temperature.has_value());
    EXPECT_FLOAT_EQ(*config.temperature, 0.7f);
    ASSERT_TRUE(config.max_tokens.has_value());
    EXPECT_EQ(*config.max_tokens, 2048);
}

TEST_F(ConfigLoaderTest, ApiParamsAbsentByDefault) {
    write_config("{}");
    auto config = preprocessor::ConfigLoader::load(temp_path_);

    EXPECT_FALSE(config.api_model.has_value());
    EXPECT_FALSE(config.api_endpoint.has_value());
    EXPECT_FALSE(config.temperature.has_value());
    EXPECT_FALSE(config.max_tokens.has_value());
}

TEST_F(ConfigLoaderTest, RejectsInvalidTemperature) {
    write_config(R"({"temperature": 3.0})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsInvalidMaxTokens) {
    write_config(R"({"max_tokens": 0})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}
