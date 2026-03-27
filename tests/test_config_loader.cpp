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
    EXPECT_FLOAT_EQ(config.similarity_threshold, 0.75f);
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
    EXPECT_EQ(config.intents[0].second, "increase volume");
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
