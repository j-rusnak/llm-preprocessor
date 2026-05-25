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

TEST_F(ConfigLoaderTest, LoadsProxySecuritySettings) {
    write_config(R"({
        "proxy_auth_bearer_tokens": ["local-a", "local-b"],
        "proxy_auth_hmac_secret": "shared-secret",
        "proxy_auth_max_clock_skew_seconds": 90,
        "proxy_rate_limit_tokens_per_second": 12.5,
        "proxy_rate_limit_burst": 30,
        "proxy_max_request_bytes": 1048576,
        "sync_cache_export_limit": 25,
        "sync_vector_export_limit": 50,
        "tokenizer_mode": "model-calibrated",
        "proxy_forward_client_authorization": false,
        "allow_unsafe_remote_proxy": false
    })");
    auto config = preprocessor::ConfigLoader::load(temp_path_);

    ASSERT_EQ(config.proxy_auth_bearer_tokens.size(), 2u);
    EXPECT_EQ(config.proxy_auth_bearer_tokens[0], "local-a");
    EXPECT_EQ(config.proxy_auth_bearer_tokens[1], "local-b");
    EXPECT_EQ(config.proxy_auth_hmac_secret, "shared-secret");
    EXPECT_EQ(config.proxy_auth_max_clock_skew_seconds, 90);
    EXPECT_DOUBLE_EQ(config.proxy_rate_limit_tokens_per_second, 12.5);
    EXPECT_DOUBLE_EQ(config.proxy_rate_limit_burst, 30.0);
    EXPECT_EQ(config.proxy_max_request_bytes, 1048576u);
    EXPECT_EQ(config.sync_cache_export_limit, 25u);
    EXPECT_EQ(config.sync_vector_export_limit, 50u);
    EXPECT_EQ(config.tokenizer_mode, "model-calibrated");
    EXPECT_FALSE(config.proxy_forward_client_authorization);
    EXPECT_FALSE(config.allow_unsafe_remote_proxy);
}

TEST_F(ConfigLoaderTest, RejectsNonLoopbackProxyWithoutAuthByDefault) {
    write_config(R"({"proxy_host": "0.0.0.0"})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, AllowsNonLoopbackProxyWithBearerAuth) {
    write_config(R"({
        "proxy_host": "0.0.0.0",
        "proxy_auth_bearer_tokens": ["local-token"]
    })");
    auto config = preprocessor::ConfigLoader::load(temp_path_);
    EXPECT_EQ(config.proxy_host, "0.0.0.0");
    ASSERT_EQ(config.proxy_auth_bearer_tokens.size(), 1u);
}

TEST_F(ConfigLoaderTest, AllowsNonLoopbackProxyWithExplicitUnsafeFlag) {
    write_config(R"({
        "proxy_host": "0.0.0.0",
        "allow_unsafe_remote_proxy": true
    })");
    auto config = preprocessor::ConfigLoader::load(temp_path_);
    EXPECT_EQ(config.proxy_host, "0.0.0.0");
    EXPECT_TRUE(config.allow_unsafe_remote_proxy);
}

TEST_F(ConfigLoaderTest, RejectsIncompleteRateLimitConfig) {
    write_config(R"({"proxy_rate_limit_tokens_per_second": 10})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsNegativeProxyMaxRequestBytes) {
    write_config(R"({"proxy_max_request_bytes": -1})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsUnknownTokenizerMode) {
    write_config(R"({"tokenizer_mode": "exact-magic"})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, LoadsModelRouterConfig) {
    write_config(R"({
        "model_tiers": [
            {
                "name": "cheap",
                "upstream_url": "https://cheap.example/v1/chat/completions",
                "model_name": "gpt-cheap",
                "api_key": "cheap-key",
                "max_context": 4000
            },
            {
                "name": "frontier",
                "upstream_url": "https://frontier.example/v1/chat/completions",
                "model_name": "gpt-frontier"
            }
        ],
        "model_routes": [
            {
                "bucket": "code_edit",
                "min_request_chars": 0,
                "max_request_chars": 2000,
                "tier": "cheap"
            },
            {
                "bucket": "code_edit",
                "min_request_chars": 2001,
                "tier": "frontier"
            }
        ]
    })");

    auto config = preprocessor::ConfigLoader::load(temp_path_);

    ASSERT_EQ(config.model_tiers.size(), 2u);
    EXPECT_EQ(config.model_tiers[0].name, "cheap");
    EXPECT_EQ(config.model_tiers[0].upstream_url,
              "https://cheap.example/v1/chat/completions");
    EXPECT_EQ(config.model_tiers[0].model_name, "gpt-cheap");
    EXPECT_EQ(config.model_tiers[0].api_key, "cheap-key");
    EXPECT_EQ(config.model_tiers[0].max_context, 4000u);
    EXPECT_EQ(config.model_tiers[1].name, "frontier");
    EXPECT_EQ(config.model_tiers[1].model_name, "gpt-frontier");

    ASSERT_EQ(config.model_routes.size(), 2u);
    EXPECT_EQ(config.model_routes[0].bucket, preprocessor::PromptBucket::CodeEdit);
    EXPECT_EQ(config.model_routes[0].max_request_chars, 2000u);
    EXPECT_EQ(config.model_routes[0].tier, "cheap");
    EXPECT_EQ(config.model_routes[1].min_request_chars, 2001u);
    EXPECT_EQ(config.model_routes[1].tier, "frontier");
}

TEST_F(ConfigLoaderTest, RejectsModelRouteWithUnknownTier) {
    write_config(R"({
        "model_routes": [
            {"bucket": "code_edit", "tier": "missing"}
        ]
    })");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsModelRouteWithInvalidBounds) {
    write_config(R"({
        "model_tiers": [
            {"name": "cheap", "model_name": "gpt-cheap"}
        ],
        "model_routes": [
            {
                "bucket": "code_edit",
                "min_request_chars": 100,
                "max_request_chars": 10,
                "tier": "cheap"
            }
        ]
    })");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}
