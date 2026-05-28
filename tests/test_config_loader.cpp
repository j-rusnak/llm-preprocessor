#include <gtest/gtest.h>
#include "config_loader.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <string>
#include <vector>

// Helper to write a temp config file and clean it up.
class ConfigLoaderTest : public ::testing::Test {
protected:
    std::string temp_path_ = "test_config_tmp.json";

    void write_config(const std::string& content) {
        std::ofstream f(temp_path_);
        f << content;
    }

    std::string repo_file(const std::string& relative_path) const {
        namespace fs = std::filesystem;
        if (fs::exists(relative_path)) return relative_path;
        auto from_build_dir = fs::path("..") / relative_path;
        if (fs::exists(from_build_dir)) return from_build_dir.string();
        return relative_path;
    }

    std::string read_file(const std::string& path) const {
        std::ifstream f(path);
        return std::string(std::istreambuf_iterator<char>(f),
                           std::istreambuf_iterator<char>());
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
    EXPECT_EQ(config.proxy_host, "127.0.0.1");
    EXPECT_EQ(config.proxy_port, 8088);
    EXPECT_EQ(config.cache_db_path, "prompt_cache.db");
    EXPECT_EQ(config.upstream_url, "https://api.openai.com/v1/chat/completions");
}

TEST_F(ConfigLoaderTest, LoadsCustomValues) {
    write_config(R"({
        "model_path": "custom/model.onnx",
        "vocab_path": "custom/vocab.txt",
        "proxy_host": "127.0.0.2",
        "proxy_port": 9090,
        "repo_root": "src",
        "retrieval_k": 9,
        "max_context_chars": 16000
    })");
    auto config = preprocessor::ConfigLoader::load(temp_path_);

    EXPECT_EQ(config.model_path, "custom/model.onnx");
    EXPECT_EQ(config.vocab_path, "custom/vocab.txt");
    EXPECT_EQ(config.proxy_host, "127.0.0.2");
    EXPECT_EQ(config.proxy_port, 9090);
    EXPECT_EQ(config.repo_root, "src");
    EXPECT_EQ(config.retrieval_k, 9u);
    EXPECT_EQ(config.max_context_chars, 16000u);
}

TEST_F(ConfigLoaderTest, RejectsLegacyCommandRouterKeys) {
    const std::vector<std::string> legacy_configs = {
        R"({"db_path": "history.db"})",
        R"({"system_prompt": "You are a helpful assistant."})",
        R"({"similarity_threshold": 0.65})",
        R"({"history_limit": 10})",
        R"({"intents": []})",
        R"({"api_model": "gpt-4"})",
        R"({"api_endpoint": "https://api.openai.com/v1/chat/completions"})",
        R"({"temperature": 0.7})",
        R"({"max_tokens": 2048})"
    };

    for (const auto& json : legacy_configs) {
        write_config(json);
        EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_),
                     std::invalid_argument) << json;
    }
}

TEST_F(ConfigLoaderTest, RejectsMissingFile) {
    EXPECT_THROW(preprocessor::ConfigLoader::load("nonexistent.json"), std::runtime_error);
}

TEST_F(ConfigLoaderTest, RejectsInvalidJson) {
    write_config("not json at all");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::runtime_error);
}

TEST_F(ConfigLoaderTest, RejectsEmptyModelPath) {
    write_config(R"({"model_path": ""})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsEmptyVocabPath) {
    write_config(R"({"vocab_path": ""})");
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
        "upstream_timeout_seconds": 45,
        "upstream_connect_timeout_seconds": 4,
        "upstream_max_response_bytes": 2097152,
        "stream_idle_timeout_seconds": 20,
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
    EXPECT_EQ(config.upstream_timeout_seconds, 45);
    EXPECT_EQ(config.upstream_connect_timeout_seconds, 4);
    EXPECT_EQ(config.upstream_max_response_bytes, 2097152u);
    EXPECT_EQ(config.stream_idle_timeout_seconds, 20);
    EXPECT_EQ(config.sync_cache_export_limit, 25u);
    EXPECT_EQ(config.sync_vector_export_limit, 50u);
    EXPECT_EQ(config.tokenizer_mode, "model-calibrated");
    EXPECT_FALSE(config.proxy_forward_client_authorization);
    EXPECT_FALSE(config.allow_unsafe_remote_proxy);
}

TEST_F(ConfigLoaderTest, LoadsProductionExampleConfig) {
    const auto path = repo_file("config.production.example.json");
    ASSERT_TRUE(std::filesystem::exists(path)) << path;

    auto config = preprocessor::ConfigLoader::load(path);

    EXPECT_EQ(config.proxy_host, "127.0.0.1");
    EXPECT_FALSE(config.allow_unsafe_remote_proxy);
    EXPECT_FALSE(config.proxy_auth_bearer_tokens.empty());
    EXPECT_FALSE(config.proxy_forward_client_authorization);
    EXPECT_GT(config.proxy_max_request_bytes, 0u);
    EXPECT_FALSE(config.upstream_url.empty());
    EXPECT_EQ(config.upstream_timeout_seconds, 60);
    EXPECT_EQ(config.upstream_connect_timeout_seconds, 10);
    EXPECT_EQ(config.upstream_max_response_bytes, 8388608u);
    EXPECT_EQ(config.stream_idle_timeout_seconds, 30);
    EXPECT_EQ(config.sync_cache_export_limit, 100u);
    EXPECT_EQ(config.sync_vector_export_limit, 100u);
    EXPECT_EQ(config.tokenizer_mode, "model-calibrated");
    EXPECT_TRUE(config.prompt_optimizer_enabled);
    EXPECT_TRUE(config.symbol_graph_enabled);
    EXPECT_TRUE(config.graph_expansion_enabled);
    EXPECT_TRUE(config.structural_fast_path_enabled);
    EXPECT_TRUE(config.prompt_rewriter_enabled);
    EXPECT_EQ(config.prompt_rewriter_kind, "heuristic");
    EXPECT_EQ(config.prompt_rewriter_max_chars, 8000u);
}

TEST_F(ConfigLoaderTest, RejectsSecuredLanExampleUntilTokenReplaced) {
    const auto path = repo_file("config.lan.example.json");
    ASSERT_TRUE(std::filesystem::exists(path)) << path;

    EXPECT_THROW(preprocessor::ConfigLoader::load(path), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, LoadsCopiedLanExampleAfterPlaceholderTokenReplaced) {
    const auto path = repo_file("config.lan.example.json");
    ASSERT_TRUE(std::filesystem::exists(path)) << path;

    auto content = read_file(path);
    const std::string placeholder = "replace-with-strong-lan-proxy-token";
    const std::string deployment_token = "team-lan-proxy-token-alpha";
    const auto pos = content.find(placeholder);
    ASSERT_NE(pos, std::string::npos);
    content.replace(pos, placeholder.size(), deployment_token);
    write_config(content);

    auto config = preprocessor::ConfigLoader::load(temp_path_);

    EXPECT_EQ(config.proxy_host, "0.0.0.0");
    ASSERT_EQ(config.proxy_auth_bearer_tokens.size(), 1u);
    EXPECT_EQ(config.proxy_auth_bearer_tokens[0], deployment_token);
    EXPECT_FALSE(config.allow_unsafe_remote_proxy);
    EXPECT_FALSE(config.proxy_forward_client_authorization);
    EXPECT_GT(config.proxy_max_request_bytes, 0u);
    EXPECT_GT(config.proxy_rate_limit_tokens_per_second, 0.0);
    EXPECT_GT(config.proxy_rate_limit_burst, 0.0);
}

TEST_F(ConfigLoaderTest, RejectsNonLoopbackProxyWithoutAuthByDefault) {
    write_config(R"({"proxy_host": "0.0.0.0"})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsRemoteServingUnlessUnsafeFlagExplicitlyTrue) {
    write_config(R"({
        "proxy_host": "0.0.0.0",
        "allow_unsafe_remote_proxy": false
    })");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_),
                 std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsEmptyUpstreamUrl) {
    write_config(R"({"upstream_url": ""})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsInvalidUpstreamTimeouts) {
    write_config(R"({"upstream_timeout_seconds": 0})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);

    write_config(R"({"upstream_connect_timeout_seconds": 0})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);

    write_config(R"({"stream_idle_timeout_seconds": -1})");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, RejectsNonLoopbackProxyWithPlaceholderBearer) {
    const std::vector<std::string> placeholders = {
        "replace-with-local-proxy-token",
        "replace-with-strong-local-token",
        "replace-with-strong-lan-proxy-token",
        "replace-with-strong-lan-proxy-token",
        "local-lan-token-changeme-32chars",
        "placeholder-lan-token",
        "example-lan-token"
    };

    for (const auto& token : placeholders) {
        write_config(std::string(R"({
            "proxy_host": "0.0.0.0",
            "proxy_auth_bearer_tokens": [")") + token + R"("]
        })");
        EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_),
                     std::invalid_argument) << token;
    }
}

TEST_F(ConfigLoaderTest, RejectsNonLoopbackProxyWithoutRequestSizeLimit) {
    write_config(R"({
        "proxy_host": "0.0.0.0",
        "proxy_auth_bearer_tokens": ["local-token"],
        "proxy_max_request_bytes": 0
    })");
    EXPECT_THROW(preprocessor::ConfigLoader::load(temp_path_), std::invalid_argument);
}

TEST_F(ConfigLoaderTest, AllowsNonLoopbackProxyWithBearerAuth) {
    write_config(R"({
        "proxy_host": "0.0.0.0",
        "proxy_auth_bearer_tokens": ["team-lan-proxy-token-alpha"],
        "proxy_forward_client_authorization": false,
        "proxy_rate_limit_tokens_per_second": 2.0,
        "proxy_rate_limit_burst": 10.0
    })");
    auto config = preprocessor::ConfigLoader::load(temp_path_);
    EXPECT_EQ(config.proxy_host, "0.0.0.0");
    ASSERT_EQ(config.proxy_auth_bearer_tokens.size(), 1u);
    EXPECT_EQ(config.proxy_auth_bearer_tokens[0],
              "team-lan-proxy-token-alpha");
    EXPECT_FALSE(config.proxy_forward_client_authorization);
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
