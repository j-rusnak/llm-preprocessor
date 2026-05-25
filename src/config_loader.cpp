#include "config_loader.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace preprocessor {

namespace {

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() &&
           std::equal(prefix.begin(), prefix.end(), s.begin());
}

bool is_loopback_host(const std::string& host) {
    return host == "localhost" ||
           host == "::1" ||
           host == "[::1]" ||
           host == "127.0.0.1" ||
           starts_with(host, "127.");
}

bool proxy_auth_enabled(const Config& config) {
    return !config.proxy_auth_bearer_tokens.empty() ||
           !config.proxy_auth_hmac_secret.empty();
}

bool is_prompt_bucket_name(const std::string& name) {
    return name == "code_edit" ||
           name == "code_explain" ||
           name == "code_generate" ||
           name == "meta_query" ||
           name == "freeform";
}

std::string required_string_field(const nlohmann::json& j,
                                  const char* object_name,
                                  const char* field_name) {
    if (!j.contains(field_name) || !j[field_name].is_string() ||
        j[field_name].get<std::string>().empty()) {
        throw std::invalid_argument(std::string(object_name) + "." +
                                    field_name + " must be a non-empty string");
    }
    return j[field_name].get<std::string>();
}

std::size_t read_size_t_field(const nlohmann::json& j,
                              const char* name,
                              std::size_t fallback) {
    if (!j.contains(name)) return fallback;
    if (!j[name].is_number_integer()) {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    auto raw = j[name].get<long long>();
    if (raw < 0) {
        throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
    }
    auto max = static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max());
    if (static_cast<unsigned long long>(raw) > max) {
        throw std::invalid_argument(std::string(name) + " is too large");
    }
    return static_cast<std::size_t>(raw);
}

long read_long_field(const nlohmann::json& j,
                     const char* name,
                     long fallback,
                     long min_value) {
    if (!j.contains(name)) return fallback;
    if (!j[name].is_number_integer()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    const auto raw = j[name].get<long long>();
    if (raw < static_cast<long long>(min_value) ||
        raw > static_cast<long long>(std::numeric_limits<long>::max())) {
        throw std::invalid_argument(std::string(name) + " is out of range");
    }
    return static_cast<long>(raw);
}

bool contains_placeholder_proxy_token(const Config& config) {
    return std::find(config.proxy_auth_bearer_tokens.begin(),
                     config.proxy_auth_bearer_tokens.end(),
                     "replace-with-local-proxy-token") !=
           config.proxy_auth_bearer_tokens.end();
}

void reject_legacy_command_router_keys(const nlohmann::json& j) {
    for (const char* key : {
             "db_path",
             "system_prompt",
             "similarity_threshold",
             "history_limit",
             "intents",
             "api_model",
             "api_endpoint",
             "temperature",
             "max_tokens",
         }) {
        if (j.contains(key)) {
            throw std::invalid_argument(
                std::string("legacy command-router config key '") + key +
                "' is no longer supported; use --serve/--mcp proxy settings instead");
        }
    }
}

} // namespace

Config ConfigLoader::load(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + filepath);
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(file);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error(std::string("Invalid JSON in config file: ") + e.what());
    }

    reject_legacy_command_router_keys(j);

    Config config;

    config.model_path = j.value("model_path", "models/model.onnx");
    config.vocab_path = j.value("vocab_path", "models/vocab.txt");

    if (config.model_path.empty()) {
        throw std::invalid_argument("model_path must not be empty");
    }
    if (config.vocab_path.empty()) {
        throw std::invalid_argument("vocab_path must not be empty");
    }

    // --- Phase 1 fields (all optional). ---
    config.proxy_host = j.value("proxy_host", config.proxy_host);
    config.proxy_port = j.value("proxy_port", config.proxy_port);
    config.repo_root  = j.value("repo_root", config.repo_root);
    config.cache_db_path = j.value("cache_db_path", config.cache_db_path);
    config.retrieval_k = j.value("retrieval_k", config.retrieval_k);
    config.embedding_dim = j.value("embedding_dim", config.embedding_dim);
    config.max_context_chars = j.value("max_context_chars", config.max_context_chars);
    config.upstream_url = j.value("upstream_url", config.upstream_url);
    config.upstream_api_key = j.value("upstream_api_key", config.upstream_api_key);
    config.upstream_timeout_seconds =
        read_long_field(j, "upstream_timeout_seconds",
                        config.upstream_timeout_seconds, 1);
    config.upstream_connect_timeout_seconds =
        read_long_field(j, "upstream_connect_timeout_seconds",
                        config.upstream_connect_timeout_seconds, 1);
    config.upstream_max_response_bytes =
        read_size_t_field(j, "upstream_max_response_bytes",
                          config.upstream_max_response_bytes);
    config.stream_idle_timeout_seconds =
        read_long_field(j, "stream_idle_timeout_seconds",
                        config.stream_idle_timeout_seconds, 0);
    if (config.upstream_url.empty()) {
        throw std::invalid_argument("upstream_url must not be empty");
    }

    std::unordered_set<std::string> model_tier_names;
    if (j.contains("model_tiers")) {
        if (!j["model_tiers"].is_array()) {
            throw std::invalid_argument("model_tiers must be an array");
        }
        for (const auto& item : j["model_tiers"]) {
            if (!item.is_object()) {
                throw std::invalid_argument("model_tiers entries must be objects");
            }
            ModelTier tier;
            tier.name = required_string_field(item, "model_tiers[]", "name");
            tier.model_name =
                required_string_field(item, "model_tiers[]", "model_name");
            tier.upstream_url = item.value("upstream_url", std::string{});
            tier.api_key = item.value("api_key", std::string{});
            tier.max_context = read_size_t_field(item, "max_context", 0);
            if (!model_tier_names.insert(tier.name).second) {
                throw std::invalid_argument("model_tiers names must be unique");
            }
            config.model_tiers.push_back(std::move(tier));
        }
    }

    if (j.contains("model_routes")) {
        if (!j["model_routes"].is_array()) {
            throw std::invalid_argument("model_routes must be an array");
        }
        for (const auto& item : j["model_routes"]) {
            if (!item.is_object()) {
                throw std::invalid_argument("model_routes entries must be objects");
            }
            const std::string bucket =
                required_string_field(item, "model_routes[]", "bucket");
            if (!is_prompt_bucket_name(bucket)) {
                throw std::invalid_argument("model_routes[].bucket is unknown");
            }

            ModelRoute route;
            route.bucket = bucket_from_string(bucket);
            route.min_request_chars =
                read_size_t_field(item, "min_request_chars", 0);
            route.max_request_chars =
                read_size_t_field(item, "max_request_chars", 0);
            if (route.max_request_chars > 0 &&
                route.min_request_chars > route.max_request_chars) {
                throw std::invalid_argument(
                    "model_routes min_request_chars must be <= max_request_chars");
            }
            route.tier = required_string_field(item, "model_routes[]", "tier");
            if (model_tier_names.find(route.tier) == model_tier_names.end()) {
                throw std::invalid_argument(
                    "model_routes[].tier must reference a configured model_tier");
            }
            config.model_routes.push_back(std::move(route));
        }
    }
    if (config.proxy_port < 0 || config.proxy_port > 65535) {
        throw std::invalid_argument("proxy_port must be 0-65535");
    }
    if (config.embedding_dim == 0) {
        throw std::invalid_argument("embedding_dim must be > 0");
    }

    // --- Phase 12 secure proxy runtime fields (all optional). ---
    if (j.contains("proxy_auth_bearer_tokens")) {
        if (!j["proxy_auth_bearer_tokens"].is_array()) {
            throw std::invalid_argument("proxy_auth_bearer_tokens must be an array");
        }
        for (const auto& token : j["proxy_auth_bearer_tokens"]) {
            if (!token.is_string() || token.get<std::string>().empty()) {
                throw std::invalid_argument(
                    "proxy_auth_bearer_tokens must contain non-empty strings");
            }
            config.proxy_auth_bearer_tokens.push_back(token.get<std::string>());
        }
    }
    config.proxy_auth_hmac_secret = j.value("proxy_auth_hmac_secret",
                                            config.proxy_auth_hmac_secret);
    config.proxy_auth_max_clock_skew_seconds =
        j.value("proxy_auth_max_clock_skew_seconds",
                config.proxy_auth_max_clock_skew_seconds);
    config.proxy_rate_limit_tokens_per_second =
        j.value("proxy_rate_limit_tokens_per_second",
                config.proxy_rate_limit_tokens_per_second);
    config.proxy_rate_limit_burst =
        j.value("proxy_rate_limit_burst", config.proxy_rate_limit_burst);
    config.proxy_max_request_bytes =
        read_size_t_field(j, "proxy_max_request_bytes", config.proxy_max_request_bytes);
    config.sync_cache_export_limit =
        read_size_t_field(j, "sync_cache_export_limit", config.sync_cache_export_limit);
    config.sync_vector_export_limit =
        read_size_t_field(j, "sync_vector_export_limit", config.sync_vector_export_limit);
    config.tokenizer_mode = j.value("tokenizer_mode", config.tokenizer_mode);
    config.allow_unsafe_remote_proxy =
        j.value("allow_unsafe_remote_proxy", config.allow_unsafe_remote_proxy);
    if (j.contains("proxy_forward_client_authorization")) {
        config.proxy_forward_client_authorization =
            j["proxy_forward_client_authorization"].get<bool>();
    } else if (proxy_auth_enabled(config)) {
        config.proxy_forward_client_authorization = false;
    }

    if (config.proxy_auth_max_clock_skew_seconds < 0) {
        throw std::invalid_argument("proxy_auth_max_clock_skew_seconds must be >= 0");
    }
    if (config.proxy_rate_limit_tokens_per_second < 0.0 ||
        config.proxy_rate_limit_burst < 0.0) {
        throw std::invalid_argument("proxy rate limit values must be >= 0");
    }
    const bool rate_limit_disabled =
        config.proxy_rate_limit_tokens_per_second == 0.0 &&
        config.proxy_rate_limit_burst == 0.0;
    const bool rate_limit_complete =
        config.proxy_rate_limit_tokens_per_second > 0.0 &&
        config.proxy_rate_limit_burst > 0.0;
    if (!rate_limit_disabled && !rate_limit_complete) {
        throw std::invalid_argument(
            "proxy_rate_limit_tokens_per_second and proxy_rate_limit_burst must both be > 0, or both 0");
    }
    if (config.tokenizer_mode != "heuristic" &&
        config.tokenizer_mode != "model-calibrated") {
        throw std::invalid_argument(
            "tokenizer_mode must be 'heuristic' or 'model-calibrated'");
    }
    if (!is_loopback_host(config.proxy_host) &&
        !proxy_auth_enabled(config) &&
        !config.allow_unsafe_remote_proxy) {
        throw std::invalid_argument(
            "non-loopback proxy_host requires proxy auth or allow_unsafe_remote_proxy=true");
    }
    if (!is_loopback_host(config.proxy_host) &&
        contains_placeholder_proxy_token(config)) {
        throw std::invalid_argument(
            "non-loopback proxy_host must not use placeholder proxy auth tokens");
    }
    if (!is_loopback_host(config.proxy_host) &&
        config.proxy_max_request_bytes == 0 &&
        !config.allow_unsafe_remote_proxy) {
        throw std::invalid_argument(
            "non-loopback proxy_host requires positive proxy_max_request_bytes or allow_unsafe_remote_proxy=true");
    }

    // --- Phase 2 fields (all optional). ---
    config.prompt_optimizer_enabled = j.value("prompt_optimizer_enabled",
                                              config.prompt_optimizer_enabled);
    config.prompt_templates_path = j.value("prompt_templates_path",
                                           config.prompt_templates_path);
    config.include_project_card = j.value("include_project_card",
                                          config.include_project_card);

    // --- Phase 3 fields (all optional). ---
    config.symbol_graph_enabled = j.value("symbol_graph_enabled",
                                          config.symbol_graph_enabled);
    config.graph_expansion_enabled = j.value("graph_expansion_enabled",
                                             config.graph_expansion_enabled);
    config.structural_fast_path_enabled = j.value("structural_fast_path_enabled",
                                                  config.structural_fast_path_enabled);

    // --- Phase 5 fields (all optional). ---
    config.prompt_rewriter_enabled = j.value("prompt_rewriter_enabled",
                                             config.prompt_rewriter_enabled);
    config.prompt_rewriter_kind = j.value("prompt_rewriter_kind",
                                          config.prompt_rewriter_kind);
    config.prompt_rewriter_max_chars = j.value("prompt_rewriter_max_chars",
                                               config.prompt_rewriter_max_chars);
    config.llama_model_path = j.value("llama_model_path", config.llama_model_path);

    return config;
}

} // namespace preprocessor
