#pragma once

#include <string>
#include <vector>
#include <utility>
#include <optional>
#include <cstddef>

#include "model_router.hpp"

namespace preprocessor {

struct Config {
    std::string model_path;
    std::string vocab_path;
    std::string db_path;
    std::string system_prompt;
    float similarity_threshold;
    int history_limit;
    std::vector<std::pair<std::string, std::vector<std::string>>> intents;

    // Optional API request parameters (for complete payload mode).
    std::optional<std::string> api_model;
    std::optional<std::string> api_endpoint;
    std::optional<float> temperature;
    std::optional<int> max_tokens;

    // --- Phase 1: proxy + retrieval settings (all optional) ---
    std::string proxy_host = "127.0.0.1";
    int proxy_port = 8088;
    std::string repo_root;                    // empty = don't index a repo
    std::string cache_db_path = "prompt_cache.db";
    std::size_t retrieval_k = 6;
    std::size_t embedding_dim = 384;
    std::size_t max_context_chars = 8000;
    std::string upstream_url = "https://api.openai.com/v1/chat/completions";
    std::string upstream_api_key;
    std::vector<ModelTier> model_tiers;
    std::vector<ModelRoute> model_routes;

    // --- Phase 12: secure proxy runtime settings (all optional) ---
    std::vector<std::string> proxy_auth_bearer_tokens;
    std::string proxy_auth_hmac_secret;
    int proxy_auth_max_clock_skew_seconds = 300;
    double proxy_rate_limit_tokens_per_second = 0.0;
    double proxy_rate_limit_burst = 0.0;
    std::size_t proxy_max_request_bytes = 8 * 1024 * 1024;
    std::size_t sync_cache_export_limit = 1000;
    std::size_t sync_vector_export_limit = 1000;
    bool proxy_forward_client_authorization = true;
    bool allow_unsafe_remote_proxy = false;

    // --- Phase 2: prompt optimiser settings (all optional) ---
    bool prompt_optimizer_enabled = false;
    std::string prompt_templates_path;  // empty = use built-in defaults
    bool include_project_card = true;

    // --- Phase 3: symbol graph + structural fast path (all optional) ---
    bool symbol_graph_enabled = false;
    bool graph_expansion_enabled = true;       // requires symbol_graph_enabled
    bool structural_fast_path_enabled = true;  // requires symbol_graph_enabled

    // --- Phase 5: prompt rewriter / context compression (all optional) ---
    bool prompt_rewriter_enabled = false;
    /// One of: "heuristic" (always available) or "llama-cpp" (requires
    /// the LLM_PREPROCESSOR_WITH_LLAMA_CPP build flag).
    std::string prompt_rewriter_kind = "heuristic";
    /// Soft char cap passed to the rewriter (0 = inherit max_context_chars).
    std::size_t prompt_rewriter_max_chars = 0;
    /// Path to a .gguf model file when prompt_rewriter_kind == "llama-cpp".
    std::string llama_model_path;
};

class ConfigLoader {
public:
    static Config load(const std::string& filepath);
};

} // namespace preprocessor
