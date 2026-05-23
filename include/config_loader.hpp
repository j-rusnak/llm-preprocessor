#pragma once

#include <string>
#include <vector>
#include <utility>
#include <optional>

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

    // --- Phase 2: prompt optimiser settings (all optional) ---
    bool prompt_optimizer_enabled = false;
    std::string prompt_templates_path;  // empty = use built-in defaults
    bool include_project_card = true;

    // --- Phase 3: symbol graph + structural fast path (all optional) ---
    bool symbol_graph_enabled = false;
    bool graph_expansion_enabled = true;       // requires symbol_graph_enabled
    bool structural_fast_path_enabled = true;  // requires symbol_graph_enabled
};

class ConfigLoader {
public:
    static Config load(const std::string& filepath);
};

} // namespace preprocessor
