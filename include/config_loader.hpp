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
};

class ConfigLoader {
public:
    static Config load(const std::string& filepath);
};

} // namespace preprocessor
