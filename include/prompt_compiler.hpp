#pragma once

#include <string>
#include <vector>
#include <utility>
#include <optional>

#include <nlohmann/json.hpp>

namespace preprocessor {

struct ApiParams {
    std::optional<std::string> model;
    std::optional<float> temperature;
    std::optional<int> max_tokens;
};

class PromptCompiler {
public:
    explicit PromptCompiler(const std::string& system_prompt);

    // Returns messages-only JSON array as a string (backward-compatible).
    std::string build_payload(
        const std::string& user_input,
        const std::string& retrieved_context,
        const std::vector<std::pair<std::string, std::string>>& history) const;

    // Returns a JSON object. When api_params.model is set, wraps messages in a
    // complete API request body including model, temperature (if set), and
    // max_tokens (if set). Without model, returns {"messages": [...]}.
    // Throws std::invalid_argument if temperature or max_tokens are provided
    // without model.
    nlohmann::json build_payload_json(
        const std::string& user_input,
        const std::string& retrieved_context,
        const std::vector<std::pair<std::string, std::string>>& history,
        const ApiParams& api_params = {}) const;

private:
    std::string system_prompt_;
};

} // namespace preprocessor
