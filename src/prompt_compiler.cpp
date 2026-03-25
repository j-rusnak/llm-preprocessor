#include "prompt_compiler.hpp"

#include <nlohmann/json.hpp>

namespace preprocessor {

PromptCompiler::PromptCompiler(const std::string& system_prompt)
    : system_prompt_(system_prompt) {}

std::string PromptCompiler::build_payload(
        const std::string& user_input,
        const std::string& retrieved_context,
        const std::vector<std::pair<std::string, std::string>>& history) const {

    nlohmann::json messages = nlohmann::json::array();

    // 1. System message
    messages.push_back({{"role", "system"}, {"content", system_prompt_}});

    // 2. Conversation history
    for (const auto& [role, content] : history) {
        messages.push_back({{"role", role}, {"content", content}});
    }

    // 3. Current user message, enriched with context if available
    std::string final_content;
    if (!retrieved_context.empty()) {
        final_content =
            "Use the following context to answer the question:\n\n<context>" +
            retrieved_context +
            "</context>\n\nQuestion: " + user_input;
    } else {
        final_content = user_input;
    }
    messages.push_back({{"role", "user"}, {"content", final_content}});

    return messages.dump(4);
}

} // namespace preprocessor
