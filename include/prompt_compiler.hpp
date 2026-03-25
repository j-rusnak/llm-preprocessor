#pragma once

#include <string>
#include <vector>
#include <utility>

namespace preprocessor {

class PromptCompiler {
public:
    explicit PromptCompiler(const std::string& system_prompt);

    std::string build_payload(
        const std::string& user_input,
        const std::string& retrieved_context,
        const std::vector<std::pair<std::string, std::string>>& history) const;

private:
    std::string system_prompt_;
};

} // namespace preprocessor
