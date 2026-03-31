#pragma once

#include <string>
#include <vector>
#include <utility>

namespace preprocessor {

struct Config {
    std::string model_path;
    std::string vocab_path;
    std::string db_path;
    std::string system_prompt;
    float similarity_threshold;
    int history_limit;
    std::vector<std::pair<std::string, std::vector<std::string>>> intents;
};

class ConfigLoader {
public:
    static Config load(const std::string& filepath);
};

} // namespace preprocessor
