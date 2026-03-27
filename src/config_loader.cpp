#include "config_loader.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace preprocessor {

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

    Config config;

    config.model_path = j.value("model_path", "models/model.onnx");
    config.vocab_path = j.value("vocab_path", "models/vocab.txt");
    config.db_path = j.value("db_path", "history.db");
    config.system_prompt = j.value("system_prompt", "You are a helpful AI assistant.");
    config.similarity_threshold = j.value("similarity_threshold", 0.75f);
    config.history_limit = j.value("history_limit", 10);

    if (config.similarity_threshold < 0.0f || config.similarity_threshold > 1.0f) {
        throw std::invalid_argument("similarity_threshold must be between 0.0 and 1.0");
    }
    if (config.history_limit < 1) {
        throw std::invalid_argument("history_limit must be at least 1");
    }

    if (j.contains("intents") && j["intents"].is_array()) {
        for (const auto& intent : j["intents"]) {
            if (!intent.contains("name") || !intent.contains("example")) {
                throw std::invalid_argument("Each intent must have 'name' and 'example' fields");
            }
            config.intents.emplace_back(
                intent["name"].get<std::string>(),
                intent["example"].get<std::string>());
        }
    }

    return config;
}

} // namespace preprocessor
