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
    config.similarity_threshold = j.value("similarity_threshold", 0.65f);
    config.history_limit = j.value("history_limit", 10);

    if (config.model_path.empty()) {
        throw std::invalid_argument("model_path must not be empty");
    }
    if (config.vocab_path.empty()) {
        throw std::invalid_argument("vocab_path must not be empty");
    }
    if (config.db_path.empty()) {
        throw std::invalid_argument("db_path must not be empty");
    }

    if (config.similarity_threshold < 0.0f || config.similarity_threshold > 1.0f) {
        throw std::invalid_argument("similarity_threshold must be between 0.0 and 1.0");
    }
    if (config.history_limit < 1) {
        throw std::invalid_argument("history_limit must be at least 1");
    }

    // Optional API parameters (for complete payload mode).
    if (j.contains("api_model") && j["api_model"].is_string()) {
        config.api_model = j["api_model"].get<std::string>();
    }
    if (j.contains("api_endpoint") && j["api_endpoint"].is_string()) {
        config.api_endpoint = j["api_endpoint"].get<std::string>();
    }
    if (j.contains("temperature") && j["temperature"].is_number()) {
        float t = j["temperature"].get<float>();
        if (t < 0.0f || t > 2.0f) {
            throw std::invalid_argument("temperature must be between 0.0 and 2.0");
        }
        config.temperature = t;
    }
    if (j.contains("max_tokens") && j["max_tokens"].is_number_integer()) {
        int mt = j["max_tokens"].get<int>();
        if (mt < 1) {
            throw std::invalid_argument("max_tokens must be at least 1");
        }
        config.max_tokens = mt;
    }

    if (j.contains("intents") && j["intents"].is_array()) {
        for (const auto& intent : j["intents"]) {
            if (!intent.contains("name")) {
                throw std::invalid_argument("Each intent must have a 'name' field");
            }

            std::string name = intent["name"].get<std::string>();
            std::vector<std::string> examples;

            if (intent.contains("examples") && intent["examples"].is_array()) {
                for (const auto& ex : intent["examples"]) {
                    examples.push_back(ex.get<std::string>());
                }
            } else if (intent.contains("example")) {
                examples.push_back(intent["example"].get<std::string>());
            } else {
                throw std::invalid_argument("Each intent must have 'example' or 'examples' field");
            }

            if (examples.empty()) {
                throw std::invalid_argument("Intent '" + name + "' must have at least one example");
            }

            config.intents.emplace_back(std::move(name), std::move(examples));
        }
    }

    return config;
}

} // namespace preprocessor
