#include "config_loader.hpp"
#include "context_gatherer.hpp"
#include "embedding_engine.hpp"
#include "intent_router.hpp"
#include "memory_engine.hpp"
#include "prompt_compiler.hpp"
#include "text_sanitizer.hpp"
#include "tokenizer.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <curl/curl.h>

#ifndef PREPROCESSOR_VERSION
#define PREPROCESSOR_VERSION "unknown"
#endif
static constexpr const char* VERSION = PREPROCESSOR_VERSION;

static void print_help() {
    std::cout << "Usage: preprocessor_app [OPTIONS] [config_path]\n\n"
              << "Options:\n"
              << "  --help      Show this help message and exit\n"
              << "  --version   Show version information and exit\n\n"
              << "Arguments:\n"
              << "  config_path  Path to JSON config file (default: config.json)\n";
}

int main(int argc, char* argv[]) {
    // Handle --help / --version before anything else.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_help();
            return 0;
        }
        if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-v") == 0) {
            std::cout << "LLM Preprocessor v" << VERSION << "\n";
            return 0;
        }
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);

    int exit_code = 0;
    try {
        // --- 1. Load configuration ---
        std::string config_path = "config.json";
        if (argc > 1) {
            config_path = argv[1];
        }

        preprocessor::Config config = preprocessor::ConfigLoader::load(config_path);

        // --- 2. Initialize all pipeline components ---
        preprocessor::MemoryEngine memory(config.db_path);
        preprocessor::PromptCompiler compiler(config.system_prompt);

        // Build ApiParams from config for the complete-payload mode.
        preprocessor::ApiParams api_params;
        api_params.model = config.api_model;
        api_params.temperature = config.temperature;
        api_params.max_tokens = config.max_tokens;
        bool use_api_payload = api_params.model.has_value();

        // Semantic routing is optional — only enabled when model files exist.
        std::unique_ptr<preprocessor::IntentRouter> router;
        bool routing_enabled = false;

        if (std::filesystem::exists(config.model_path) &&
            std::filesystem::exists(config.vocab_path)) {
            auto tokenizer = std::make_shared<preprocessor::Tokenizer>(config.vocab_path);
            auto engine = std::make_shared<preprocessor::EmbeddingEngine>(config.model_path, tokenizer);
            router = std::make_unique<preprocessor::IntentRouter>(config.similarity_threshold, engine);

            for (const auto& [name, examples] : config.intents) {
                for (const auto& example : examples) {
                    router->add_intent(name, example);
                }
                std::cout << "  Registered intent: " << name
                          << " (" << examples.size() << " examples)\n";
            }
            routing_enabled = true;
        } else {
            std::cout << "[INFO] Model files not found — semantic routing disabled.\n";
            std::cout << "       model_path: " << config.model_path << "\n";
            std::cout << "       vocab_path: " << config.vocab_path << "\n";
        }

        std::cout << "\nLLM Preprocessor ready. Type your input (or 'quit' to exit).\n\n";

        // --- 3. Interactive loop ---
        std::string line;
        while (true) {
            std::cout << "> ";
            if (!std::getline(std::cin, line)) {
                break; // EOF
            }

            // Sanitize input.
            std::string user_input = preprocessor::TextSanitizer::sanitize(line);
            if (user_input.empty()) {
                continue;
            }
            if (user_input == "quit" || user_input == "exit") {
                break;
            }

            // --- 4. Attempt semantic routing (if available) ---
            if (routing_enabled) {
                auto matched = router->route(user_input);
                if (matched) {
                    std::cout << "[ACTION] " << matched->intent_name
                              << " (score: " << matched->score << ")\n\n";
                    memory.add_message("user", user_input);
                    memory.add_message("assistant", "Executed local action: " + matched->intent_name);
                    continue;
                }
            }

            // --- 5. Gather external context from URLs found in input ---
            std::string retrieved_context;
            auto urls = preprocessor::ContextGatherer::extract_urls(line);
            for (const auto& url : urls) {
                try {
                    std::cout << "[FETCH] " << url << "\n";
                    retrieved_context += preprocessor::ContextGatherer::fetch_url(url);
                } catch (const std::exception& e) {
                    std::cerr << "[WARN] Failed to fetch " << url << ": " << e.what() << "\n";
                }
            }

            // --- 6. Retrieve conversation history and compile payload ---
            auto history = memory.get_recent_history(config.history_limit);

            if (use_api_payload) {
                // Build display version (no history) for terminal output.
                std::vector<std::pair<std::string, std::string>> empty_history;
                auto display = compiler.build_payload_json(user_input, retrieved_context, empty_history, api_params);
                std::cout << "\n=== LLM Payload ===\n" << display.dump(4) << "\n";
            } else {
                std::vector<std::pair<std::string, std::string>> empty_history;
                std::string display_payload = compiler.build_payload(user_input, retrieved_context, empty_history);
                std::cout << "\n=== LLM Payload ===\n" << display_payload << "\n";
            }
            if (!history.empty()) {
                std::cout << "(+ " << history.size() << " history messages included in payload)\n";
            }
            std::cout << "\n";

            // Record this exchange (user message only — update when LLM responds).
            memory.add_message("user", user_input);

            // Auto-prune history to prevent unbounded growth.
            memory.prune(config.history_limit * 2);
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        exit_code = 1;
    }

    curl_global_cleanup();
    return exit_code;
}
