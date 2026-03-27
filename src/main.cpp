#include "config_loader.hpp"
#include "context_gatherer.hpp"
#include "embedding_engine.hpp"
#include "intent_router.hpp"
#include "memory_engine.hpp"
#include "prompt_compiler.hpp"
#include "text_sanitizer.hpp"
#include "tokenizer.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char* argv[]) {
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

        // Semantic routing is optional — only enabled when model files exist.
        std::unique_ptr<preprocessor::IntentRouter> router;
        bool routing_enabled = false;

        if (std::filesystem::exists(config.model_path) &&
            std::filesystem::exists(config.vocab_path)) {
            auto tokenizer = std::make_shared<preprocessor::Tokenizer>(config.vocab_path);
            auto engine = std::make_shared<preprocessor::EmbeddingEngine>(config.model_path, tokenizer);
            router = std::make_unique<preprocessor::IntentRouter>(config.similarity_threshold, engine);

            for (const auto& [name, example] : config.intents) {
                router->add_intent(name, example);
                std::cout << "  Registered intent: " << name << "\n";
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
                    std::cout << "[ACTION] " << *matched << "\n\n";
                    memory.add_message("user", user_input);
                    memory.add_message("assistant", "Executed local action: " + *matched);
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
            std::string payload = compiler.build_payload(user_input, retrieved_context, history);

            std::cout << "\n=== LLM Payload ===\n" << payload << "\n\n";

            // Record this exchange.
            memory.add_message("user", user_input);
            memory.add_message("assistant", "(awaiting LLM response)");
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
