#include "context_gatherer.hpp"
#include "embedding_engine.hpp"
#include "intent_router.hpp"
#include "memory_engine.hpp"
#include "prompt_compiler.hpp"
#include "tokenizer.hpp"

#include <iostream>
#include <memory>
#include <string>

int main() {
    try {
        // --- 1. Initialize all pipeline components ---
        preprocessor::MemoryEngine memory("history.db");
        preprocessor::PromptCompiler compiler("You are a helpful AI assistant.");

        auto tokenizer = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
        auto engine = std::make_shared<preprocessor::EmbeddingEngine>("models/model.onnx", tokenizer);
        preprocessor::IntentRouter router(0.75f, engine);

        // Register known intents that can be handled locally.
        router.add_intent("ACTION_DECREASE_VOLUME", "turn down the volume");
        router.add_intent("ACTION_OPEN_FILE", "open the document file");

        // --- 2. Receive user input ---
        std::string user_input = "Can you summarize this site? http://example.com";
        std::cout << "Input: \"" << user_input << "\"\n\n";

        // --- 3. Attempt semantic routing ---
        auto matched = router.route(user_input);
        if (matched) {
            // A local action was matched — the host app would execute it directly.
            std::cout << "Routed to local action: " << *matched << "\n";
            return 0;
        }

        // No local action matched — pass through to the LLM pipeline.
        std::cout << "No local action matched. Preparing LLM payload...\n\n";

        // --- 4. Gather external context ---
        // Mock URL extraction — a real implementation would parse
        // the input with a regex or URI library.
        std::string url = "http://example.com";
        std::cout << "Fetching context from: " << url << "\n";
        std::string retrieved_context = preprocessor::ContextGatherer::fetch_url(url);

        // --- 5. Retrieve conversation history ---
        // Seed some history so the demo has something to show.
        memory.add_message("user", "Hello, what can you do?");
        memory.add_message("assistant", "I can summarize web pages, open files, and more.");

        auto history = memory.get_recent_history(5);

        // --- 6. Compile the final JSON payload ---
        std::string payload = compiler.build_payload(user_input, retrieved_context, history);

        std::cout << "\n=== Final LLM Payload ===\n" << payload << "\n";

        // The host application would now send `payload` to its LLM.

        // --- 7. Record this exchange in memory ---
        memory.add_message("user", user_input);
        memory.add_message("assistant", "(LLM response would go here)");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
