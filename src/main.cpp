#include "context_gatherer.hpp"
#include "embedding_engine.hpp"
#include "intent_router.hpp"
#include "tokenizer.hpp"

#include <iostream>
#include <memory>
#include <string>

int main() {
    try {
        // Load the tokenizer vocabulary and the ONNX embedding model.
        // Replace these paths with real files for production use.
        auto tokenizer = std::make_shared<preprocessor::Tokenizer>("vocab.txt");
        auto engine = std::make_shared<preprocessor::EmbeddingEngine>("model.onnx", tokenizer);

        preprocessor::IntentRouter router(0.75f, engine);

        // Register intents using representative phrases.
        router.add_intent("ACTION_DECREASE_VOLUME", "turn down the volume");
        router.add_intent("ACTION_OPEN_FILE", "open the document file");
        router.add_intent("ACTION_SUMMARIZE_URL", "summarize this web page article");

        std::string test_input = "Please summarize this article: http://example.com";
        std::cout << "Input: \"" << test_input << "\"\n";

        auto result = router.route(test_input);
        if (result) {
            std::cout << "Matched intent: " << *result << "\n";

            // If the matched intent involves a URL, gather external context.
            if (*result == "ACTION_SUMMARIZE_URL") {
                // Mock URL extraction — a real implementation would parse
                // the input with a regex or URI library.
                std::string url = "http://example.com";
                std::cout << "Fetching context from: " << url << "\n";

                std::string page_content = preprocessor::ContextGatherer::fetch_url(url);

                // Build an enriched prompt the host application would send to its LLM.
                std::string enriched_prompt =
                    "The user asked: \"" + test_input + "\"\n\n"
                    "--- BEGIN FETCHED CONTEXT ---\n" +
                    page_content +
                    "\n--- END FETCHED CONTEXT ---\n\n"
                    "Please provide a concise summary of the above content.";

                std::cout << "\n=== Enriched Prompt ===\n" << enriched_prompt << "\n";
            }
        } else {
            std::cout << "No intent matched (below threshold).\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
