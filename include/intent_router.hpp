#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace preprocessor {

class EmbeddingEngine;

struct Intent {
    std::string name;
    std::vector<float> embedding;
};

class IntentRouter {
public:
    IntentRouter(float similarity_threshold, std::shared_ptr<EmbeddingEngine> engine);

    void add_intent(const std::string& name, const std::string& representative_text);

    std::optional<std::string> route(const std::string& user_input) const;

    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

private:

    float threshold_;
    std::shared_ptr<EmbeddingEngine> engine_;
    std::vector<Intent> intents_;
};

} // namespace preprocessor
