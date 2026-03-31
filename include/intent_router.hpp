#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <functional>

namespace preprocessor {

class EmbeddingEngine;

struct Intent {
    std::string name;
    std::vector<float> embedding;
};

struct RouteResult {
    std::string intent_name;
    float score;
};

namespace detail {
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
} // namespace detail

class IntentRouter {
public:
    using ActionCallback = std::function<void(const RouteResult&, const std::string& /* user_input */)>;

    IntentRouter(float similarity_threshold, std::shared_ptr<EmbeddingEngine> engine);

    void add_intent(const std::string& name, const std::string& representative_text);
    void remove_intent(const std::string& name);
    void clear_intents();

    void on_action(ActionCallback callback);

    std::optional<RouteResult> route(const std::string& user_input) const;

private:
    static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

    float threshold_;
    std::shared_ptr<EmbeddingEngine> engine_;
    std::vector<Intent> intents_;
    ActionCallback action_callback_;
};

} // namespace preprocessor
