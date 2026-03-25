#include "intent_router.hpp"
#include "embedding_engine.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace preprocessor {

IntentRouter::IntentRouter(float similarity_threshold, std::shared_ptr<EmbeddingEngine> engine)
    : threshold_(similarity_threshold), engine_(std::move(engine)) {
    if (threshold_ < 0.0f || threshold_ > 1.0f) {
        throw std::invalid_argument("Similarity threshold must be between 0.0 and 1.0");
    }
    if (!engine_) {
        throw std::invalid_argument("EmbeddingEngine must not be null");
    }
}

void IntentRouter::add_intent(const std::string& name, const std::string& representative_text) {
    std::vector<float> embedding = engine_->generate_embedding(representative_text);
    if (embedding.empty()) {
        throw std::runtime_error("EmbeddingEngine returned an empty vector");
    }
    intents_.push_back({name, std::move(embedding)});
}

std::optional<std::string> IntentRouter::route(const std::string& user_input) const {
    if (intents_.empty()) {
        return std::nullopt;
    }

    std::vector<float> input_embedding = engine_->generate_embedding(user_input);

    float best_score = -1.0f;
    const Intent* best_intent = nullptr;

    for (const auto& intent : intents_) {
        float score = cosine_similarity(input_embedding, intent.embedding);
        if (score > best_score) {
            best_score = score;
            best_intent = &intent;
        }
    }

    if (best_intent && best_score >= threshold_) {
        return best_intent->name;
    }

    return std::nullopt;
}

float IntentRouter::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    return detail::cosine_similarity(a, b);
}

namespace detail {

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must have the same dimensionality");
    }
    if (a.empty()) {
        return 0.0f;
    }

    float dot = std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
    float mag_a = std::sqrt(std::inner_product(a.begin(), a.end(), a.begin(), 0.0f));
    float mag_b = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0f));

    if (mag_a == 0.0f || mag_b == 0.0f) {
        return 0.0f;
    }

    return dot / (mag_a * mag_b);
}

} // namespace detail

} // namespace preprocessor
