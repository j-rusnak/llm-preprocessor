#include "intent_router.hpp"
#include "embedding_engine.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace preprocessor {

// Common English stop / filler words that dilute embedding similarity.
static const std::unordered_set<std::string>& stop_words() {
    static const std::unordered_set<std::string> sw = {
        "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "have", "has", "had", "having",
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "it", "its", "they", "them", "their", "that", "this",
        "these", "those", "who", "whom", "which", "what", "where", "when",
        "how", "why", "if", "then", "so", "but", "and", "or", "not", "no",
        "to", "of", "in", "on", "at", "by", "for", "with", "from",
        "about", "into", "over", "after", "before",
        "can", "could", "will", "would", "shall", "should", "may", "might",
        "must", "just", "also", "very", "really", "too", "quite",
        "hey", "hi", "hello", "yo", "oh", "ok", "okay", "please", "thanks",
        "yeah", "yep", "nah", "well", "like", "um", "uh",
        "bro", "dude", "man", "buddy", "mate", "guys"
    };
    return sw;
}

static std::vector<std::string> remove_stop_words(const std::vector<std::string>& words) {
    const auto& sw = stop_words();
    std::vector<std::string> filtered;
    filtered.reserve(words.size());
    for (const auto& w : words) {
        if (sw.find(w) == sw.end()) {
            filtered.push_back(w);
        }
    }
    return filtered;
}

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

void IntentRouter::remove_intent(const std::string& name) {
    intents_.erase(
        std::remove_if(intents_.begin(), intents_.end(),
            [&name](const Intent& i) { return i.name == name; }),
        intents_.end());
}

void IntentRouter::clear_intents() {
    intents_.clear();
}

void IntentRouter::on_action(ActionCallback callback) {
    action_callback_ = std::move(callback);
}

std::optional<RouteResult> IntentRouter::route(const std::string& user_input) const {
    if (intents_.empty()) {
        return std::nullopt;
    }

    // Cap on total ONNX inference calls in a single route() invocation.
    static constexpr int max_inferences = 15;
    int inference_count = 0;

    auto make_result = [&](const Intent* best, float score) -> std::optional<RouteResult> {
        if (best && score >= threshold_) {
            RouteResult result{best->name, score};
            if (action_callback_) {
                action_callback_(result, user_input);
            }
            return result;
        }
        return std::nullopt;
    };

    // --- Fast path: try the full input first. ---
    std::vector<float> input_embedding = engine_->generate_embedding(user_input);
    ++inference_count;

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
        return make_result(best_intent, best_score);
    }

    // --- Early exit: if the full input scores very low against all intents,
    //     subphrase matching is unlikely to help. Skip the expensive sliding
    //     window to avoid 8-10x latency for clearly non-matching inputs. ---
    static constexpr float early_exit_ceiling = 0.35f;
    if (best_score < early_exit_ceiling) {
        return std::nullopt;
    }

    // --- Sliding-window subphrase matching. ---
    std::vector<std::string> words;
    {
        std::istringstream iss(user_input);
        std::string w;
        while (iss >> w) {
            words.push_back(w);
        }
    }

    // Try the stop-word-filtered phrase first — often sufficient.
    auto content_words = remove_stop_words(words);
    if (!content_words.empty() && content_words.size() < words.size() &&
        inference_count < max_inferences) {
        std::string filtered_phrase;
        for (const auto& w : content_words) {
            if (!filtered_phrase.empty()) filtered_phrase += ' ';
            filtered_phrase += w;
        }

        auto emb = engine_->generate_embedding(filtered_phrase);
        ++inference_count;
        for (const auto& intent : intents_) {
            float score = cosine_similarity(emb, intent.embedding);
            if (score > best_score) {
                best_score = score;
                best_intent = &intent;
            }
        }
        if (best_score >= threshold_) {
            return make_result(best_intent, best_score);
        }
    }

    // For inputs of 1-2 words the full input already covered them.
    if (words.size() <= 2) {
        return std::nullopt;
    }

    // Try windows from largest to smallest.
    const size_t max_window = std::min<size_t>(5, words.size() - 1);

    for (size_t window_size = max_window; window_size >= 1 && inference_count < max_inferences; --window_size) {
        for (size_t start = 0; start + window_size <= words.size() && inference_count < max_inferences; ++start) {
            std::string phrase;
            for (size_t i = start; i < start + window_size; ++i) {
                if (!phrase.empty()) phrase += ' ';
                phrase += words[i];
            }

            auto emb = engine_->generate_embedding(phrase);
            ++inference_count;
            for (const auto& intent : intents_) {
                float score = cosine_similarity(emb, intent.embedding);
                if (score > best_score) {
                    best_score = score;
                    best_intent = &intent;
                }
            }

            if (best_score >= threshold_) {
                return make_result(best_intent, best_score);
            }
        }
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
