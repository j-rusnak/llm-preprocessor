#include "intent_router.hpp"
#include "i_embedding_engine.hpp"

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
        "bro", "dude", "man", "buddy", "mate", "guys", "shit", "ts", "fucking",
        "goddamn", "damn"
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

IntentRouter::IntentRouter(float similarity_threshold, std::shared_ptr<IEmbeddingEngine> engine)
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

    auto best_against_intents = [&](const std::vector<float>& emb,
                                    float& best_score,
                                    const Intent*& best_intent) {
        for (const auto& intent : intents_) {
            const float score = cosine_similarity(emb, intent.embedding);
            if (score > best_score) {
                best_score = score;
                best_intent = &intent;
            }
        }
    };

    // --- Fast path: try the full input first via a single inference. ---
    std::vector<float> input_embedding = engine_->generate_embedding(user_input);

    float best_score = -1.0f;
    const Intent* best_intent = nullptr;
    best_against_intents(input_embedding, best_score, best_intent);

    if (best_intent && best_score >= threshold_) {
        return make_result(best_intent, best_score);
    }

    // Early exit when the full input is nowhere near any intent — subphrase
    // matching is unlikely to recover from this.
    static constexpr float early_exit_ceiling = 0.35f;
    if (best_score < early_exit_ceiling) {
        return std::nullopt;
    }

    // --- Build every candidate subphrase up front and embed them in ONE
    //     batched ONNX call. No per-call cap; ONNX Runtime handles batches
    //     efficiently. ---
    std::vector<std::string> words;
    {
        std::istringstream iss(user_input);
        std::string w;
        while (iss >> w) {
            words.push_back(w);
        }
    }

    std::vector<std::string> candidates;

    // Stop-word-filtered phrase (often the highest-signal candidate).
    auto content_words = remove_stop_words(words);
    if (!content_words.empty() && content_words.size() < words.size()) {
        std::string filtered_phrase;
        for (const auto& w : content_words) {
            if (!filtered_phrase.empty()) filtered_phrase += ' ';
            filtered_phrase += w;
        }
        candidates.push_back(std::move(filtered_phrase));
    }

    // Sliding windows of length 1..min(5, n-1). For very short inputs the
    // full-input pass already covered them.
    if (words.size() > 2) {
        const std::size_t max_window = std::min<std::size_t>(5, words.size() - 1);
        for (std::size_t window_size = max_window; window_size >= 1; --window_size) {
            for (std::size_t start = 0; start + window_size <= words.size(); ++start) {
                std::string phrase;
                for (std::size_t i = start; i < start + window_size; ++i) {
                    if (!phrase.empty()) phrase += ' ';
                    phrase += words[i];
                }
                candidates.push_back(std::move(phrase));
            }
        }
    }

    if (candidates.empty()) {
        return std::nullopt;
    }

    const auto sub_embeddings = engine_->generate_embeddings(candidates);
    for (const auto& emb : sub_embeddings) {
        best_against_intents(emb, best_score, best_intent);
        if (best_score >= threshold_) {
            return make_result(best_intent, best_score);
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
