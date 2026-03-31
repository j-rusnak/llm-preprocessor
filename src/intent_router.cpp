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
        "to", "of", "in", "on", "at", "by", "for", "with", "from", "up",
        "out", "about", "into", "over", "after", "before",
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

std::optional<std::string> IntentRouter::route(const std::string& user_input) const {
    if (intents_.empty()) {
        return std::nullopt;
    }

    // --- Fast path: try the full input first. ---
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

    // --- Sliding-window subphrase matching. ---
    // When the full sentence doesn't match (noise words dilute the embedding),
    // first try the input with stop words removed, then extract sliding windows.
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
    if (!content_words.empty() && content_words.size() < words.size()) {
        std::string filtered_phrase;
        for (const auto& w : content_words) {
            if (!filtered_phrase.empty()) filtered_phrase += ' ';
            filtered_phrase += w;
        }

        auto emb = engine_->generate_embedding(filtered_phrase);
        for (const auto& intent : intents_) {
            float score = cosine_similarity(emb, intent.embedding);
            if (score > best_score) {
                best_score = score;
                best_intent = &intent;
            }
        }
        if (best_score >= threshold_) {
            return best_intent->name;
        }
    }

    // For inputs of 1-2 words the full input already covered them.
    if (words.size() <= 2) {
        return std::nullopt;
    }

    // Try windows from largest to smallest — larger subphrases carry more
    // context and produce fewer false positives, so prefer them first.
    const size_t max_window = std::min<size_t>(5, words.size() - 1);

    for (size_t window_size = max_window; window_size >= 1; --window_size) {
        for (size_t start = 0; start + window_size <= words.size(); ++start) {
            std::string phrase;
            for (size_t i = start; i < start + window_size; ++i) {
                if (!phrase.empty()) phrase += ' ';
                phrase += words[i];
            }

            auto emb = engine_->generate_embedding(phrase);
            for (const auto& intent : intents_) {
                float score = cosine_similarity(emb, intent.embedding);
                if (score > best_score) {
                    best_score = score;
                    best_intent = &intent;
                }
            }

            // Early exit once a subphrase crosses the threshold.
            if (best_score >= threshold_) {
                return best_intent->name;
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
