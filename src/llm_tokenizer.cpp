#include "llm_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace preprocessor {

HeuristicLLMTokenizer::HeuristicLLMTokenizer(double chars_per_token)
    : chars_per_token_(chars_per_token) {
    if (chars_per_token_ <= 0.0) {
        throw std::invalid_argument("chars_per_token must be > 0");
    }
}

std::size_t HeuristicLLMTokenizer::count_tokens(const std::string& text) const {
    if (text.empty()) {
        return 0;
    }

    // Two-signal estimate: weighted average of char-based and word-based counts.
    // BPE merges short common words to 1 token but splits long/rare tokens
    // into multiple — averaging the two signals tracks tiktoken better than
    // either alone.

    const double char_estimate =
        static_cast<double>(text.size()) / chars_per_token_;

    std::size_t word_count = 0;
    bool in_word = false;
    for (unsigned char c : text) {
        const bool is_ws = std::isspace(c) != 0;
        if (!is_ws && !in_word) {
            ++word_count;
            in_word = true;
        } else if (is_ws) {
            in_word = false;
        }
    }

    // Code/punctuation-heavy text inflates token count vs. words; bias word
    // signal up by 1.3x as a coarse correction.
    const double word_estimate = static_cast<double>(word_count) * 1.3;

    const double blended = 0.6 * char_estimate + 0.4 * word_estimate;
    return static_cast<std::size_t>(std::ceil(std::max(1.0, blended)));
}

} // namespace preprocessor
