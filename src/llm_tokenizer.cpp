#include "llm_tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

namespace preprocessor {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return s;
}

bool starts_with(const std::string& s, const char* prefix) {
    const std::string p(prefix);
    return s.rfind(p, 0) == 0;
}

double chars_per_token_for_family(const std::string& family) {
    if (family == "gpt-4o") return 3.75;
    if (family == "gpt-4.1") return 3.85;
    if (family == "gpt-5") return 3.85;
    if (family == "openai-reasoning") return 3.8;
    if (family == "claude") return 3.55;
    if (family == "gemini") return 4.05;
    return 4.0;
}

std::size_t count_with_chars_per_token(const std::string& text,
                                       double chars_per_token) {
    if (text.empty()) {
        return 0;
    }

    const double char_estimate =
        static_cast<double>(text.size()) / chars_per_token;

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

    const double word_estimate = static_cast<double>(word_count) * 1.3;
    const double blended = 0.6 * char_estimate + 0.4 * word_estimate;
    return static_cast<std::size_t>(std::ceil((std::max)(1.0, blended)));
}

} // namespace

std::size_t ILLMTokenizer::count_tokens_for_model(
    const std::string&,
    const std::string& text) const {
    return count_tokens(text);
}

std::string ILLMTokenizer::model_family(const std::string& model) const {
    return model.empty() ? std::string{"default"} : model;
}

HeuristicLLMTokenizer::HeuristicLLMTokenizer(double chars_per_token)
    : chars_per_token_(chars_per_token) {
    if (chars_per_token_ <= 0.0) {
        throw std::invalid_argument("chars_per_token must be > 0");
    }
}

std::size_t HeuristicLLMTokenizer::count_tokens(const std::string& text) const {
    return count_with_chars_per_token(text, chars_per_token_);
}

std::size_t ModelCalibratedLLMTokenizer::count_tokens(
    const std::string& text) const {
    return count_tokens_for_model("", text);
}

std::size_t ModelCalibratedLLMTokenizer::count_tokens_for_model(
    const std::string& model,
    const std::string& text) const {
    return count_with_chars_per_token(
        text, chars_per_token_for_family(model_family(model)));
}

std::string ModelCalibratedLLMTokenizer::model_family(
    const std::string& model) const {
    const std::string m = lower(model);
    if (starts_with(m, "gpt-4o")) return "gpt-4o";
    if (starts_with(m, "gpt-4.1")) return "gpt-4.1";
    if (starts_with(m, "gpt-5")) return "gpt-5";
    if (starts_with(m, "o1") || starts_with(m, "o3") ||
        starts_with(m, "o4")) {
        return "openai-reasoning";
    }
    if (m.find("claude") != std::string::npos) return "claude";
    if (m.find("gemini") != std::string::npos) return "gemini";
    return "default";
}

} // namespace preprocessor
