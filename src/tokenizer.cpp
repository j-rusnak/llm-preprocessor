#include "tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace preprocessor {

Tokenizer::Tokenizer(const std::string& vocab_path) {
    load_vocab(vocab_path);
}

void Tokenizer::load_vocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file: " + vocab_path);
    }

    std::string line;
    int64_t id = 0;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            vocab_[line] = id;
        }
        ++id;
    }
}

std::vector<std::string> Tokenizer::basic_tokenize(const std::string& text) const {
    // Lowercase the entire input.
    std::string lower;
    lower.reserve(text.size());
    for (unsigned char c : text) {
        lower.push_back(static_cast<char>(std::tolower(c)));
    }

    // Split punctuation from words and collapse whitespace.
    std::vector<std::string> tokens;
    std::string current;

    for (char c : lower) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
        } else if (std::ispunct(static_cast<unsigned char>(c))) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
            tokens.emplace_back(1, c);
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        tokens.push_back(std::move(current));
    }

    return tokens;
}

std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int64_t> token_ids;
    token_ids.push_back(cls_token_id); // [CLS]

    std::vector<std::string> words = basic_tokenize(text);

    for (const auto& word : words) {
        // If the word is excessively long, treat the whole thing as [UNK].
        if (word.size() > max_word_length) {
            token_ids.push_back(unk_token_id);
            continue;
        }

        // WordPiece: greedy longest-match-first from left to right.
        std::size_t start = 0;
        bool is_unknown = false;

        while (start < word.size()) {
            std::size_t end = word.size();
            bool found = false;

            while (start < end) {
                std::string substr = word.substr(start, end - start);

                // Subwords after the first character span get the "##" prefix.
                if (start > 0) {
                    substr = "##" + substr;
                }

                auto it = vocab_.find(substr);
                if (it != vocab_.end()) {
                    token_ids.push_back(it->second);
                    start = end;
                    found = true;
                    break;
                }

                --end;
            }

            if (!found) {
                // No subword match at all — the entire original word is [UNK].
                is_unknown = true;
                break;
            }
        }

        if (is_unknown) {
            token_ids.push_back(unk_token_id);
        }
    }

    token_ids.push_back(sep_token_id); // [SEP]
    return token_ids;
}

} // namespace preprocessor
