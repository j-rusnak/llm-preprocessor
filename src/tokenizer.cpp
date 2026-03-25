#include "tokenizer.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace preprocessor {

Tokenizer::Tokenizer(const std::string& vocab_path) {
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

std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int64_t> token_ids;
    token_ids.push_back(cls_token_id); // [CLS]

    // Whitespace-based tokenization stub.
    // TODO: Replace with a real WordPiece algorithm for production use.
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::istringstream stream(lower_text);
    std::string word;
    while (stream >> word) {
        auto it = vocab_.find(word);
        if (it != vocab_.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(unk_token_id); // [UNK]
        }
    }

    token_ids.push_back(sep_token_id); // [SEP]
    return token_ids;
}

} // namespace preprocessor
