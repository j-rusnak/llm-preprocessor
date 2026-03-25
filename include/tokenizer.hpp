#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace preprocessor {

class Tokenizer {
public:
    explicit Tokenizer(const std::string& vocab_path);

    std::vector<int64_t> encode(const std::string& text) const;

private:
    void load_vocab(const std::string& vocab_path);
    std::vector<std::string> basic_tokenize(const std::string& text) const;

    std::unordered_map<std::string, int64_t> vocab_;

    static constexpr int64_t cls_token_id = 101;
    static constexpr int64_t sep_token_id = 102;
    static constexpr int64_t unk_token_id = 100;
    static constexpr std::size_t max_word_length = 200;
};

} // namespace preprocessor
