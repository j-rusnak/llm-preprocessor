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

    static constexpr std::size_t max_sequence_length = 512;

private:
    void load_vocab(const std::string& vocab_path);
    std::vector<std::string> basic_tokenize(const std::string& text) const;

    std::unordered_map<std::string, int64_t> vocab_;

    int64_t cls_token_id_ = 101;
    int64_t sep_token_id_ = 102;
    int64_t unk_token_id_ = 100;
    static constexpr std::size_t max_word_length = 200;
};

} // namespace preprocessor
