#include <gtest/gtest.h>
#include "tokenizer.hpp"

#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

// Writes a minimal WordPiece vocab file for testing.
class TokenizerTest : public ::testing::Test {
protected:
    std::string vocab_path_ = "test_vocab_tmp.txt";

    void SetUp() override {
        // A minimal BERT-style vocab with special tokens and a few WordPiece entries.
        // Line number = token ID  (0-indexed).
        std::ofstream f(vocab_path_);
        f << "[PAD]\n";      // 0
        f << "[UNK]\n";      // 1
        f << "[CLS]\n";      // 2
        f << "[SEP]\n";      // 3
        f << "[MASK]\n";     // 4
        f << "hello\n";      // 5
        f << "world\n";      // 6
        f << "un\n";         // 7
        f << "##known\n";    // 8
        f << "##ing\n";      // 9
        f << "test\n";       // 10
        f << ".\n";          // 11
        f << "the\n";        // 12
        f << "a\n";          // 13
        f << "embed\n";      // 14
        f << "##d\n";        // 15
        f << "##ding\n";     // 16
    }

    void TearDown() override {
        std::remove(vocab_path_.c_str());
    }
};

TEST_F(TokenizerTest, WrapsWithClsAndSep) {
    preprocessor::Tokenizer tok(vocab_path_);
    auto ids = tok.encode("hello");
    // Should start with [CLS]=2, end with [SEP]=3
    ASSERT_GE(ids.size(), 3u);
    EXPECT_EQ(ids.front(), 2); // [CLS]
    EXPECT_EQ(ids.back(), 3);  // [SEP]
}

TEST_F(TokenizerTest, EncodesKnownWord) {
    preprocessor::Tokenizer tok(vocab_path_);
    auto ids = tok.encode("hello");
    // [CLS]=2, "hello"=5, [SEP]=3
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], 2);
    EXPECT_EQ(ids[1], 5);
    EXPECT_EQ(ids[2], 3);
}

TEST_F(TokenizerTest, EncodesMultipleWords) {
    preprocessor::Tokenizer tok(vocab_path_);
    auto ids = tok.encode("hello world");
    // [CLS]=2, "hello"=5, "world"=6, [SEP]=3
    ASSERT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids[1], 5);
    EXPECT_EQ(ids[2], 6);
}

TEST_F(TokenizerTest, WordPieceSubwordSplit) {
    preprocessor::Tokenizer tok(vocab_path_);
    auto ids = tok.encode("unknown");
    // "unknown" -> "un"=7, "##known"=8
    // Full: [CLS]=2, 7, 8, [SEP]=3
    ASSERT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids[1], 7);  // "un"
    EXPECT_EQ(ids[2], 8);  // "##known"
}

TEST_F(TokenizerTest, UnknownWordProducesUnk) {
    preprocessor::Tokenizer tok(vocab_path_);
    auto ids = tok.encode("xyzzy");
    // "xyzzy" not in vocab, no subword match -> [UNK]=1
    // [CLS]=2, [UNK]=1, [SEP]=3
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[1], 1);
}

TEST_F(TokenizerTest, PunctuationIsSplitAndTokenized) {
    preprocessor::Tokenizer tok(vocab_path_);
    auto ids = tok.encode("hello.");
    // "hello" + "." -> [CLS]=2, "hello"=5, "."=11, [SEP]=3
    ASSERT_EQ(ids.size(), 4u);
    EXPECT_EQ(ids[1], 5);
    EXPECT_EQ(ids[2], 11);
}

TEST_F(TokenizerTest, LowercasesInput) {
    preprocessor::Tokenizer tok(vocab_path_);
    auto ids = tok.encode("HELLO");
    // "HELLO" lowercased to "hello" = 5
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[1], 5);
}

TEST_F(TokenizerTest, MaxWordLengthProducesUnk) {
    preprocessor::Tokenizer tok(vocab_path_);
    std::string long_word(250, 'a'); // > max_word_length (200)
    auto ids = tok.encode(long_word);
    // [CLS], [UNK], [SEP]
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[1], 1); // [UNK]
}

TEST_F(TokenizerTest, EmptyInputReturnsClsSepOnly) {
    preprocessor::Tokenizer tok(vocab_path_);
    auto ids = tok.encode("");
    // Only [CLS] and [SEP]
    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 2);
    EXPECT_EQ(ids[1], 3);
}

TEST_F(TokenizerTest, TruncatesAtMaxSequenceLength) {
    preprocessor::Tokenizer tok(vocab_path_);

    // Build input with many words to exceed 512 tokens.
    std::string long_input;
    for (int i = 0; i < 600; ++i) {
        long_input += "hello ";
    }

    auto ids = tok.encode(long_input);
    EXPECT_LE(ids.size(), preprocessor::Tokenizer::max_sequence_length);
    EXPECT_EQ(ids.front(), 2); // [CLS]
    EXPECT_EQ(ids.back(), 3);  // [SEP]
}

TEST_F(TokenizerTest, SpecialTokenIdsFromVocab) {
    // Our test vocab has [CLS]=2, [SEP]=3, [UNK]=1 (not the default 101/102/100).
    // Verify the tokenizer correctly looked them up.
    preprocessor::Tokenizer tok(vocab_path_);
    auto ids = tok.encode("xyzzy");
    EXPECT_EQ(ids.front(), 2); // [CLS] from vocab line 2
    EXPECT_EQ(ids[1], 1);      // [UNK] from vocab line 1
    EXPECT_EQ(ids.back(), 3);  // [SEP] from vocab line 3
}

TEST_F(TokenizerTest, FailsOnMissingVocabFile) {
    EXPECT_THROW(preprocessor::Tokenizer("nonexistent_vocab.txt"), std::runtime_error);
}
