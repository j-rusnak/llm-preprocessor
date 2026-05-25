#include <gtest/gtest.h>
#include "llm_tokenizer.hpp"

#include <string>
#include <vector>

TEST(HeuristicLLMTokenizerTest, EmptyStringIsZeroTokens) {
    preprocessor::HeuristicLLMTokenizer tk;
    EXPECT_EQ(tk.count_tokens(""), 0u);
}

TEST(HeuristicLLMTokenizerTest, ShortStringYieldsAtLeastOne) {
    preprocessor::HeuristicLLMTokenizer tk;
    EXPECT_GE(tk.count_tokens("hi"), 1u);
}

TEST(HeuristicLLMTokenizerTest, MonotonicGrowth) {
    preprocessor::HeuristicLLMTokenizer tk;
    EXPECT_LT(tk.count_tokens("hello"),
              tk.count_tokens("hello world hello world hello world"));
}

TEST(HeuristicLLMTokenizerTest, BatchSumMatchesIndividual) {
    preprocessor::HeuristicLLMTokenizer tk;
    std::vector<std::string> v = {"alpha beta", "gamma delta epsilon", "zeta"};
    std::size_t sum = 0;
    for (const auto& s : v) sum += tk.count_tokens(s);
    // The single-string overload hides the batch overload; call through the
    // interface to reach it.
    preprocessor::ILLMTokenizer& itk = tk;
    EXPECT_EQ(itk.count_tokens(v), sum);
}

TEST(HeuristicLLMTokenizerTest, RejectsBadConfig) {
    EXPECT_THROW(preprocessor::HeuristicLLMTokenizer(0.0), std::invalid_argument);
    EXPECT_THROW(preprocessor::HeuristicLLMTokenizer(-1.0), std::invalid_argument);
}

TEST(HeuristicLLMTokenizerTest, NameStable) {
    preprocessor::HeuristicLLMTokenizer tk;
    EXPECT_EQ(tk.name(), "heuristic-v1");
}

TEST(ModelCalibratedLLMTokenizerTest, MapsKnownModelFamilies) {
    preprocessor::ModelCalibratedLLMTokenizer tk;
    EXPECT_EQ(tk.model_family("gpt-4o-mini"), "gpt-4o");
    EXPECT_EQ(tk.model_family("gpt-4.1"), "gpt-4.1");
    EXPECT_EQ(tk.model_family("claude-3-5-sonnet"), "claude");
    EXPECT_EQ(tk.model_family("unknown-local-model"), "default");
}

TEST(ModelCalibratedLLMTokenizerTest, UsesFamilySpecificEstimates) {
    preprocessor::ModelCalibratedLLMTokenizer tk;
    std::string text;
    for (int i = 0; i < 24; ++i) {
        text +=
            "This is a moderately long mixed code and prose prompt with symbols foo_bar(). ";
    }
    EXPECT_NE(tk.count_tokens_for_model("gpt-4o-mini", text),
              tk.count_tokens_for_model("claude-3-5-sonnet", text));
}
