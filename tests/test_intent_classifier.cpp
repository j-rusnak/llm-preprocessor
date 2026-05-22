#include "intent_classifier.hpp"

#include <gtest/gtest.h>

using preprocessor::HeuristicIntentClassifier;
using preprocessor::PromptBucket;
using preprocessor::bucket_from_string;
using preprocessor::to_string;

TEST(IntentClassifier, EmptyIsFreeform) {
    HeuristicIntentClassifier c;
    EXPECT_EQ(c.classify(""), PromptBucket::Freeform);
}

TEST(IntentClassifier, EditVerbs) {
    HeuristicIntentClassifier c;
    EXPECT_EQ(c.classify("Refactor the OpenAIProxy class"), PromptBucket::CodeEdit);
    EXPECT_EQ(c.classify("rename foo to bar"), PromptBucket::CodeEdit);
    EXPECT_EQ(c.classify("Fix the off-by-one in build_context_block"), PromptBucket::CodeEdit);
}

TEST(IntentClassifier, GenerateVerbs) {
    HeuristicIntentClassifier c;
    EXPECT_EQ(c.classify("Write a function that parses JSON"), PromptBucket::CodeGenerate);
    EXPECT_EQ(c.classify("implement a hash table"), PromptBucket::CodeGenerate);
    EXPECT_EQ(c.classify("scaffold a new module"), PromptBucket::CodeGenerate);
}

TEST(IntentClassifier, ExplainPhrases) {
    HeuristicIntentClassifier c;
    EXPECT_EQ(c.classify("explain how BM25 ranking works"), PromptBucket::CodeExplain);
    EXPECT_EQ(c.classify("what does this lambda capture by reference"), PromptBucket::CodeExplain);
    EXPECT_EQ(c.classify("why is this loop unrolled"), PromptBucket::CodeExplain);
}

TEST(IntentClassifier, MetaQueriesTakePrecedence) {
    HeuristicIntentClassifier c;
    // Contains "fix" (an edit keyword) but is clearly a meta query.
    EXPECT_EQ(c.classify("which files would I need to fix?"), PromptBucket::MetaQuery);
    EXPECT_EQ(c.classify("project structure overview"), PromptBucket::MetaQuery);
}

TEST(IntentClassifier, UnknownDefaultsFreeform) {
    HeuristicIntentClassifier c;
    EXPECT_EQ(c.classify("Hello there"), PromptBucket::Freeform);
}

TEST(IntentClassifier, ToStringRoundTrip) {
    for (auto b : {PromptBucket::CodeEdit, PromptBucket::CodeExplain,
                   PromptBucket::CodeGenerate, PromptBucket::MetaQuery,
                   PromptBucket::Freeform}) {
        EXPECT_EQ(bucket_from_string(to_string(b)), b);
    }
    // Unknown -> Freeform.
    EXPECT_EQ(bucket_from_string("totally_invented"), PromptBucket::Freeform);
}

TEST(IntentClassifier, WholeWordBoundary) {
    HeuristicIntentClassifier c;
    // "addendum" contains "add" as a substring but not as a whole word.
    EXPECT_EQ(c.classify("the addendum is helpful"), PromptBucket::Freeform);
}
