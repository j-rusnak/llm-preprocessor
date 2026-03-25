#include <gtest/gtest.h>
#include "text_sanitizer.hpp"

TEST(TextSanitizerTest, NormalizesWhitespaceAndCase) {
    std::string result = preprocessor::TextSanitizer::sanitize("  HELLO   World  ");
    EXPECT_EQ(result, "hello world");
}

TEST(TextSanitizerTest, HandlesEmptyString) {
    EXPECT_EQ(preprocessor::TextSanitizer::sanitize(""), "");
}

TEST(TextSanitizerTest, HandlesAllWhitespace) {
    EXPECT_EQ(preprocessor::TextSanitizer::sanitize("   \t\n  "), "");
}

TEST(TextSanitizerTest, PreservesSingleWord) {
    EXPECT_EQ(preprocessor::TextSanitizer::sanitize("Hello"), "hello");
}

TEST(TextSanitizerTest, CollapsesTabsAndNewlines) {
    EXPECT_EQ(preprocessor::TextSanitizer::sanitize("foo\t\tbar\nbaz"), "foo bar baz");
}
