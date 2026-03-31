#include <gtest/gtest.h>
#include "context_gatherer.hpp"

#include <string>
#include <vector>

TEST(UrlExtractionTest, ExtractsSingleHttpUrl) {
    auto urls = preprocessor::ContextGatherer::extract_urls("check out http://example.com please");
    ASSERT_EQ(urls.size(), 1u);
    EXPECT_EQ(urls[0], "http://example.com");
}

TEST(UrlExtractionTest, ExtractsSingleHttpsUrl) {
    auto urls = preprocessor::ContextGatherer::extract_urls("visit https://example.com/path");
    ASSERT_EQ(urls.size(), 1u);
    EXPECT_EQ(urls[0], "https://example.com/path");
}

TEST(UrlExtractionTest, ExtractsMultipleUrls) {
    auto urls = preprocessor::ContextGatherer::extract_urls(
        "see http://one.com and https://two.com/page for info");
    ASSERT_EQ(urls.size(), 2u);
    EXPECT_EQ(urls[0], "http://one.com");
    EXPECT_EQ(urls[1], "https://two.com/page");
}

TEST(UrlExtractionTest, NoUrlsReturnsEmpty) {
    auto urls = preprocessor::ContextGatherer::extract_urls("no links here");
    EXPECT_TRUE(urls.empty());
}

TEST(UrlExtractionTest, StripsTrailingPunctuation) {
    auto urls = preprocessor::ContextGatherer::extract_urls("Go to https://example.com/page.");
    ASSERT_EQ(urls.size(), 1u);
    EXPECT_EQ(urls[0], "https://example.com/page");
}

TEST(UrlExtractionTest, HandlesUrlWithQueryParams) {
    auto urls = preprocessor::ContextGatherer::extract_urls(
        "search https://example.com/search?q=hello&lang=en now");
    ASSERT_EQ(urls.size(), 1u);
    EXPECT_EQ(urls[0], "https://example.com/search?q=hello&lang=en");
}
