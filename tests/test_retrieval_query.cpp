#include "retrieval_query.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace {

bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::size_t count_value(const std::vector<std::string>& values,
                        const std::string& value) {
    return static_cast<std::size_t>(
        std::count(values.begin(), values.end(), value));
}

} // namespace

TEST(RetrievalQuery, SplitsIdentifiersIntoStableSearchTerms) {
    const auto query = preprocessor::parse_retrieval_query(
        "signatureVerify verify_signature getHTTPStatus");

    EXPECT_EQ(query.original_text,
              "signatureVerify verify_signature getHTTPStatus");
    EXPECT_TRUE(contains(query.terms, "signature"));
    EXPECT_TRUE(contains(query.terms, "verify"));
    EXPECT_TRUE(contains(query.terms, "http"));
    EXPECT_TRUE(contains(query.terms, "status"));
    EXPECT_TRUE(contains(query.identifier_terms, "signatureverify"));
    EXPECT_TRUE(contains(query.identifier_terms, "verify_signature"));
    EXPECT_TRUE(contains(query.identifier_terms, "gethttpstatus"));
    EXPECT_TRUE(query.has_symbol_signal());
}

TEST(RetrievalQuery, ExtractsPathAndLanguageHints) {
    const auto query = preprocessor::parse_retrieval_query(
        "fix src\\openai_proxy.cpp and tests/test_openai_proxy.cpp");

    EXPECT_TRUE(contains(query.path_hints, "src/openai_proxy.cpp"));
    EXPECT_TRUE(contains(query.path_hints, "tests/test_openai_proxy.cpp"));
    EXPECT_TRUE(contains(query.language_hints, "cpp"));
    EXPECT_TRUE(query.has_path_signal());
}

TEST(RetrievalQuery, MapsNaturalLanguageLanguageHints) {
    const auto query = preprocessor::parse_retrieval_query(
        "TypeScript AbortController for python jsonl ingestion and markdown docs");

    EXPECT_TRUE(contains(query.language_hints, "typescript"));
    EXPECT_TRUE(contains(query.language_hints, "python"));
    EXPECT_TRUE(contains(query.language_hints, "markdown"));
    EXPECT_TRUE(contains(query.language_hints, "json"));
    EXPECT_TRUE(contains(query.terms, "abort"));
    EXPECT_TRUE(contains(query.terms, "controller"));
}

TEST(RetrievalQuery, MapsInfraAndDataFileLanguageHints) {
    const auto query = preprocessor::parse_retrieval_query(
        "review deploy/kubernetes-deployment.yaml, db/schema.sql, Cargo.toml, "
        "rust cache store and go http server");

    EXPECT_TRUE(contains(query.path_hints, "deploy/kubernetes-deployment.yaml"));
    EXPECT_TRUE(contains(query.path_hints, "db/schema.sql"));
    EXPECT_TRUE(contains(query.path_hints, "cargo.toml"));
    EXPECT_TRUE(contains(query.language_hints, "yaml"));
    EXPECT_TRUE(contains(query.language_hints, "sql"));
    EXPECT_TRUE(contains(query.language_hints, "toml"));
    EXPECT_TRUE(contains(query.language_hints, "rust"));
    EXPECT_TRUE(contains(query.language_hints, "go"));
}

TEST(RetrievalQuery, InfersLanguageFromSourcePathExtensions) {
    const auto query = preprocessor::parse_retrieval_query(
        "open src/server.go src/cache.rs src/AuthFilter.java src/App.kt src/Worker.cs");

    EXPECT_TRUE(contains(query.language_hints, "go"));
    EXPECT_TRUE(contains(query.language_hints, "rust"));
    EXPECT_TRUE(contains(query.language_hints, "java"));
    EXPECT_TRUE(contains(query.language_hints, "kotlin"));
    EXPECT_TRUE(contains(query.language_hints, "csharp"));
}

TEST(RetrievalQuery, DedupesTermsWhilePreservingFirstOccurrence) {
    const auto query = preprocessor::parse_retrieval_query(
        "config CONFIG config_loader config-loader config");

    ASSERT_GE(query.terms.size(), 3u);
    EXPECT_EQ(query.terms[0], "config");
    EXPECT_EQ(query.terms[1], "loader");
    EXPECT_EQ(query.terms[2], "configloader");
    EXPECT_EQ(count_value(query.terms, "config"), 1u);
    EXPECT_EQ(count_value(query.terms, "loader"), 1u);
}

TEST(RetrievalQuery, DropsShortStopWordsButKeepsCodeAcronyms) {
    const auto query = preprocessor::parse_retrieval_query(
        "move db io ui to of in on go services");

    EXPECT_TRUE(contains(query.terms, "db"));
    EXPECT_TRUE(contains(query.terms, "io"));
    EXPECT_TRUE(contains(query.terms, "ui"));
    EXPECT_TRUE(contains(query.terms, "go"));
    EXPECT_TRUE(contains(query.language_hints, "go"));
    EXPECT_FALSE(contains(query.terms, "to"));
    EXPECT_FALSE(contains(query.terms, "of"));
    EXPECT_FALSE(contains(query.terms, "in"));
    EXPECT_FALSE(contains(query.terms, "on"));
}

TEST(RetrievalQuery, BuildsLexicalQueryTextFromAllRankingSignals) {
    const auto query = preprocessor::parse_retrieval_query(
        "fix src\\net\\getHTTPStatus.cpp in TypeScript getHTTPStatus");

    EXPECT_EQ(preprocessor::build_lexical_query_text(query),
              "fix src net get http status cpp typescript gethttpstatus "
              "src/net/gethttpstatus.cpp");
}
