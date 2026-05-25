#pragma once

#include <string>
#include <vector>

namespace preprocessor {

/// Normalized signals extracted from a user retrieval query.
///
/// This keeps query parsing shared across lexical search, graph expansion, and
/// future context-packing work so each stage ranks from the same interpretation
/// of identifiers, paths, and language hints.
struct RetrievalQuery {
    std::string original_text;
    std::string normalized_text;
    std::vector<std::string> terms;
    std::vector<std::string> identifier_terms;
    std::vector<std::string> path_hints;
    std::vector<std::string> language_hints;

    bool has_symbol_signal() const noexcept {
        return !identifier_terms.empty();
    }

    bool has_path_signal() const noexcept {
        return !path_hints.empty();
    }
};

RetrievalQuery parse_retrieval_query(const std::string& text);

/// Build an order-preserving, deduped text query suitable for lexical
/// retrieval from a parsed query's terms, identifiers, paths, and languages.
std::string build_lexical_query_text(const RetrievalQuery& query);

} // namespace preprocessor
