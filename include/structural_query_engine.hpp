#pragma once

#include <optional>
#include <string>

namespace preprocessor {

class SymbolGraph;
class RepoIndex;

/// Zero-LLM fast path for purely structural questions.
///
/// Tries to answer simple "where is X defined?" / "what calls Y?" /
/// "what files reference Z?" / "list functions in F" style questions from
/// the `SymbolGraph` and `RepoIndex` alone. When it succeeds, the proxy
/// short-circuits the upstream call and serves a synthetic completion.
///
/// Returning `std::nullopt` means "I cannot answer this; please continue
/// the normal pipeline."
class StructuralQueryEngine {
public:
    StructuralQueryEngine(const SymbolGraph& graph, const RepoIndex& index);

    /// Attempt to answer `user_message`. Returns the answer text on success.
    std::optional<std::string> try_answer(const std::string& user_message) const;

private:
    const SymbolGraph& graph_;
    const RepoIndex& index_;
};

} // namespace preprocessor
