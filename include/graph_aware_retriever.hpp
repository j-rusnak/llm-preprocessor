#pragma once

#include "repo_index.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace preprocessor {

class SymbolGraph;

/// Wraps a base retrieval result with one hop of graph expansion: for each
/// returned chunk, pull in the chunks that *define* identifiers the seed
/// chunks reference. The expanded chunks are appended after the seeds and
/// receive a discounted score so the original ranking is preserved.
///
/// This is a free function rather than a class because it is stateless and
/// composes naturally with `RepoIndex::search()`.
struct GraphExpansionConfig {
    std::size_t max_expanded = 4;
    /// Multiplier applied to the lowest seed score to derive the score of
    /// expanded chunks. Keeps them ranked below the originals.
    float score_decay = 0.5f;
    /// Optional original user query. When present, candidates whose symbol,
    /// file path, or chunk text matches query terms are promoted within the
    /// expanded section.
    std::string query_text;
    /// Maximum definitions kept for the same referenced symbol. 0 disables.
    std::size_t max_per_symbol = 1;
    /// Maximum expanded chunks kept per file. 0 disables.
    std::size_t max_per_file = 2;
    /// Ranking weights for graph candidates. These only affect the order of
    /// expanded chunks; seed chunks remain first.
    float query_match_weight = 4.0f;
    float reference_count_weight = 1.0f;
    float kind_weight = 0.25f;
};

/// Expand `seeds` using `graph`, hydrating any new chunk ids via `index`.
/// Returns a new vector that starts with `seeds` (unchanged order) followed
/// by up to `cfg.max_expanded` newly-introduced chunks.
std::vector<RetrievedChunk>
expand_with_graph(const std::vector<RetrievedChunk>& seeds,
                  const SymbolGraph& graph,
                  const RepoIndex& index,
                  GraphExpansionConfig cfg = {});

} // namespace preprocessor
