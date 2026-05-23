#pragma once

#include "repo_index.hpp"

#include <cstddef>
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
