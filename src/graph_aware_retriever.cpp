#include "graph_aware_retriever.hpp"
#include "symbol_graph.hpp"

#include <algorithm>
#include <unordered_set>

namespace preprocessor {

std::vector<RetrievedChunk>
expand_with_graph(const std::vector<RetrievedChunk>& seeds,
                  const SymbolGraph& graph,
                  const RepoIndex& index,
                  GraphExpansionConfig cfg) {
    std::vector<RetrievedChunk> out = seeds;
    if (seeds.empty() || cfg.max_expanded == 0) return out;

    std::vector<std::uint64_t> seed_ids;
    seed_ids.reserve(seeds.size());
    std::unordered_set<std::uint64_t> seen;
    float min_seed_score = seeds.front().score;
    for (const auto& s : seeds) {
        seed_ids.push_back(s.chunk.id);
        seen.insert(s.chunk.id);
        if (s.score < min_seed_score) min_seed_score = s.score;
    }

    auto neighbours = graph.neighbors_of(seed_ids, cfg.max_expanded);
    const float expand_score = min_seed_score * cfg.score_decay;
    for (auto id : neighbours) {
        if (!seen.insert(id).second) continue;
        CodeChunk c;
        if (!index.try_get_chunk(id, c)) continue;
        RetrievedChunk rc;
        rc.chunk = std::move(c);
        rc.score = expand_score;
        out.push_back(std::move(rc));
    }
    return out;
}

} // namespace preprocessor
