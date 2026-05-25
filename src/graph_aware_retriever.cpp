#include "graph_aware_retriever.hpp"
#include "symbol_graph.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace preprocessor {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return s;
}

std::vector<std::string> query_terms(const std::string& query) {
    std::vector<std::string> out;
    std::string current;
    auto flush = [&]() {
        if (current.size() >= 3) out.push_back(std::move(current));
        current.clear();
    };

    unsigned char prev = 0;
    for (unsigned char c : query) {
        if (!std::isalnum(c)) {
            flush();
            prev = 0;
            continue;
        }

        if (!current.empty() &&
            std::isupper(c) &&
            (std::islower(prev) || std::isdigit(prev))) {
            flush();
        }
        current.push_back(static_cast<char>(std::tolower(c)));
        prev = c;
    }
    flush();
    return out;
}

float kind_priority(SymbolKind kind) {
    switch (kind) {
        case SymbolKind::Function:
        case SymbolKind::Method:
            return 3.0f;
        case SymbolKind::Class:
        case SymbolKind::Struct:
        case SymbolKind::Enum:
            return 2.0f;
        case SymbolKind::Variable:
        case SymbolKind::Macro:
            return 1.0f;
        case SymbolKind::Unknown:
        default:
            return 0.0f;
    }
}

float query_match_count(const std::vector<std::string>& terms,
                        const SymbolExpansionCandidate& candidate,
                        const CodeChunk& chunk) {
    if (terms.empty()) return 0.0f;
    const std::string haystack = lower(candidate.symbol + " " +
                                      chunk.symbol + " " +
                                      chunk.file_path + " " +
                                      chunk.text);
    float matches = 0.0f;
    for (const auto& term : terms) {
        if (haystack.find(term) != std::string::npos) {
            matches += 1.0f;
        }
    }
    return matches;
}

struct RankedExpansion {
    RetrievedChunk retrieved;
    std::string symbol;
    std::string file_path;
    float rank_score = 0.0f;
};

} // namespace

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

    const auto terms = query_terms(cfg.query_text);
    auto candidates = graph.expansion_candidates(seed_ids);
    const float expand_score = min_seed_score * cfg.score_decay;

    std::vector<RankedExpansion> ranked;
    ranked.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        const auto id = candidate.chunk_id;
        if (seen.count(id)) continue;
        CodeChunk c;
        if (!index.try_get_chunk(id, c)) continue;
        const float rank =
            cfg.query_match_weight * query_match_count(terms, candidate, c) +
            cfg.reference_count_weight *
                static_cast<float>(candidate.reference_count) +
            cfg.kind_weight * kind_priority(candidate.kind) -
            static_cast<float>(candidate.best_seed_rank) * 0.01f;

        float score = expand_score;
        if (min_seed_score > expand_score) {
            const float max_bonus = (min_seed_score - expand_score) * 0.5f;
            score += (std::min)(max_bonus, rank * 0.001f);
            score = (std::min)(score, std::nextafter(min_seed_score, expand_score));
        }

        RetrievedChunk rc;
        rc.chunk = std::move(c);
        rc.score = score;
        ranked.push_back(RankedExpansion{
            std::move(rc),
            candidate.symbol,
            {},
            rank
        });
        ranked.back().file_path = ranked.back().retrieved.chunk.file_path;
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const RankedExpansion& a, const RankedExpansion& b) {
                  if (a.rank_score != b.rank_score) {
                      return a.rank_score > b.rank_score;
                  }
                  if (a.file_path != b.file_path) return a.file_path < b.file_path;
                  if (a.symbol != b.symbol) return a.symbol < b.symbol;
                  return a.retrieved.chunk.id < b.retrieved.chunk.id;
              });

    std::unordered_map<std::string, std::size_t> per_symbol;
    std::unordered_map<std::string, std::size_t> per_file;
    for (auto& candidate : ranked) {
        if (out.size() >= seeds.size() + cfg.max_expanded) break;
        if (!seen.insert(candidate.retrieved.chunk.id).second) continue;
        if (cfg.max_per_symbol > 0 &&
            per_symbol[candidate.symbol] >= cfg.max_per_symbol) {
            continue;
        }
        if (cfg.max_per_file > 0 &&
            per_file[candidate.file_path] >= cfg.max_per_file) {
            continue;
        }
        ++per_symbol[candidate.symbol];
        ++per_file[candidate.file_path];
        out.push_back(std::move(candidate.retrieved));
    }

    return out;
}

} // namespace preprocessor
