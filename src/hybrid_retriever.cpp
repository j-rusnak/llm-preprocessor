#include "hybrid_retriever.hpp"

#include "bm25_index.hpp"
#include "retrieval_query.hpp"
#include "vector_store.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace preprocessor {

HybridRetriever::HybridRetriever(const VectorStore& vectors,
                                 const BM25Index& keywords,
                                 std::size_t rrf_k)
    : vectors_(vectors), keywords_(keywords), rrf_k_(rrf_k) {
    if (rrf_k_ == 0) throw std::invalid_argument("rrf_k must be > 0");
}

std::vector<HybridHit>
HybridRetriever::search(const std::string& query_text,
                        const std::vector<float>& query_embedding,
                        std::size_t k,
                        std::size_t oversample) const {
    if (k == 0) return {};
    const std::size_t per_backend = std::max<std::size_t>(k, k * oversample);

    auto v_hits = vectors_.search(query_embedding, per_backend);
    const auto parsed_query = parse_retrieval_query(query_text);
    auto b_hits = keywords_.search(build_lexical_query_text(parsed_query),
                                   per_backend);
    const bool exact_query = parsed_query.has_path_signal() ||
                             parsed_query.has_symbol_signal();
    const bool language_query = !parsed_query.language_hints.empty();
    const float vector_weight = exact_query ? 0.25f : (language_query ? 0.75f : 1.0f);
    const float keyword_weight = exact_query ? 8.0f : (language_query ? 2.5f : 1.75f);

    struct Acc {
        float score = 0.0f;
        float vector_score = 0.0f;
        float keyword_score = 0.0f;
    };
    std::unordered_map<std::uint64_t, Acc> fused;
    fused.reserve(v_hits.size() + b_hits.size());

    for (std::size_t rank = 0; rank < v_hits.size(); ++rank) {
        auto& a = fused[v_hits[rank].id];
        a.score += vector_weight / static_cast<float>(rrf_k_ + rank);
        a.vector_score = 1.0f - v_hits[rank].distance; // assume cosine/IP
    }
    for (std::size_t rank = 0; rank < b_hits.size(); ++rank) {
        auto& a = fused[b_hits[rank].id];
        a.score += keyword_weight / static_cast<float>(rrf_k_ + rank);
        a.keyword_score = b_hits[rank].score;
    }
    if (fused.empty()) return {};

    std::vector<HybridHit> out;
    out.reserve(fused.size());
    for (auto& [id, a] : fused) {
        out.push_back({id, a.score, a.vector_score, a.keyword_score});
    }
    if (out.size() > k) {
        std::partial_sort(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(k),
                          out.end(),
                          [](const HybridHit& a, const HybridHit& b) {
                              if (a.score != b.score) return a.score > b.score;
                              if (a.keyword_score != b.keyword_score) {
                                  return a.keyword_score > b.keyword_score;
                              }
                              if (a.vector_score != b.vector_score) {
                                  return a.vector_score > b.vector_score;
                              }
                              return a.id < b.id;
                          });
        out.resize(k);
    } else {
        std::sort(out.begin(), out.end(),
                  [](const HybridHit& a, const HybridHit& b) {
                      if (a.score != b.score) return a.score > b.score;
                      if (a.keyword_score != b.keyword_score) {
                          return a.keyword_score > b.keyword_score;
                      }
                      if (a.vector_score != b.vector_score) {
                          return a.vector_score > b.vector_score;
                      }
                      return a.id < b.id;
                  });
    }
    return out;
}

} // namespace preprocessor
