#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace preprocessor {

class VectorStore;
class BM25Index;

struct HybridHit {
    std::uint64_t id;
    float score;          // fused RRF score (higher = better)
    float vector_score;   // 1 - cosine_distance, or 0 if not in vector hits
    float keyword_score;  // raw BM25 score, or 0 if not in keyword hits
};

/// Reciprocal-Rank-Fusion combiner for dense + sparse retrieval.
///
/// For each ranking r in {vector_hits, bm25_hits} a document at position p
/// (0-based) contributes `weight / (rrf_k + p)` to its total score. Exact
/// coding-agent signals (paths, identifiers, language hints) weight BM25 more
/// heavily so random dense-neighbour noise cannot outrank explicit file or
/// symbol searches. Final results are sorted by total score descending.
/// RRF is parameter-light and scale-invariant, which makes it useful when
/// fusing two scorers (cosine similarity, BM25) whose raw scores live on
/// incompatible scales.
///
/// `rrf_k` (default 60) is the standard constant from the original RRF
/// paper; smaller values aggravate top-rank dominance, larger values flatten
/// the contribution curve.
class HybridRetriever {
public:
    explicit HybridRetriever(const VectorStore& vectors,
                             const BM25Index& keywords,
                             std::size_t rrf_k = 60);

    /// Search for `k` results, drawing `oversample * k` candidates from each
    /// backend before fusing. `query_embedding` length must equal the
    /// VectorStore's `dim()`.
    std::vector<HybridHit> search(const std::string& query_text,
                                  const std::vector<float>& query_embedding,
                                  std::size_t k,
                                  std::size_t oversample = 4) const;

private:
    const VectorStore& vectors_;
    const BM25Index& keywords_;
    std::size_t rrf_k_;
};

} // namespace preprocessor
