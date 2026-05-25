#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace preprocessor {

struct BM25Hit {
    std::uint64_t id;
    float score;
};

/// In-memory BM25-Okapi keyword index over text chunks. Tokenisation is
/// case-folded, identifier-aware (alpha-num + `_`), with snake_case /
/// camelCase splitting so `getUserName` indexes as {get, user, name,
/// getusername}. No stopword removal.
///
/// Use alongside `VectorStore` for hybrid retrieval: BM25 catches exact
/// symbol / API name matches that dense embeddings often miss; the vector
/// store catches paraphrased / semantic matches. Fuse with `HybridRetriever`.
///
/// API:
///   - `add(id, text)` indexes one document.
///   - `remove(id)` drops it from the postings.
///   - `search(query, k)` returns top-k by BM25 score (higher = better).
///
/// Standard Okapi BM25 parameters (k1=1.5, b=0.75) are exposed via the
/// constructor for future tuning. Thread-safety: concurrent search is OK;
/// mutators must be externally serialised.
class BM25Index {
public:
    BM25Index(float k1 = 1.5f, float b = 0.75f);

    void add(std::uint64_t id, const std::string& text);
    void remove(std::uint64_t id);

    std::vector<BM25Hit> search(const std::string& query, std::size_t k) const;

    std::size_t size() const { return doc_lengths_.size(); }
    std::size_t vocab_size() const { return postings_.size(); }

private:
    struct Posting {
        std::uint64_t id;
        std::uint32_t tf;
    };

    float k1_;
    float b_;

    // term -> list of (doc id, term-frequency)
    std::unordered_map<std::string, std::vector<Posting>> postings_;
    // doc id -> token count
    std::unordered_map<std::uint64_t, std::uint32_t> doc_lengths_;
    // running sum for avgdl
    std::uint64_t total_tokens_ = 0;
};

} // namespace preprocessor
