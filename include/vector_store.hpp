#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hnswlib {
template <typename dist_t> class HierarchicalNSW;
template <typename MTYPE> class SpaceInterface;
} // namespace hnswlib

namespace preprocessor {

/// Single ANN search result: external id of the chunk + cosine distance
/// (lower = more similar; convert to similarity via `1 - distance`).
struct VectorHit {
    std::uint64_t id;
    float distance;
};

/// Persistent HNSW-backed approximate-nearest-neighbour store for code-chunk
/// embeddings. Thread-safe for concurrent search; `add()` / `remove()` /
/// `save()` are NOT thread-safe and should be serialized by the caller.
///
/// IDs are caller-supplied 64-bit integers (typically xxhash of chunk content
/// or a monotonic chunk index), so the store can be reconciled with an
/// external content-addressable cache without secondary lookup tables.
class VectorStore {
public:
    /// Create or open a store. If `index_path` exists it is loaded; otherwise
    /// an empty index is initialised with the given `dim`/`max_elements`.
    ///
    /// `metric` accepts "cosine" (default; embeddings should be L2-normalised
    /// by the embedding engine) or "l2".
    VectorStore(std::size_t dim,
                std::size_t max_elements,
                const std::string& index_path = std::string{},
                const std::string& metric = "cosine");

    ~VectorStore();

    VectorStore(const VectorStore&) = delete;
    VectorStore& operator=(const VectorStore&) = delete;
    VectorStore(VectorStore&&) noexcept;
    VectorStore& operator=(VectorStore&&) noexcept;

    /// Insert or replace the embedding for `id`. The vector must have length
    /// equal to the store's `dim`.
    void add(std::uint64_t id, const std::vector<float>& embedding);

    /// Mark a vector as deleted (HNSW supports lazy deletion).
    void remove(std::uint64_t id);

    /// Return the top-`k` nearest neighbours of `query`. Excludes vectors
    /// previously passed to `remove()`.
    std::vector<VectorHit> search(const std::vector<float>& query, std::size_t k) const;

    /// Persist the index to `path` (or to the path supplied at construction
    /// if `path` is empty).
    void save(const std::string& path = std::string{}) const;

    /// Current number of inserted (non-deleted) vectors.
    std::size_t size() const;

    /// Embedding dimensionality.
    std::size_t dim() const { return dim_; }

private:
    std::size_t dim_;
    std::size_t max_elements_;
    std::string index_path_;
    std::string metric_;
    std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

} // namespace preprocessor
