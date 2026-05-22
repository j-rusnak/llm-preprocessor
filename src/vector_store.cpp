#include "vector_store.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include <hnswlib/hnswlib.h>

namespace preprocessor {

VectorStore::VectorStore(std::size_t dim,
                         std::size_t max_elements,
                         const std::string& index_path,
                         const std::string& metric)
    : dim_(dim),
      max_elements_(max_elements),
      index_path_(index_path),
      metric_(metric) {
    if (dim_ == 0) {
        throw std::invalid_argument("VectorStore: dim must be > 0");
    }
    if (max_elements_ == 0) {
        throw std::invalid_argument("VectorStore: max_elements must be > 0");
    }

    if (metric_ == "cosine" || metric_ == "ip") {
        space_ = std::make_unique<hnswlib::InnerProductSpace>(dim_);
    } else if (metric_ == "l2") {
        space_ = std::make_unique<hnswlib::L2Space>(dim_);
    } else {
        throw std::invalid_argument("VectorStore: unknown metric '" + metric_ + "'");
    }

    const bool load_existing =
        !index_path_.empty() && std::filesystem::exists(index_path_);

    if (load_existing) {
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), index_path_, /*nmslib*/ false,
            max_elements_, /*allow_replace_deleted*/ true);
    } else {
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), max_elements_, /*M*/ 16, /*ef_construction*/ 200,
            /*seed*/ 100, /*allow_replace_deleted*/ true);
        index_->setEf(64);
    }
}

VectorStore::~VectorStore() = default;
VectorStore::VectorStore(VectorStore&&) noexcept = default;
VectorStore& VectorStore::operator=(VectorStore&&) noexcept = default;

void VectorStore::add(std::uint64_t id, const std::vector<float>& embedding) {
    if (embedding.size() != dim_) {
        throw std::invalid_argument("VectorStore::add: embedding dimensionality mismatch");
    }
    // addPoint replaces existing labels when allow_replace_deleted is set.
    index_->addPoint(embedding.data(), static_cast<hnswlib::labeltype>(id),
                     /*replace_deleted*/ true);
}

void VectorStore::remove(std::uint64_t id) {
    try {
        index_->markDelete(static_cast<hnswlib::labeltype>(id));
    } catch (const std::exception&) {
        // Idempotent: removing an unknown id is a no-op.
    }
}

std::vector<VectorHit> VectorStore::search(const std::vector<float>& query, std::size_t k) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("VectorStore::search: query dimensionality mismatch");
    }
    if (k == 0 || index_->cur_element_count == 0) {
        return {};
    }

    const std::size_t current = static_cast<std::size_t>(index_->cur_element_count);
    auto pq = index_->searchKnn(query.data(), std::min(k, current));

    std::vector<VectorHit> hits;
    hits.reserve(pq.size());
    while (!pq.empty()) {
        const auto& top = pq.top();
        hits.push_back({static_cast<std::uint64_t>(top.second), top.first});
        pq.pop();
    }
    std::reverse(hits.begin(), hits.end()); // nearest first
    return hits;
}

void VectorStore::save(const std::string& path) const {
    const std::string& target = path.empty() ? index_path_ : path;
    if (target.empty()) {
        throw std::invalid_argument("VectorStore::save: no path supplied");
    }
    index_->saveIndex(target);
}

std::size_t VectorStore::size() const {
    // cur_element_count includes lazily-deleted slots; subtract them.
    return static_cast<std::size_t>(index_->cur_element_count) - index_->getDeletedCount();
}

} // namespace preprocessor
