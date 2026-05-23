#include "model_router.hpp"

namespace preprocessor {

void ModelRouter::add_tier(ModelTier tier) {
    std::lock_guard<std::mutex> lock(mu_);
    tiers_[tier.name] = std::move(tier);
}

void ModelRouter::add_route(ModelRoute route) {
    std::lock_guard<std::mutex> lock(mu_);
    routes_.push_back(std::move(route));
}

const ModelTier* ModelRouter::route(PromptBucket bucket,
                                    std::size_t request_chars) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& r : routes_) {
        if (r.bucket != bucket) continue;
        if (r.min_request_chars > 0 && request_chars < r.min_request_chars) continue;
        if (r.max_request_chars > 0 && request_chars > r.max_request_chars) continue;
        auto it = tiers_.find(r.tier);
        if (it != tiers_.end()) return &it->second;
    }
    return nullptr;
}

const ModelTier* ModelRouter::tier(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = tiers_.find(name);
    return it == tiers_.end() ? nullptr : &it->second;
}

std::size_t ModelRouter::tier_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return tiers_.size();
}

std::size_t ModelRouter::route_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return routes_.size();
}

}  // namespace preprocessor
