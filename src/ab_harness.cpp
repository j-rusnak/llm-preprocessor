#include "ab_harness.hpp"

#include <stdexcept>
#include <string>

#include <xxhash.h>

namespace preprocessor {

namespace {

std::string composite_key(std::string_view exp, std::string_view variant) {
    std::string k;
    k.reserve(exp.size() + 2 + variant.size());
    k.append(exp.data(), exp.size());
    k.append("::");
    k.append(variant.data(), variant.size());
    return k;
}

}  // namespace

void AbHarness::define(AbExperiment experiment) {
    if (experiment.variants.empty()) {
        throw std::invalid_argument("AbHarness::define: experiment '" +
                                    experiment.id + "' has no variants");
    }
    double sum = 0.0;
    for (const auto& v : experiment.variants) {
        if (v.weight < 0.0) {
            throw std::invalid_argument("AbHarness::define: negative weight in '" +
                                        experiment.id + "'");
        }
        sum += v.weight;
    }
    if (sum <= 0.0) {
        throw std::invalid_argument("AbHarness::define: all-zero weights in '" +
                                    experiment.id + "'");
    }
    std::lock_guard<std::mutex> lock(mu_);
    experiments_[experiment.id] = std::move(experiment);
}

std::string AbHarness::assign(std::string_view experiment_id,
                              std::string_view sticky_key) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = experiments_.find(std::string(experiment_id));
    if (it == experiments_.end() || it->second.variants.empty()) return {};
    const auto& exp = it->second;

    XXH64_state_t* st = XXH64_createState();
    XXH64_reset(st, 0);
    XXH64_update(st, experiment_id.data(), experiment_id.size());
    static const char sep = '\0';
    XXH64_update(st, &sep, 1);
    XXH64_update(st, sticky_key.data(), sticky_key.size());
    auto h = XXH64_digest(st);
    XXH64_freeState(st);

    double total = 0.0;
    for (const auto& v : exp.variants) total += v.weight;
    // Map [0, 2^64) -> [0, total). Use double to keep this header-free.
    double pick = (static_cast<double>(h) / 18446744073709551616.0) * total;
    double acc = 0.0;
    for (const auto& v : exp.variants) {
        acc += v.weight;
        if (pick < acc) return v.name;
    }
    return exp.variants.back().name;
}

std::uint64_t AbHarness::record_hit(std::string_view experiment_id,
                                    std::string_view variant) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& n = hits_[composite_key(experiment_id, variant)];
    return ++n;
}

std::unordered_map<std::string, std::uint64_t> AbHarness::hit_counts() const {
    std::lock_guard<std::mutex> lock(mu_);
    return hits_;
}

std::size_t AbHarness::experiment_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return experiments_.size();
}

}  // namespace preprocessor
