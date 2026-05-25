#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace preprocessor {

/// One variant in an A/B experiment. Variants are normally template names
/// (resolved by `PromptTemplates`) but can be any opaque string the caller
/// understands.
struct AbVariant {
    std::string name;     // identifier returned to the caller
    double weight = 1.0;  // unnormalised; weights are summed per experiment
};

/// Definition of a single experiment. Phase 9 of the roadmap. Bucketing is
/// deterministic per `(experiment_id, sticky_key)` so the same client sees
/// the same variant for the lifetime of the rollout — important so cache
/// keys stay stable across turns inside one chat.
struct AbExperiment {
    std::string id;                  // human-readable label
    std::vector<AbVariant> variants; // at least one
};

/// Deterministic, hash-based A/B router. Uses xxhash64 over
/// `(experiment_id || '\0' || sticky_key)` so two independent
/// experiments do not interfere.
class AbHarness {
public:
    AbHarness() = default;

    /// Register or replace an experiment. Throws on empty `variants`.
    void define(AbExperiment experiment);

    /// Pick a variant. Returns the experiment's first variant (when
    /// defined) or an empty string when `experiment_id` is unknown.
    std::string assign(std::string_view experiment_id,
                       std::string_view sticky_key) const;

    /// Increment a hit counter for telemetry. Returns the new value.
    std::uint64_t record_hit(std::string_view experiment_id,
                             std::string_view variant);

    /// Snapshot all per-(experiment, variant) hit counts for telemetry.
    std::unordered_map<std::string, std::uint64_t> hit_counts() const;

    /// Number of registered experiments.
    std::size_t experiment_count() const;

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, AbExperiment> experiments_;
    std::unordered_map<std::string, std::uint64_t> hits_;  // "exp::variant" -> n
};

}  // namespace preprocessor
