#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "intent_classifier.hpp"

namespace preprocessor {

/// One upstream-LLM tier definition. Used by `ModelRouter` to swap the
/// upstream URL + model name per request based on cost / capability needs.
struct ModelTier {
    std::string name;            // "cheap" / "medium" / "frontier" / etc.
    std::string upstream_url;    // OpenAI-compatible URL
    std::string model_name;      // value placed in JSON body "model" field
    std::string api_key;         // optional override (else proxy default is used)
    std::size_t max_context = 0; // 0 = no opinion; otherwise advisory cap
};

/// Single routing rule. The first matching rule wins.
struct ModelRoute {
    PromptBucket bucket = PromptBucket::Freeform;
    std::size_t min_request_chars = 0;  // 0 = no lower bound
    std::size_t max_request_chars = 0;  // 0 = no upper bound
    std::string tier;                   // name of the target tier
};

/// Phase 8 multi-tier upstream router. Pure data structure + lookup —
/// `OpenAIProxy` consults it after intent classification + token estimation
/// and rewrites the upstream URL / model field before forwarding.
class ModelRouter {
public:
    ModelRouter() = default;

    /// Register an upstream tier.
    void add_tier(ModelTier tier);

    /// Register a routing rule. Evaluated in insertion order.
    void add_route(ModelRoute route);

    /// Pick the tier for the given (bucket, request_chars). Returns `nullptr`
    /// when no rule matches; callers should fall back to their default.
    const ModelTier* route(PromptBucket bucket, std::size_t request_chars) const;

    /// Look up a tier by name. Returns `nullptr` when unknown.
    const ModelTier* tier(const std::string& name) const;

    /// Count of registered tiers (for telemetry / tests).
    std::size_t tier_count() const;

    /// Count of registered routes.
    std::size_t route_count() const;

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, ModelTier> tiers_;
    std::vector<ModelRoute> routes_;
};

}  // namespace preprocessor
