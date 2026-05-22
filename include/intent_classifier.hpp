#pragma once

#include <memory>
#include <string>

namespace preprocessor {

/// Buckets used by Phase 2's prompt optimiser to pick a template.
///
/// New buckets MUST be appended (the string mapping is stable wire data
/// that ends up in metrics and template files).
enum class PromptBucket {
    CodeEdit,     ///< "modify X", "rename Y", "refactor Z"
    CodeExplain,  ///< "what does X do", "why is Y written like this"
    CodeGenerate, ///< "write a function that...", "implement..."
    MetaQuery,    ///< project-level questions ("which files use foo", "how big")
    Freeform,     ///< default; anything else
};

/// Canonical string form, used as the key in template files and metrics.
std::string to_string(PromptBucket bucket);
/// Inverse of `to_string`. Returns `PromptBucket::Freeform` for unknown input.
PromptBucket bucket_from_string(const std::string& name);

/// Maps a user message to a `PromptBucket`. Implementations are expected to
/// be pure and side-effect free.
class IIntentClassifier {
public:
    virtual ~IIntentClassifier() = default;
    virtual PromptBucket classify(const std::string& user_message) const = 0;
    virtual std::string name() const = 0;
};

/// Phase 2 default: keyword + lightweight pattern heuristic. Deterministic,
/// dependency-free, and zero-allocation in the common case. Designed to be
/// replaced behind `IIntentClassifier` in a later phase if an ML version
/// proves worthwhile.
class HeuristicIntentClassifier : public IIntentClassifier {
public:
    PromptBucket classify(const std::string& user_message) const override;
    std::string name() const override { return "heuristic"; }
};

} // namespace preprocessor
