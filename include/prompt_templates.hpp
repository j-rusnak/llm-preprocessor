#pragma once

#include "intent_classifier.hpp"

#include <string>
#include <unordered_map>

namespace preprocessor {

/// Variables exposed to inja templates. Any field may be empty; templates
/// are expected to guard with `{% if %}` when omission matters.
struct PromptRenderInput {
    /// Markdown rendering of `ProjectCard` (empty when the optimiser is told
    /// not to inject one).
    std::string project_card;
    /// Concatenated retrieved code chunks (already trimmed to budget).
    std::string context;
    /// The raw user message (sanitised).
    std::string user_message;
    /// Bucket name as a string (e.g. "code_edit") - useful for diagnostics
    /// inside templates.
    std::string bucket;
};

/// Per-bucket inja-rendered system-prompt templates.
///
/// A `PromptTemplates` instance always has a fallback for every bucket: when
/// no override is supplied the built-in default is used. Loading from a JSON
/// file is opt-in.
class PromptTemplates {
public:
    /// Build with built-in defaults for every bucket.
    PromptTemplates();

    /// Load templates from a JSON file of the shape:
    ///   { "code_edit": "...inja src...", "freeform": "...", ... }
    /// Missing buckets keep their built-in defaults. Throws on parse error
    /// or unknown bucket names.
    static PromptTemplates load_from_file(const std::string& path);

    /// Override one template at runtime.
    void set_template(PromptBucket bucket, std::string source);

    /// Inja source currently registered for `bucket`.
    const std::string& source_for(PromptBucket bucket) const;

    /// Render the template for `bucket` with `input`. Throws
    /// `std::runtime_error` on template syntax errors.
    std::string render(PromptBucket bucket, const PromptRenderInput& input) const;

private:
    std::unordered_map<std::string, std::string> templates_;
};

} // namespace preprocessor
