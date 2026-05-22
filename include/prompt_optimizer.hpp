#pragma once

#include "intent_classifier.hpp"
#include "project_card.hpp"
#include "prompt_templates.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace preprocessor {

struct RetrievedChunk;

/// Knobs for `PromptOptimizer`.
struct PromptOptimizerConfig {
    /// Master toggle. When false, `optimise()` produces the same plain
    /// "Retrieved code context (most relevant first)" block as Phase 1.
    bool enabled = true;
    /// Cap on the total character length of the chunk-derived context.
    /// 0 disables the cap.
    std::size_t max_context_chars = 8000;
    /// When true and a project card has been set, the card is rendered
    /// into the `{{project_card}}` slot.
    bool include_project_card = true;
};

/// Phase 2 prompt optimiser.
///
/// Pipeline: classify(user) -> select template -> render(card + context +
/// user) -> single rendered system message returned to the caller (typically
/// `OpenAIProxy`).
///
/// Thread-safety: `optimise()` is `const` and re-entrant. Mutators
/// (`set_project_card`, `set_enabled`, `set_templates`) are NOT thread-safe
/// against concurrent `optimise()` calls; perform them at startup or while
/// the proxy is paused.
class PromptOptimizer {
public:
    PromptOptimizer(PromptTemplates templates,
                    std::shared_ptr<IIntentClassifier> classifier,
                    PromptOptimizerConfig config = {});

    void set_project_card(ProjectCard card);
    void set_enabled(bool enabled) noexcept;
    bool enabled() const noexcept;

    /// Replace the template set. Useful for hot-reload from a file.
    void set_templates(PromptTemplates templates);

    /// One pass through the pipeline.
    struct Result {
        PromptBucket bucket = PromptBucket::Freeform;
        /// Final system message; empty when nothing should be injected.
        std::string system_message;
        /// True when a template was actually rendered (i.e. optimiser
        /// enabled and either chunks or project card were present).
        bool used_template = false;
    };

    Result optimise(const std::string& user_message,
                    const std::vector<RetrievedChunk>& chunks) const;

private:
    PromptTemplates templates_;
    std::shared_ptr<IIntentClassifier> classifier_;
    PromptOptimizerConfig config_;
    std::atomic<bool> enabled_;
    ProjectCard card_;
    bool has_card_ = false;
};

} // namespace preprocessor
