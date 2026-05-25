#include "prompt_optimizer.hpp"

#include "repo_index.hpp"  // RetrievedChunk + CodeChunk

#include <stdexcept>
#include <utility>

namespace preprocessor {

namespace {

PackedContext plain_context(const std::vector<RetrievedChunk>& chunks,
                            std::size_t max_chars) {
    ContextPackerConfig cfg;
    cfg.max_context_chars = max_chars;
    cfg.include_header = true;
    return pack_context(chunks, cfg);
}

} // namespace

PromptOptimizer::PromptOptimizer(PromptTemplates templates,
                                 std::shared_ptr<IIntentClassifier> classifier,
                                 PromptOptimizerConfig config)
    : templates_(std::move(templates)),
      classifier_(std::move(classifier)),
      config_(std::move(config)),
      enabled_(config_.enabled) {
    if (!classifier_) {
        throw std::invalid_argument("PromptOptimizer: classifier must not be null");
    }
}

void PromptOptimizer::set_project_card(ProjectCard card) {
    card_ = std::move(card);
    has_card_ = true;
}

void PromptOptimizer::set_enabled(bool enabled) noexcept {
    enabled_.store(enabled, std::memory_order_relaxed);
}

bool PromptOptimizer::enabled() const noexcept {
    return enabled_.load(std::memory_order_relaxed);
}

void PromptOptimizer::set_templates(PromptTemplates templates) {
    templates_ = std::move(templates);
}

PromptOptimizer::Result PromptOptimizer::optimise(
    const std::string& user_message,
    const std::vector<RetrievedChunk>& chunks) const {

    Result r;
    r.bucket = classifier_->classify(user_message);

    // Disabled path: behave exactly like Phase 1.
    if (!enabled()) {
        r.packed_context = plain_context(chunks, config_.max_context_chars);
        r.system_message = r.packed_context.text;
        return r;
    }

    PromptRenderInput in;
    in.user_message = user_message;
    in.bucket = to_string(r.bucket);
    ContextPackerConfig pack_cfg;
    pack_cfg.max_context_chars = config_.max_context_chars;
    pack_cfg.include_header = false;
    r.packed_context = pack_context(chunks, pack_cfg);
    in.context = r.packed_context.text;
    if (config_.include_project_card && has_card_) {
        in.project_card = card_.to_markdown();
    }

    // If we have nothing useful to add, don't inject a system message at all.
    if (in.context.empty() && in.project_card.empty()) {
        return r;
    }

    r.system_message = templates_.render(r.bucket, in);
    r.used_template = true;
    return r;
}

} // namespace preprocessor
