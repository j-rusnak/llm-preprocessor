#include "prompt_optimizer.hpp"

#include "repo_index.hpp"  // RetrievedChunk + CodeChunk

#include <sstream>
#include <stdexcept>
#include <utility>

namespace preprocessor {

namespace {

std::string build_context_block(const std::vector<RetrievedChunk>& chunks,
                                std::size_t max_chars) {
    if (chunks.empty()) return {};
    std::ostringstream out;
    std::size_t used = 0;
    for (const auto& rc : chunks) {
        std::ostringstream entry;
        entry << "\n--- " << rc.chunk.file_path
              << " [L" << rc.chunk.start_line << "-L" << rc.chunk.end_line << "]";
        if (!rc.chunk.symbol.empty()) entry << " " << rc.chunk.symbol;
        entry << " ---\n" << rc.chunk.text << "\n";
        const std::string s = entry.str();
        if (max_chars > 0 && used + s.size() > max_chars) break;
        out << s;
        used += s.size();
    }
    return out.str();
}

std::string plain_context_message(const std::vector<RetrievedChunk>& chunks,
                                  std::size_t max_chars) {
    if (chunks.empty()) return {};
    std::ostringstream out;
    out << "Retrieved code context (most relevant first):\n";
    out << build_context_block(chunks, max_chars);
    return out.str();
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
        r.system_message = plain_context_message(chunks, config_.max_context_chars);
        return r;
    }

    PromptRenderInput in;
    in.user_message = user_message;
    in.bucket = to_string(r.bucket);
    in.context = build_context_block(chunks, config_.max_context_chars);
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
