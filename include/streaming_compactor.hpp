#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace preprocessor {

/// One completed chat turn (assistant or user role).
struct ChatTurn {
    std::string role;     // "user" / "assistant" / "system"
    std::string content;
};

/// Output of a single compaction pass.
struct CompactionResult {
    /// Turns that were folded into `rolled_summary` and may be dropped.
    std::size_t rolled = 0;
    /// Rolling summary string suitable for re-injection as a system message.
    std::string rolled_summary;
    /// Turns kept verbatim because they fit within the budget.
    std::vector<ChatTurn> kept;
};

/// Phase 11 streaming-aware compactor. Folds older completed turns into a
/// rolling summary so long-running sessions stay under the model's
/// effective context window. The summary is a deterministic, dependency-
/// free heuristic: extract role-prefixed first-sentence excerpts. A future
/// upgrade can swap in `LlamaCppRewriter` behind the same surface.
class StreamingCompactor {
public:
    struct Config {
        /// Hard char cap applied across all kept turns + the rolled summary.
        std::size_t max_total_chars = 6000;
        /// Per-turn excerpt length used when building the rolling summary.
        std::size_t summary_chars_per_turn = 160;
        /// Always keep at least this many trailing turns verbatim.
        std::size_t keep_recent = 4;
    };

    explicit StreamingCompactor(Config cfg = {}) : cfg_(cfg) {}

    /// Compact the chat history.
    CompactionResult compact(const std::vector<ChatTurn>& history) const;

    const Config& config() const noexcept { return cfg_; }

private:
    Config cfg_;
};

}  // namespace preprocessor
