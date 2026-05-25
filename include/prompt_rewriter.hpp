#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace preprocessor {

/// Pluggable post-processor that takes an already-assembled context /
/// system-message block and returns a (typically smaller) replacement.
///
/// Phase 5 contract: rewriters MUST be lossless w.r.t. user intent — they
/// may strip comments, collapse whitespace, dedupe near-duplicate spans, or
/// summarise large code blocks, but they must not invent facts, rename
/// identifiers, or reorder semantically meaningful content. The caller
/// (currently `OpenAIProxy`) always accounts the post-rewrite token count
/// in `ProxyMetrics::observe_tokens`.
///
/// Implementations:
///   - `HeuristicCompressionRewriter` - dependency-free, always available.
///   - `LlamaCppRewriter`             - small local model summariser; gated
///                                      behind the `LLM_PREPROCESSOR_WITH_LLAMA_CPP`
///                                      build flag.
class IPromptRewriter {
public:
    virtual ~IPromptRewriter() = default;

    /// Return a rewritten version of `context`. `max_chars` is a soft hint
    /// — if positive, the rewriter should try to land at or below it. A
    /// value of 0 means "no extra cap beyond your own internal policy".
    /// Implementations must be exception-safe; on internal failure they
    /// should return the input unchanged.
    virtual std::string rewrite(const std::string& context,
                                std::size_t max_chars) const = 0;

    /// Short human-readable identifier (used in metrics + logs).
    virtual std::string name() const = 0;
};

/// Tunables for `HeuristicCompressionRewriter`.
struct HeuristicCompressionConfig {
    bool strip_line_comments = true;     // //... and #... (outside strings)
    bool strip_block_comments = true;    // /* ... */
    bool collapse_blank_lines = true;    // >=2 consecutive blank lines -> 1
    bool trim_trailing_whitespace = true;
    bool dedupe_adjacent_lines = true;   // drop immediate dup lines
    /// If positive, hard-truncate the final output to this many chars
    /// (with a trailing "... [truncated]" marker). Overrides the `max_chars`
    /// argument when both are set; `max_chars` wins only when smaller.
    std::size_t hard_truncate_chars = 0;
};

/// Dependency-free heuristic compressor. Typical reduction on retrieved
/// C/C++/Python context: 15-40% character savings (and roughly the same in
/// tokens) with zero loss of structural information. Always safe to enable
/// in front of the upstream LLM.
class HeuristicCompressionRewriter : public IPromptRewriter {
public:
    explicit HeuristicCompressionRewriter(HeuristicCompressionConfig cfg = {});

    std::string rewrite(const std::string& context,
                        std::size_t max_chars) const override;
    std::string name() const override { return "heuristic"; }

private:
    HeuristicCompressionConfig cfg_;
};

/// Configuration for the optional local-LLM rewriter.
struct LlamaCppRewriterConfig {
    std::string model_path;          // path to a .gguf file
    std::size_t context_window = 4096;
    int n_threads = 4;
    float temperature = 0.0f;        // greedy by default for determinism
    /// System instruction prepended to the compression prompt.
    std::string system_prompt =
        "You compress retrieved code context for an LLM coding assistant. "
        "Preserve all identifier names, file paths, line numbers, and "
        "semantic facts. Remove comments, redundant blank lines, and "
        "near-duplicate spans. Output ONLY the compressed context.";
};

/// Local small-LLM rewriter backed by `llama.cpp`.
///
/// This implementation is gated behind the CMake option
/// `LLM_PREPROCESSOR_WITH_LLAMA_CPP` (off by default to keep the core build
/// dependency-free). When the option is off, the constructor throws
/// `std::runtime_error` so callers can detect the missing capability and
/// fall back to `HeuristicCompressionRewriter`.
class LlamaCppRewriter : public IPromptRewriter {
public:
    explicit LlamaCppRewriter(LlamaCppRewriterConfig cfg);
    ~LlamaCppRewriter();

    LlamaCppRewriter(const LlamaCppRewriter&) = delete;
    LlamaCppRewriter& operator=(const LlamaCppRewriter&) = delete;

    std::string rewrite(const std::string& context,
                        std::size_t max_chars) const override;
    std::string name() const override { return "llama-cpp"; }

    /// True when the build was configured with llama.cpp support. When
    /// false, the constructor will throw.
    static bool is_available() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace preprocessor
