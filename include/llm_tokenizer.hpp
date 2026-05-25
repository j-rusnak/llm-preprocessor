#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace preprocessor {

/// Lightweight downstream-LLM tokenizer interface.
///
/// Phase 0 ships a heuristic implementation (`HeuristicLLMTokenizer`) that
/// estimates token counts from character/word statistics. A future phase will
/// add a real BPE/tiktoken-compatible backend. Code paths that need to budget
/// tokens (prompt assembly, cache keys, retrieval truncation) should depend on
/// this interface, NOT on the heuristic concretely.
class ILLMTokenizer {
public:
    virtual ~ILLMTokenizer() = default;

    /// Return the estimated token count for a single string.
    virtual std::size_t count_tokens(const std::string& text) const = 0;

    /// Return the estimated token count using a model-specific calibration.
    /// Implementations without model-specific data may fall back to
    /// `count_tokens(text)`.
    virtual std::size_t count_tokens_for_model(const std::string& model,
                                               const std::string& text) const;

    /// Return the estimated token count for many strings (sum).
    virtual std::size_t count_tokens(const std::vector<std::string>& texts) const {
        std::size_t total = 0;
        for (const auto& t : texts) {
            total += count_tokens(t);
        }
        return total;
    }

    /// Identifier for the tokenizer family (used in cache keys).
    virtual std::string name() const = 0;

    /// Coarse family bucket used for token telemetry.
    virtual std::string model_family(const std::string& model) const;
};

/// Heuristic tokenizer: approximates GPT/Claude-style BPE token counts using
/// a hybrid char-length + word-count model. Empirically within ~10-15% of
/// tiktoken `cl100k_base` for English / code on average — good enough for
/// budgeting / cache keying. Replace with a true BPE backend in a later phase.
class HeuristicLLMTokenizer : public ILLMTokenizer {
public:
    /// chars_per_token defaults to 4.0 (rough GPT-4 average for English).
    explicit HeuristicLLMTokenizer(double chars_per_token = 4.0);

    std::size_t count_tokens(const std::string& text) const override;
    std::string name() const override { return "heuristic-v1"; }

private:
    double chars_per_token_;
};

/// Model-calibrated estimator for OpenAI-compatible runtimes. This is still a
/// lightweight estimator, not exact BPE tokenization, but it uses model-family
/// calibration factors so token budgets and telemetry track routed models more
/// closely than one global heuristic.
class ModelCalibratedLLMTokenizer : public ILLMTokenizer {
public:
    ModelCalibratedLLMTokenizer() = default;

    std::size_t count_tokens(const std::string& text) const override;
    std::size_t count_tokens_for_model(const std::string& model,
                                       const std::string& text) const override;
    std::string name() const override { return "model-calibrated-v1"; }
    std::string model_family(const std::string& model) const override;
};

} // namespace preprocessor
