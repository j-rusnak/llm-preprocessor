#pragma once

#include <string>
#include <vector>

namespace preprocessor {

/// Abstract interface for embedding generation.
/// Keeping this separate from the concrete EmbeddingEngine enables
/// dependency injection and lightweight mocking in unit tests.
class IEmbeddingEngine {
public:
    virtual ~IEmbeddingEngine() = default;

    virtual std::vector<float> generate_embedding(const std::string& text) = 0;

    /// Generate embeddings for many inputs in one model invocation.
    /// Default implementation loops over generate_embedding(); concrete
    /// engines should override for true batched inference.
    virtual std::vector<std::vector<float>>
    generate_embeddings(const std::vector<std::string>& texts) {
        std::vector<std::vector<float>> out;
        out.reserve(texts.size());
        for (const auto& t : texts) {
            out.push_back(generate_embedding(t));
        }
        return out;
    }
};

} // namespace preprocessor
