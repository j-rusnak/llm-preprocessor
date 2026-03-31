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
};

} // namespace preprocessor
