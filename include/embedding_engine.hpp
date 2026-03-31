#pragma once

#include <string>
#include <vector>
#include <memory>
#include <mutex>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

namespace preprocessor {

class Tokenizer;

/// Thread-safe embedding generator. generate_embedding() may be called from
/// multiple threads; an internal mutex serializes ONNX session access.
class EmbeddingEngine {
public:
    EmbeddingEngine(const std::string& model_path, std::shared_ptr<Tokenizer> tokenizer);

    std::vector<float> generate_embedding(const std::string& text);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::shared_ptr<Tokenizer> tokenizer_;
    mutable std::mutex mutex_;
};

} // namespace preprocessor
