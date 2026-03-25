#pragma once

#include <string>
#include <vector>
#include <memory>

#include <onnxruntime_cxx_api.h>

namespace preprocessor {

class Tokenizer;

class EmbeddingEngine {
public:
    EmbeddingEngine(const std::string& model_path, std::shared_ptr<Tokenizer> tokenizer);

    std::vector<float> generate_embedding(const std::string& text);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::shared_ptr<Tokenizer> tokenizer_;
};

} // namespace preprocessor
