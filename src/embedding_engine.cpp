#include "embedding_engine.hpp"
#include "tokenizer.hpp"

#include <stdexcept>
#include <numeric>
#include <cmath>

namespace preprocessor {

EmbeddingEngine::EmbeddingEngine(const std::string& model_path,
                                 std::shared_ptr<Tokenizer> tokenizer)
    : env_(ORT_LOGGING_LEVEL_WARNING, "EmbeddingEngine"),
      session_options_(),
      tokenizer_(std::move(tokenizer)) {
    if (!tokenizer_) {
        throw std::invalid_argument("Tokenizer must not be null");
    }

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
#ifdef _WIN32
        std::wstring wide_path(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(env_, wide_path.c_str(), session_options_);
#else
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("Failed to load ONNX model: ") + e.what());
    }
}

std::vector<float> EmbeddingEngine::generate_embedding(const std::string& text) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Tokenize the input text into token IDs via the Tokenizer.
    std::vector<int64_t> input_ids = tokenizer_->encode(text);
    auto sequence_length = static_cast<int64_t>(input_ids.size());

    // Build the attention mask: 1 for every real token.
    std::vector<int64_t> attention_mask(static_cast<std::size_t>(sequence_length), 1);

    std::array<int64_t, 2> input_shape = {1, sequence_length};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, input_ids.data(), input_ids.size(),
        input_shape.data(), input_shape.size()));

    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, attention_mask.data(), attention_mask.size(),
        input_shape.data(), input_shape.size()));

    // Query model for input/output node names
    auto input_name_0 = session_->GetInputNameAllocated(0, allocator);
    auto input_name_1 = session_->GetInputNameAllocated(1, allocator);
    auto output_name_0 = session_->GetOutputNameAllocated(0, allocator);

    std::array<const char*, 2> input_names = {input_name_0.get(), input_name_1.get()};
    std::array<const char*, 1> output_names = {output_name_0.get()};

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX inference failed: ") + e.what());
    }

    // Extract the output embedding
    auto& output_tensor = output_tensors.front();
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::size_t total_elements = tensor_info.GetElementCount();

    const float* raw_output = output_tensor.GetTensorData<float>();
    std::vector<float> embedding(raw_output, raw_output + total_elements);

    // L2-normalize the embedding vector
    float magnitude = std::sqrt(
        std::inner_product(embedding.begin(), embedding.end(), embedding.begin(), 0.0f));
    if (magnitude > 0.0f) {
        for (auto& val : embedding) {
            val /= magnitude;
        }
    }

    return embedding;
}

} // namespace preprocessor
