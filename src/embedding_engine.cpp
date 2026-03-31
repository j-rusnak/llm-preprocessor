#include "embedding_engine.hpp"
#include "tokenizer.hpp"

#include <stdexcept>
#include <numeric>
#include <cmath>
#include <filesystem>

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

    // If the model file ends in .ort, tell ORT to load it as ORT format
    // (bypasses ONNX schema validation, which is broken in some vcpkg builds).
    if (model_path.size() >= 4 &&
        model_path.compare(model_path.size() - 4, 4, ".ort") == 0) {
        session_options_.AddConfigEntry("session.load_model_format", "ORT");
    }

    try {
#ifdef _WIN32
        std::filesystem::path fs_path(model_path);
        session_ = std::make_unique<Ort::Session>(env_, fs_path.wstring().c_str(), session_options_);
#else
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
#endif
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("Failed to load ONNX model: ") + e.what());
    }
}

std::vector<float> EmbeddingEngine::generate_embedding(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Tokenize the input text into token IDs via the Tokenizer.
    std::vector<int64_t> input_ids = tokenizer_->encode(text);
    auto sequence_length = static_cast<int64_t>(input_ids.size());

    // Build the attention mask: 1 for every real token.
    std::vector<int64_t> attention_mask(static_cast<std::size_t>(sequence_length), 1);

    // Token type IDs: all zeros for single-sequence input.
    std::vector<int64_t> token_type_ids(static_cast<std::size_t>(sequence_length), 0);

    std::array<int64_t, 2> input_shape = {1, sequence_length};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, input_ids.data(), input_ids.size(),
        input_shape.data(), input_shape.size()));

    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, attention_mask.data(), attention_mask.size(),
        input_shape.data(), input_shape.size()));

    input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, token_type_ids.data(), token_type_ids.size(),
        input_shape.data(), input_shape.size()));

    // Query model for input/output node names dynamically
    std::size_t num_inputs = session_->GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    std::vector<const char*> input_names;
    for (std::size_t i = 0; i < num_inputs; ++i) {
        input_name_ptrs.push_back(session_->GetInputNameAllocated(i, allocator));
        input_names.push_back(input_name_ptrs.back().get());
    }

    auto output_name_0 = session_->GetOutputNameAllocated(0, allocator);
    std::array<const char*, 1> output_names = {output_name_0.get()};

    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), num_inputs,
            output_names.data(), output_names.size());
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX inference failed: ") + e.what());
    }

    // Extract the output embedding
    // Output shape is typically [1, seq_len, hidden_dim]. Mean-pool over seq_len.
    auto& output_tensor = output_tensors.front();
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    const float* raw_output = output_tensor.GetTensorData<float>();

    std::vector<float> embedding;
    if (shape.size() == 3) {
        // [batch=1, seq_len, hidden_dim] — attention-mask-aware mean pooling
        auto seq_len = static_cast<std::size_t>(shape[1]);
        auto hidden_dim = static_cast<std::size_t>(shape[2]);
        embedding.resize(hidden_dim, 0.0f);
        float mask_sum = 0.0f;
        for (std::size_t t = 0; t < seq_len; ++t) {
            float mask_val = static_cast<float>(attention_mask[t]);
            mask_sum += mask_val;
            for (std::size_t d = 0; d < hidden_dim; ++d) {
                embedding[d] += raw_output[t * hidden_dim + d] * mask_val;
            }
        }
        if (mask_sum > 0.0f) {
            for (auto& val : embedding) {
                val /= mask_sum;
            }
        }
    } else {
        // [1, hidden_dim] or flat — use as-is
        std::size_t total_elements = tensor_info.GetElementCount();
        embedding.assign(raw_output, raw_output + total_elements);
    }

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
