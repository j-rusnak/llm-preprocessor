#include "embedding_engine.hpp"
#include "tokenizer.hpp"

#include <algorithm>
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
    auto results = generate_embeddings(std::vector<std::string>{text});
    return std::move(results.front());
}

std::vector<std::vector<float>>
EmbeddingEngine::generate_embeddings(const std::vector<std::string>& texts) {
    if (texts.empty()) {
        return {};
    }

    std::lock_guard<std::mutex> lock(mutex_);
    Ort::AllocatorWithDefaultOptions allocator;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // --- Tokenize each input and find the max sequence length for padding. ---
    const std::size_t batch = texts.size();
    std::vector<std::vector<int64_t>> per_input_ids;
    per_input_ids.reserve(batch);

    std::size_t max_seq = 0;
    for (const auto& t : texts) {
        per_input_ids.push_back(tokenizer_->encode(t));
        max_seq = std::max(max_seq, per_input_ids.back().size());
    }
    if (max_seq == 0) {
        max_seq = 1; // avoid zero-sized tensors
    }

    // --- Build padded [batch, max_seq] tensors. ---
    std::vector<int64_t> input_ids(batch * max_seq, 0);
    std::vector<int64_t> attention_mask(batch * max_seq, 0);
    std::vector<int64_t> token_type_ids(batch * max_seq, 0);

    for (std::size_t b = 0; b < batch; ++b) {
        const auto& row = per_input_ids[b];
        for (std::size_t t = 0; t < row.size(); ++t) {
            input_ids[b * max_seq + t] = row[t];
            attention_mask[b * max_seq + t] = 1;
        }
    }

    std::array<int64_t, 2> input_shape = {
        static_cast<int64_t>(batch), static_cast<int64_t>(max_seq)};

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

    // --- Resolve input/output names dynamically. ---
    const std::size_t num_inputs = session_->GetInputCount();
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

    // --- Pool + normalise each row independently. ---
    auto& output_tensor = output_tensors.front();
    auto tensor_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    const float* raw_output = output_tensor.GetTensorData<float>();

    std::vector<std::vector<float>> results(batch);

    if (shape.size() == 3) {
        // [batch, seq_len, hidden_dim] — attention-mask-aware mean pooling.
        const auto seq_len    = static_cast<std::size_t>(shape[1]);
        const auto hidden_dim = static_cast<std::size_t>(shape[2]);

        for (std::size_t b = 0; b < batch; ++b) {
            std::vector<float> embedding(hidden_dim, 0.0f);
            float mask_sum = 0.0f;
            for (std::size_t t = 0; t < seq_len; ++t) {
                const float m = static_cast<float>(attention_mask[b * max_seq + t]);
                if (m == 0.0f) continue;
                mask_sum += m;
                const float* row = raw_output + (b * seq_len + t) * hidden_dim;
                for (std::size_t d = 0; d < hidden_dim; ++d) {
                    embedding[d] += row[d] * m;
                }
            }
            if (mask_sum > 0.0f) {
                for (auto& v : embedding) v /= mask_sum;
            }
            const float mag = std::sqrt(
                std::inner_product(embedding.begin(), embedding.end(),
                                   embedding.begin(), 0.0f));
            if (mag > 0.0f) {
                for (auto& v : embedding) v /= mag;
            }
            results[b] = std::move(embedding);
        }
    } else if (shape.size() == 2) {
        // [batch, hidden_dim] — model already pooled.
        const auto hidden_dim = static_cast<std::size_t>(shape[1]);
        for (std::size_t b = 0; b < batch; ++b) {
            const float* row = raw_output + b * hidden_dim;
            std::vector<float> embedding(row, row + hidden_dim);
            const float mag = std::sqrt(
                std::inner_product(embedding.begin(), embedding.end(),
                                   embedding.begin(), 0.0f));
            if (mag > 0.0f) {
                for (auto& v : embedding) v /= mag;
            }
            results[b] = std::move(embedding);
        }
    } else {
        throw std::runtime_error(
            "EmbeddingEngine: unsupported output tensor rank " +
            std::to_string(shape.size()));
    }

    return results;
}

} // namespace preprocessor

