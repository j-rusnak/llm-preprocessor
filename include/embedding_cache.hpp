#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace preprocessor {

/// Persistent, content-addressed cache of chunk embeddings. The cache key
/// is `xxhash64(model_id || '|' || chunk_content)` so the same content
/// embedded by the same model is reused across sibling repos and across
/// process restarts. Phase 7 of the roadmap.
///
/// Storage: a single SQLite table `embeddings(key INTEGER PRIMARY KEY,
/// model TEXT, dim INTEGER, vec BLOB)`. The blob is a packed array of
/// little-endian `float` values; we assume the host is little-endian
/// (Windows + Linux x86_64), matching the rest of the codebase.
class EmbeddingCache {
public:
    explicit EmbeddingCache(const std::string& db_path,
                            std::string model_id);
    ~EmbeddingCache();

    EmbeddingCache(const EmbeddingCache&) = delete;
    EmbeddingCache& operator=(const EmbeddingCache&) = delete;

    /// Hash content + model into a stable 64-bit key.
    std::uint64_t key_for(std::string_view content) const noexcept;

    /// Look up by content. Returns `std::nullopt` on miss.
    std::optional<std::vector<float>> get(std::string_view content) const;

    /// Look up by precomputed key.
    std::optional<std::vector<float>> get_by_key(std::uint64_t key) const;

    /// Insert or replace.
    void put(std::string_view content, const std::vector<float>& vec);
    void put_by_key(std::uint64_t key, const std::vector<float>& vec);

    /// Total number of cached vectors (for telemetry / tests).
    std::size_t size() const;

    /// Wipe everything (for tests).
    void clear();

    const std::string& model_id() const noexcept { return model_id_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string model_id_;
};

}  // namespace preprocessor
