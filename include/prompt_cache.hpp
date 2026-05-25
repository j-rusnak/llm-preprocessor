#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct sqlite3;

namespace preprocessor {

struct PromptCacheEntry {
    std::string key;
    std::string payload;
};

/// SQLite-backed result cache for upstream LLM responses.
///
/// Key derivation (`make_key`) is content-addressed:
///   xxhash64( model || '\0' || prompt || '\0' || sorted_chunk_ids_hex )
/// so identical prompts compiled against the same retrieved chunk set hit
/// the cache regardless of formatting noise. Chunk ids are sorted before
/// hashing so retrieval order does not invalidate hits.
///
/// Schema (one table):
///   cache(key TEXT PRIMARY KEY, payload TEXT NOT NULL,
///         created_at INTEGER NOT NULL)
///
/// `put` overwrites on conflict. `get` honours the optional TTL set at
/// construction (0 disables expiry).
class PromptCache {
public:
    explicit PromptCache(const std::string& db_path, std::uint64_t ttl_seconds = 0);
    ~PromptCache();

    PromptCache(const PromptCache&) = delete;
    PromptCache& operator=(const PromptCache&) = delete;
    PromptCache(PromptCache&& other) noexcept;
    PromptCache& operator=(PromptCache&& other) noexcept;

    /// Compute the cache key for a prompt + model + chunk set. Returned as a
    /// 16-char lowercase hex string.
    static std::string make_key(const std::string& model,
                                const std::string& prompt,
                                const std::vector<std::uint64_t>& chunk_ids);

    /// Retrieve a cached payload, honouring TTL. Returns nullopt on miss or
    /// expiry.
    std::optional<std::string> get(const std::string& key) const;

    /// Store/overwrite a cached payload.
    void put(const std::string& key, const std::string& payload);

    /// Remove a single entry.
    void erase(const std::string& key);

    /// Drop everything (used in tests).
    void clear();

    /// Current row count.
    std::size_t size() const;

    /// Snapshot cache entries, newest first. `limit == 0` means no limit.
    std::vector<PromptCacheEntry> snapshot(std::size_t limit = 0) const;

private:
    sqlite3* db_ = nullptr;
    std::uint64_t ttl_seconds_;
};

} // namespace preprocessor
