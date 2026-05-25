#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace preprocessor {

class PromptCache;
class VectorStore;

/// A single shareable cache entry pushed across the wire.
struct SyncCacheEntry {
    std::string key;
    std::string body;
};

/// A single vector pushed across the wire.
struct SyncVectorEntry {
    std::uint64_t chunk_id = 0;
    std::vector<float> vec;
    std::string source_path;
    std::string text;
    std::size_t start_line = 0;
    std::size_t end_line = 0;
    std::string symbol;
};

/// Aggregate bundle traded between team-mode peers.
struct SyncBundle {
    std::vector<SyncCacheEntry> cache;
    std::vector<SyncVectorEntry> vectors;
};

/// Phase 10 team-mode shim. Provides JSON serialization helpers + counters
/// for sharing `PromptCache` + `VectorStore` snapshots across a small team's
/// proxies. Cache HTTP routes are wired separately in `OpenAIProxy` so this
/// class stays transport-agnostic and easy to test.
///
/// Auth is intentionally out of scope here; `OpenAIProxy` applies the same
/// bearer/HMAC middleware used for the chat and stats routes.
class SyncEndpoint {
public:
    /// Merge a peer's bundle into the local `PromptCache`. Returns the
    /// number of entries actually written. `cache` may be `nullptr`.
    std::size_t apply_to_cache(const SyncBundle& bundle, PromptCache* cache);

    /// JSON helpers.
    std::string to_json(const SyncBundle& bundle) const;
    SyncBundle from_json(const std::string& json) const;

    /// Record a successful export (telemetry only).
    void note_export() noexcept;

    /// Record a successful import (telemetry only).
    void note_import() noexcept;

    /// Telemetry.
    std::size_t bundles_exported() const noexcept;
    std::size_t bundles_imported() const noexcept;

private:
    mutable std::mutex mu_;
    std::size_t exports_ = 0;
    std::size_t imports_ = 0;
};

}  // namespace preprocessor
