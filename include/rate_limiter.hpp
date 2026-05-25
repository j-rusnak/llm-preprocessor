#pragma once

#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

namespace preprocessor {

/// Phase 12 token-bucket rate limiter, keyed by an opaque caller string
/// (typically an API key or remote IP). Each key has its own bucket that
/// refills at `tokens_per_second` up to `burst` tokens. Disabled (always
/// allows) when both knobs are zero.
class RateLimiter {
public:
    struct Config {
        double tokens_per_second = 0.0;  // 0 = disabled
        double burst = 0.0;              // 0 = disabled
    };

    RateLimiter() = default;
    explicit RateLimiter(Config cfg) : cfg_(cfg) {}

    /// Try to consume one token. Returns `true` on success. `now_seconds`
    /// is injectable for tests; pass `0.0` to use the real clock.
    bool try_acquire(const std::string& key, double now_seconds = 0.0);

    /// Update config at runtime.
    void set_config(Config cfg);

    bool enabled() const noexcept;

    /// Telemetry: count of rejected requests since startup.
    std::uint64_t rejected_count() const noexcept;

private:
    struct Bucket {
        double tokens = 0.0;
        double last_refill = 0.0;
    };

    mutable std::mutex mu_;
    Config cfg_;
    std::unordered_map<std::string, Bucket> buckets_;
    std::uint64_t rejected_ = 0;
};

}  // namespace preprocessor
