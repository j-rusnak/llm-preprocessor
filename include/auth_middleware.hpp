#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace preprocessor {

/// Phase 12 authentication middleware. Supports two mutually-compatible
/// schemes:
///  - **Bearer tokens** matched against an allow-list (`accept_bearer`).
///  - **HMAC-SHA256** over `(timestamp || '.' || body)` with a shared
///    secret, presented as `X-Signature: hex(...)` plus `X-Timestamp: <unix>`.
///    The timestamp must be within `max_clock_skew_seconds` of `now`.
///
/// Both schemes are entirely optional; when no secrets / tokens are
/// configured the middleware allows every request through. This keeps the
/// proxy a single-binary, zero-config experience by default and lets
/// production deployments opt in.
class AuthMiddleware {
public:
    struct Config {
        std::unordered_set<std::string> bearer_tokens;
        std::string hmac_secret;
        std::chrono::seconds max_clock_skew{300};
    };

    AuthMiddleware() = default;
    explicit AuthMiddleware(Config cfg) : cfg_(std::move(cfg)) {}

    /// `true` when at least one auth mode is configured.
    bool enabled() const noexcept;

    /// Validate a request. `now_unix` is injectable for tests; pass `0` to
    /// use the real clock.
    bool verify(std::string_view authorization_header,
                std::string_view signature_header,
                std::string_view timestamp_header,
                std::string_view body,
                std::int64_t now_unix = 0) const;

    /// Reload config (e.g. after a SIGHUP). Thread-safe.
    void set_config(Config cfg);

    /// Compute the HMAC-SHA256 hex digest used by `verify`. Exposed for
    /// tests and for clients that need to sign requests symmetrically.
    static std::string sign(std::string_view secret,
                            std::int64_t timestamp,
                            std::string_view body);

private:
    mutable std::mutex mu_;
    Config cfg_;
};

}  // namespace preprocessor
