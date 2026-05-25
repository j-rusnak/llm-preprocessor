#include "auth_middleware.hpp"

#include <chrono>
#include <string>

namespace preprocessor {

bool verify_local_proxy_request(const Request& request,
                                const AuthMiddleware& auth,
                                std::int64_t now_seconds) {
    const std::string local_token =
        request.header("X-Preprocessor-Authorization");
    const std::string bearer = request.header("Authorization");
    const std::string signature = request.header("X-Preprocessor-Signature");
    const std::string timestamp = request.header("X-Preprocessor-Timestamp");

    if (!local_token.empty()) {
        return auth.verify_bearer_token(local_token);
    }

    if (!signature.empty() && !timestamp.empty()) {
        return auth.verify_hmac_signature(signature,
                                          timestamp,
                                          request.body(),
                                          now_seconds);
    }

    // Authorization may belong to the upstream provider. Do not treat it as
    // local proxy auth unless the proxy config explicitly enables that legacy
    // compatibility mode.
    return auth.legacy_authorization_enabled() &&
           auth.verify_bearer_token(bearer);
}

bool reject_replay_window(std::int64_t timestamp,
                          std::int64_t now_seconds,
                          std::int64_t allowed_skew_seconds) {
    const auto delta = now_seconds > timestamp
        ? now_seconds - timestamp
        : timestamp - now_seconds;
    return delta > allowed_skew_seconds;
}

} // namespace preprocessor
