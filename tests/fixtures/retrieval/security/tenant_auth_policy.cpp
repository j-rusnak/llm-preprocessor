#include <string>

struct TenantAuthRequest {
    std::string tenant_id;
    std::string preprocessor_authorization;
    std::string hmac_signature;
    std::string nonce;
};

bool reject_replay_nonce(const TenantAuthRequest& request, const std::string& last_nonce) {
    return request.nonce.empty() || request.nonce == last_nonce;
}
