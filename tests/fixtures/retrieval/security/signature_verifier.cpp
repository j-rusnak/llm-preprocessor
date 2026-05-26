#include <string>

bool check_hmac_digest(const std::string& body);

void verify_signature(const std::string& body) {
    if (!check_hmac_digest(body)) {
        reject_request_signature();
    }
}
