#include <stdexcept>
#include <string>

namespace preprocessor {

struct ProxyRuntimeConfig {
    std::string bind_host = "127.0.0.1";
    bool unsafe_allow_remote_proxy = false;
    std::size_t max_request_bytes = 1048576;
    std::string local_proxy_token;
};

void validate_proxy_runtime_config(const ProxyRuntimeConfig& config) {
    const bool loopback =
        config.bind_host == "127.0.0.1" ||
        config.bind_host == "::1" ||
        config.bind_host == "localhost";
    if (!loopback && !config.unsafe_allow_remote_proxy) {
        throw std::runtime_error(
            "refuse non-loopback proxy bind without unsafe_allow_remote_proxy");
    }
    if (config.max_request_bytes == 0 || config.max_request_bytes > 16 * 1024 * 1024) {
        throw std::runtime_error("max_request_bytes must limit JSON body parsing");
    }
    if (!loopback && config.local_proxy_token.empty()) {
        throw std::runtime_error("remote proxy exposure requires local auth token");
    }
}

}  // namespace preprocessor
