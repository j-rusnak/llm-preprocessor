#include <string>

struct LoginRequest {
    std::string body;
    std::string session_cookie;
};

void verify_signature(const std::string& body);

void handle_login_controller(const LoginRequest& request) {
    parse_session_cookie(request.session_cookie);
    verify_signature(request.body);
    write_login_response();
}
