#include "auth_middleware.hpp"

#include <gtest/gtest.h>

using preprocessor::AuthMiddleware;

TEST(AuthMiddleware, DisabledByDefaultAllowsAll) {
    AuthMiddleware a;
    EXPECT_FALSE(a.enabled());
    EXPECT_TRUE(a.verify("", "", "", "body"));
}

TEST(AuthMiddleware, BearerAllowList) {
    AuthMiddleware::Config cfg;
    cfg.bearer_tokens = {"token-A", "token-B"};
    AuthMiddleware a(cfg);
    EXPECT_TRUE(a.enabled());
    EXPECT_TRUE(a.verify("Bearer token-A", "", "", "body"));
    EXPECT_TRUE(a.verify("Bearer token-B", "", "", "body"));
    EXPECT_FALSE(a.verify("Bearer token-C", "", "", "body"));
    EXPECT_FALSE(a.verify("", "", "", "body"));
}

TEST(AuthMiddleware, HmacRoundtrip) {
    AuthMiddleware::Config cfg;
    cfg.hmac_secret = "s3cret";
    AuthMiddleware a(cfg);
    std::int64_t ts = 1700000000;
    std::string body = R"({"hello":"world"})";
    auto sig = AuthMiddleware::sign("s3cret", ts, body);
    EXPECT_TRUE(a.verify("", sig, std::to_string(ts), body, ts));
}

TEST(AuthMiddleware, HmacRejectsBadSignature) {
    AuthMiddleware::Config cfg;
    cfg.hmac_secret = "s3cret";
    AuthMiddleware a(cfg);
    EXPECT_FALSE(a.verify("", "deadbeef", "1700000000", "body", 1700000000));
}

TEST(AuthMiddleware, HmacRejectsStaleTimestamp) {
    AuthMiddleware::Config cfg;
    cfg.hmac_secret = "s3cret";
    cfg.max_clock_skew = std::chrono::seconds{60};
    AuthMiddleware a(cfg);
    auto sig = AuthMiddleware::sign("s3cret", 1700000000, "body");
    // now is 1 hour later; sig is stale.
    EXPECT_FALSE(a.verify("", sig, "1700000000", "body", 1700003600));
}

TEST(AuthMiddleware, SignProducesHex64Chars) {
    auto sig = AuthMiddleware::sign("k", 1, "body");
    EXPECT_EQ(sig.size(), 64u);
    for (char c : sig) {
        EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    }
}
