#include "rate_limiter.hpp"

#include <gtest/gtest.h>

using preprocessor::RateLimiter;

TEST(RateLimiter, DisabledAllowsAll) {
    RateLimiter r;
    EXPECT_FALSE(r.enabled());
    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(r.try_acquire("k"));
    }
    EXPECT_EQ(r.rejected_count(), 0u);
}

TEST(RateLimiter, RejectsWhenBucketEmpty) {
    RateLimiter::Config cfg;
    cfg.tokens_per_second = 1.0;
    cfg.burst = 2.0;
    RateLimiter r(cfg);
    EXPECT_TRUE(r.enabled());
    double t = 1000.0;
    EXPECT_TRUE(r.try_acquire("k", t));
    EXPECT_TRUE(r.try_acquire("k", t));
    EXPECT_FALSE(r.try_acquire("k", t));
    EXPECT_EQ(r.rejected_count(), 1u);
}

TEST(RateLimiter, RefillsOverTime) {
    RateLimiter::Config cfg;
    cfg.tokens_per_second = 2.0;
    cfg.burst = 2.0;
    RateLimiter r(cfg);
    double t = 100.0;
    EXPECT_TRUE(r.try_acquire("k", t));
    EXPECT_TRUE(r.try_acquire("k", t));
    EXPECT_FALSE(r.try_acquire("k", t));
    // Advance one second -> 2 tokens refilled.
    EXPECT_TRUE(r.try_acquire("k", t + 1.0));
    EXPECT_TRUE(r.try_acquire("k", t + 1.0));
}

TEST(RateLimiter, BucketsAreKeyed) {
    RateLimiter::Config cfg;
    cfg.tokens_per_second = 1.0;
    cfg.burst = 1.0;
    RateLimiter r(cfg);
    double t = 10.0;
    EXPECT_TRUE(r.try_acquire("a", t));
    EXPECT_FALSE(r.try_acquire("a", t));
    EXPECT_TRUE(r.try_acquire("b", t));
}
