#include "rate_limiter.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace preprocessor {

namespace {

double real_now_seconds() {
    using clock = std::chrono::steady_clock;
    auto t = clock::now().time_since_epoch();
    return std::chrono::duration<double>(t).count();
}

}  // namespace

bool RateLimiter::try_acquire(const std::string& key, double now_seconds) {
    std::lock_guard<std::mutex> lock(mu_);
    if (cfg_.tokens_per_second <= 0.0 || cfg_.burst <= 0.0) return true;

    if (now_seconds <= 0.0) now_seconds = real_now_seconds();

    auto it = buckets_.find(key);
    if (it == buckets_.end()) {
        Bucket b;
        b.tokens = cfg_.burst;
        b.last_refill = now_seconds;
        it = buckets_.emplace(key, b).first;
    }
    auto& b = it->second;
    double elapsed = std::max(0.0, now_seconds - b.last_refill);
    b.tokens = std::min(cfg_.burst, b.tokens + elapsed * cfg_.tokens_per_second);
    b.last_refill = now_seconds;

    if (b.tokens >= 1.0) {
        b.tokens -= 1.0;
        return true;
    }
    ++rejected_;
    return false;
}

void RateLimiter::set_config(Config cfg) {
    std::lock_guard<std::mutex> lock(mu_);
    cfg_ = cfg;
    buckets_.clear();
}

bool RateLimiter::enabled() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return cfg_.tokens_per_second > 0.0 && cfg_.burst > 0.0;
}

std::uint64_t RateLimiter::rejected_count() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return rejected_;
}

}  // namespace preprocessor
