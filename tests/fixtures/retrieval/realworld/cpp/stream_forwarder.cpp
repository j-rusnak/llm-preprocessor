#include <atomic>
#include <chrono>

struct StreamForwarder {
    std::atomic_bool client_disconnected{false};

    bool should_cancel_upstream(std::chrono::steady_clock::time_point last_event,
                                std::chrono::steady_clock::time_point now,
                                std::chrono::seconds idle_timeout) const {
        return client_disconnected.load() || now - last_event > idle_timeout;
    }
};
