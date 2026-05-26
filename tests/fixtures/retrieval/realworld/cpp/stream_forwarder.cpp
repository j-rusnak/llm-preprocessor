#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

struct ForwardEvent {
    std::string payload;
    bool terminal = false;
};

struct ForwardResult {
    int events_forwarded = 0;
    bool cancelled = false;
    std::string cancel_reason;
    std::string cache_hint;
};

class ClientConnection {
public:
    bool write(std::string_view frame) {
        if (closed_) {
            return false;
        }
        frames_.push_back(std::string(frame));
        return true;
    }

    void close() { closed_ = true; }
    bool closed() const { return closed_; }
    const std::vector<std::string>& frames() const { return frames_; }

private:
    bool closed_ = false;
    std::vector<std::string> frames_;
};

class UpstreamStream {
public:
    explicit UpstreamStream(std::vector<ForwardEvent> events) : events_(std::move(events)) {}

    std::optional<ForwardEvent> next() {
        if (cursor_ >= events_.size()) {
            return std::nullopt;
        }
        return events_[cursor_++];
    }

    void cancel() { cancelled_ = true; }
    bool cancelled() const { return cancelled_; }

private:
    std::vector<ForwardEvent> events_;
    std::size_t cursor_ = 0;
    bool cancelled_ = false;
};

struct StreamForwarder {
    std::atomic_bool client_disconnected{false};

    bool should_cancel_upstream(std::chrono::steady_clock::time_point last_event,
                                std::chrono::steady_clock::time_point now,
                                std::chrono::seconds idle_timeout) const {
        return client_disconnected.load() || now - last_event > idle_timeout;
    }

    ForwardResult forward_until_done(UpstreamStream& upstream,
                                     ClientConnection& client,
                                     std::chrono::steady_clock::time_point started,
                                     std::chrono::seconds idle_timeout,
                                     const std::function<std::chrono::steady_clock::time_point()>& clock) {
        ForwardResult result;
        auto last_event = started;
        while (true) {
            if (client.closed()) {
                client_disconnected.store(true);
            }
            if (should_cancel_upstream(last_event, clock(), idle_timeout)) {
                upstream.cancel();
                result.cancelled = true;
                result.cancel_reason = client_disconnected.load() ? "client disconnected" : "idle timeout";
                return result;
            }

            auto event = upstream.next();
            if (!event.has_value()) {
                return result;
            }
            last_event = clock();
            if (!client.write(format_sse_frame(event->payload))) {
                client_disconnected.store(true);
                continue;
            }
            ++result.events_forwarded;
            if (event->terminal) {
                return result;
            }
        }
    }

    static std::string format_sse_frame(std::string_view payload) {
        return "data: " + std::string(payload) + "\n\n";
    }
};

std::vector<ForwardEvent> make_forwarder_fixture_events() {
    return {{"{\"type\":\"delta\"}"}, {"{\"type\":\"usage\"}"}, {"[DONE]", true}};
}
