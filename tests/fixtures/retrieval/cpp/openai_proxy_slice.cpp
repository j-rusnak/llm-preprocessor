#include "openai_proxy.hpp"
#include "proxy_metrics.hpp"

#include <curl/curl.h>
#include <httplib.h>

namespace preprocessor {

struct StreamState {
    httplib::DataSink* sink = nullptr;
    bool client_cancelled = false;
};

std::size_t stream_callback(char* data, std::size_t size, std::size_t count,
                            void* userdata) {
    auto* state = static_cast<StreamState*>(userdata);
    const std::size_t bytes = size * count;
    if (!state || !state->sink) return 0;
    if (!state->sink->write(data, bytes)) {
        state->client_cancelled = true;
        return 0;
    }
    return bytes;
}

void record_proxy_stats(ProxyMetrics& metrics,
                        bool auth_failed,
                        bool upstream_timeout,
                        bool stream_cancelled) {
    if (auth_failed) {
        metrics.on_error();
    }
    if (upstream_timeout) {
        metrics.on_error();
    }
    if (stream_cancelled) {
        metrics.on_stream_cancellation();
    }
}

} // namespace preprocessor
