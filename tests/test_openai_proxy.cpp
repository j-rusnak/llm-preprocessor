#include "code_chunker.hpp"
#include "i_embedding_engine.hpp"
#include "llm_tokenizer.hpp"
#include "model_router.hpp"
#include "openai_proxy.hpp"
#include "prompt_cache.hpp"
#include "proxy_metrics.hpp"
#include "repo_index.hpp"
#include "structural_query_engine.hpp"
#include "symbol_graph.hpp"
#include "sync_endpoint.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

using nlohmann::json;

namespace {

class MockEmbedder : public preprocessor::IEmbeddingEngine {
public:
    explicit MockEmbedder(std::size_t dim) : dim_(dim) {}
    std::vector<float> generate_embedding(const std::string& text) override {
        std::vector<float> v(dim_, 0.0f);
        std::size_t h = std::hash<std::string>{}(text);
        for (std::size_t i = 0; i < dim_; ++i)
            v[i] = static_cast<float>(((h >> (i % 32)) & 0xFF) / 255.0);
        float n = 0.0f;
        for (float x : v) n += x * x;
        n = n > 0.0f ? std::sqrt(n) : 1.0f;
        for (auto& x : v) x /= n;
        return v;
    }
private:
    std::size_t dim_;
};

/// Start a fake OpenAI upstream that echoes the body it received under
/// "received". Returns (server, port, thread). Stop via server->stop() + join.
struct FakeUpstream {
    std::shared_ptr<httplib::Server> server;
    int port = 0;
    std::thread thr;
    std::atomic<int> calls{0};
    std::string last_body;
    std::string last_authorization;
    bool streaming_response = false;
    int streaming_chunk_count = 2;
    int streaming_chunk_delay_ms = 0;
    std::atomic<int> streaming_chunks_written{0};
    std::atomic<int> streaming_write_failures{0};
    int response_status = 200;
    std::string response_body;
    std::string response_content_type = "application/json";

    FakeUpstream() : server(std::make_shared<httplib::Server>()) {
        server->Post("/v1/chat/completions",
                     [this](const httplib::Request& req, httplib::Response& res) {
            calls.fetch_add(1);
            last_body = req.body;
            last_authorization = req.has_header("Authorization")
                ? req.get_header_value("Authorization")
                : std::string{};
            res.status = response_status;
            if (streaming_response) {
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [this](std::size_t, httplib::DataSink& sink) {
                        const std::string done = "data: [DONE]\n\n";
                        for (int i = 0; i < streaming_chunk_count; ++i) {
                            const std::string chunk =
                                "data: {\"choices\":[{\"delta\":{\"content\":\"" +
                                std::to_string(i) + "\"}}]}\n\n";
                            if (!sink.write(chunk.data(), chunk.size())) {
                                streaming_write_failures.fetch_add(1);
                                return false;
                            }
                            streaming_chunks_written.fetch_add(1);
                            if (streaming_chunk_delay_ms > 0) {
                                std::this_thread::sleep_for(
                                    std::chrono::milliseconds(streaming_chunk_delay_ms));
                            }
                        }
                        if (!sink.write(done.data(), done.size())) return false;
                        sink.done();
                        return true;
                    });
                return;
            }
            if (!response_body.empty()) {
                res.set_content(response_body, response_content_type);
                return;
            }
            json out = {
                {"id", "fake-1"},
                {"object", "chat.completion"},
                {"choices", json::array({
                    {{"index", 0},
                     {"message", {{"role", "assistant"}, {"content", "ok"}}},
                     {"finish_reason", "stop"}}
                })},
                {"echo_body_len", req.body.size()}
            };
            res.set_content(out.dump(), response_content_type);
        });
        port = server->bind_to_any_port("127.0.0.1");
        thr = std::thread([this] { server->listen_after_bind(); });
        // wait briefly for the server to actually be ready
        for (int i = 0; i < 50 && !server->is_running(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    ~FakeUpstream() {
        server->stop();
        if (thr.joinable()) thr.join();
    }
};

struct ProxyHarness {
    std::shared_ptr<MockEmbedder> embedder = std::make_shared<MockEmbedder>(16);
    std::shared_ptr<preprocessor::BraceAwareChunker> chunker =
        std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndexConfig idx_cfg = []() {
        preprocessor::RepoIndexConfig c;
        c.embedding_dim = 16;
        c.watch_for_changes = false;
        return c;
    }();
    preprocessor::RepoIndex index{embedder, chunker, idx_cfg};
    preprocessor::PromptCache cache{":memory:"};
    preprocessor::ProxyMetrics metrics;
    preprocessor::ModelCalibratedLLMTokenizer tokenizer;
    std::unique_ptr<preprocessor::OpenAIProxy> proxy;
    int port = 0;
    std::thread thr;

    void start(const std::string& upstream_url,
               preprocessor::OpenAIProxyConfig pcfg = {}) {
        pcfg.upstream_url = upstream_url;
        pcfg.retrieval_k = 4;
        proxy = std::make_unique<preprocessor::OpenAIProxy>(
            index, cache, metrics, tokenizer, pcfg);
        port = proxy->bind_to_port("127.0.0.1", 0);
        thr = std::thread([this] { proxy->listen_after_bind(); });
        for (int i = 0; i < 50; ++i) {
            httplib::Client c("127.0.0.1", port);
            auto r = c.Get("/healthz");
            if (r && r->status == 200) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    ~ProxyHarness() {
        if (proxy) proxy->stop();
        if (thr.joinable()) thr.join();
    }
};

} // namespace

TEST(OpenAIProxy, Healthz) {
    ProxyHarness h;
    FakeUpstream up;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions");
    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Get("/healthz");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 200);
}

TEST(OpenAIProxy, ForwardsAndCachesAndMeasures) {
    ProxyHarness h;
    FakeUpstream up;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions");

    json body = {
        {"model", "gpt-test"},
        {"messages", json::array({
            {{"role", "user"}, {"content", "explain the project"}}
        })}
    };

    httplib::Client cli("127.0.0.1", h.port);
    cli.set_read_timeout(5, 0);
    auto r1 = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(r1);
    EXPECT_EQ(r1->status, 200);
    EXPECT_EQ(up.calls.load(), 1);

    // Second identical request: cache hit, upstream call count unchanged.
    auto r2 = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(r2);
    EXPECT_EQ(r2->status, 200);
    EXPECT_EQ(up.calls.load(), 1);

    auto stats = cli.Get("/stats");
    ASSERT_TRUE(stats);
    auto j = json::parse(stats->body);
    EXPECT_EQ(j["requests_total"], 2u);
    EXPECT_EQ(j["cache_hits"], 1u);
    EXPECT_EQ(j["upstream_calls"], 1u);
}

TEST(OpenAIProxy, CacheKeyIncludesFullCompiledRequest) {
    ProxyHarness h;
    FakeUpstream up;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions");

    json first = {
        {"model", "gpt-test"},
        {"temperature", 0.1},
        {"messages", json::array({
            {{"role", "user"}, {"content", "explain the project"}}
        })}
    };
    json second = first;
    second["temperature"] = 0.9;

    httplib::Client cli("127.0.0.1", h.port);
    cli.set_read_timeout(5, 0);
    auto r1 = cli.Post("/v1/chat/completions", first.dump(), "application/json");
    ASSERT_TRUE(r1);
    EXPECT_EQ(r1->status, 200);

    auto r2 = cli.Post("/v1/chat/completions", second.dump(), "application/json");
    ASSERT_TRUE(r2);
    EXPECT_EQ(r2->status, 200);

    EXPECT_EQ(up.calls.load(), 2);
}

TEST(OpenAIProxy, BadJsonReturns400) {
    ProxyHarness h;
    FakeUpstream up;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions");
    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Post("/v1/chat/completions", "{ not json", "application/json");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 400);
}

TEST(OpenAIProxy, AuthRejectsMissingBearerBeforeUpstream) {
    ProxyHarness h;
    FakeUpstream up;
    preprocessor::OpenAIProxyConfig cfg;
    cfg.auth.bearer_tokens = {"local-token"};
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions",
            cfg);

    json body = {
        {"model", "gpt-test"},
        {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}
    };
    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 401);
    EXPECT_EQ(up.calls.load(), 0);

    const auto metrics = h.metrics.snapshot();
    EXPECT_EQ(metrics["auth_failures_total"], 1u);
}

TEST(OpenAIProxy, AuthAllowsBearerAndDoesNotLeakLocalTokenWhenDisabledForwarding) {
    ProxyHarness h;
    FakeUpstream up;
    preprocessor::OpenAIProxyConfig cfg;
    cfg.auth.bearer_tokens = {"local-token"};
    cfg.upstream_api_key = "upstream-token";
    cfg.forward_client_authorization = false;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions",
            cfg);

    json body = {
        {"model", "gpt-test"},
        {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}
    };
    httplib::Client cli("127.0.0.1", h.port);
    httplib::Headers headers{{"Authorization", "Bearer local-token"}};
    auto r = cli.Post("/v1/chat/completions", headers, body.dump(), "application/json");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 200);
    EXPECT_EQ(up.calls.load(), 1);
    EXPECT_EQ(up.last_authorization, "Bearer upstream-token");
}

TEST(OpenAIProxy, RateLimitRejectsBeforeCacheOrUpstream) {
    ProxyHarness h;
    FakeUpstream up;
    preprocessor::OpenAIProxyConfig cfg;
    cfg.rate_limit.tokens_per_second = 0.01;
    cfg.rate_limit.burst = 1.0;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions",
            cfg);

    json body = {
        {"model", "gpt-test"},
        {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}
    };
    httplib::Client cli("127.0.0.1", h.port);
    auto r1 = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(r1);
    EXPECT_EQ(r1->status, 200);

    auto r2 = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(r2);
    EXPECT_EQ(r2->status, 429);
    EXPECT_EQ(up.calls.load(), 1);

    const auto metrics = h.metrics.snapshot();
    EXPECT_EQ(metrics["rate_limit_denials_total"], 1u);
}

TEST(OpenAIProxy, RequestBodyTooLargeReturns413BeforeUpstream) {
    ProxyHarness h;
    FakeUpstream up;
    preprocessor::OpenAIProxyConfig cfg;
    cfg.max_request_bytes = 24;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions",
            cfg);

    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Post("/v1/chat/completions",
                      R"({"model":"gpt-test","messages":[{"role":"user","content":"too large"}]})",
                      "application/json");
    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 413);
    EXPECT_EQ(up.calls.load(), 0);

    const auto metrics = h.metrics.snapshot();
    EXPECT_EQ(metrics["request_too_large_denials_total"], 1u);
}

TEST(OpenAIProxy, UpstreamResponseTooLargeReturns502AndDoesNotCache) {
    ProxyHarness h;
    FakeUpstream up;
    up.response_body = R"({"choices":[{"message":{"content":"larger than limit"}}]})";
    preprocessor::OpenAIProxyConfig cfg;
    cfg.upstream_max_response_bytes = 16;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions",
            cfg);

    json body = {
        {"model", "gpt-test"},
        {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}
    };

    httplib::Client cli("127.0.0.1", h.port);
    auto r1 = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(r1);
    EXPECT_EQ(r1->status, 502);
    EXPECT_NE(r1->body.find("upstream response exceeds upstream_max_response_bytes"),
              std::string::npos);

    auto r2 = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(r2);
    EXPECT_EQ(r2->status, 502);
    EXPECT_EQ(up.calls.load(), 2);

    const auto metrics = h.metrics.snapshot();
    EXPECT_EQ(metrics["upstream_errors_total"], 2u);
}

TEST(OpenAIProxy, StreamingForwardsSseWithEventStreamContentType) {
    ProxyHarness h;
    FakeUpstream up;
    up.streaming_response = true;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions");

    json body = {
        {"model", "gpt-test"},
        {"stream", true},
        {"messages", json::array({{{"role", "user"}, {"content", "say hello"}}})}
    };

    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Post("/v1/chat/completions", body.dump(), "application/json");

    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 200);
    EXPECT_EQ(r->get_header_value("Content-Type"), "text/event-stream");
    EXPECT_NE(r->body.find("data: {\"choices\""), std::string::npos);
    EXPECT_NE(r->body.find("data: [DONE]"), std::string::npos);
    EXPECT_EQ(up.calls.load(), 1);

    auto forwarded = json::parse(up.last_body);
    EXPECT_TRUE(forwarded.value("stream", false));
}

TEST(OpenAIProxy, StreamingRequestsAreNotServedFromPromptCache) {
    ProxyHarness h;
    FakeUpstream up;
    up.streaming_response = true;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions");

    json body = {
        {"model", "gpt-test"},
        {"stream", true},
        {"messages", json::array({{{"role", "user"}, {"content", "say hello"}}})}
    };

    httplib::Client cli("127.0.0.1", h.port);
    auto r1 = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(r1);
    EXPECT_EQ(r1->status, 200);

    auto r2 = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_TRUE(r2);
    EXPECT_EQ(r2->status, 200);
    EXPECT_EQ(up.calls.load(), 2);
}

TEST(OpenAIProxy, PreservesUpstreamErrorStatusAndContentType) {
    ProxyHarness h;
    FakeUpstream up;
    up.response_status = 429;
    up.response_content_type = "application/problem+json";
    up.response_body = R"({"error":{"message":"slow down"}})";
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions");

    json body = {
        {"model", "gpt-test"},
        {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}
    };

    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Post("/v1/chat/completions", body.dump(), "application/json");

    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 429);
    EXPECT_EQ(r->get_header_value("Content-Type"), "application/problem+json");
    EXPECT_EQ(r->body, up.response_body);
}

TEST(OpenAIProxy, StreamingTransportFailureEmitsErrorAndDone) {
    ProxyHarness h;
    preprocessor::OpenAIProxyConfig cfg;
    cfg.upstream_timeout_seconds = 1;
    h.start("http://127.0.0.1:1/v1/chat/completions", cfg);

    json body = {
        {"model", "gpt-test"},
        {"stream", true},
        {"messages", json::array({{{"role", "user"}, {"content", "hello"}}})}
    };

    httplib::Client cli("127.0.0.1", h.port);
    cli.set_read_timeout(5, 0);
    auto r = cli.Post("/v1/chat/completions", body.dump(), "application/json");

    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 200);
    EXPECT_EQ(r->get_header_value("Content-Type"), "text/event-stream");
    EXPECT_NE(r->body.find("event: error"), std::string::npos);
    EXPECT_NE(r->body.find("data: [DONE]"), std::string::npos);

    const auto metrics = h.metrics.snapshot();
    EXPECT_EQ(metrics["upstream_errors_total"], 1u);
}

TEST(OpenAIProxy, StreamingClientDisconnectCancelsWithoutSseErrorTrailer) {
    ProxyHarness h;
    FakeUpstream up;
    up.streaming_response = true;
    up.streaming_chunk_count = 64;
    up.streaming_chunk_delay_ms = 2;
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions");

    json body = {
        {"model", "gpt-test"},
        {"stream", true},
        {"messages", json::array({{{"role", "user"}, {"content", "say hello"}}})}
    };

    httplib::Client cli("127.0.0.1", h.port);
    std::string received;
    std::atomic<int> callbacks{0};
    auto r = cli.Post(
        "/v1/chat/completions", httplib::Headers{}, body.dump(), "application/json",
        [&](const char* data, std::size_t len) {
            received.append(data, len);
            callbacks.fetch_add(1);
            return false;
        });

    (void)r;
    EXPECT_GE(callbacks.load(), 1);
    EXPECT_NE(received.find("data: {\"choices\""), std::string::npos);
    EXPECT_EQ(received.find("event: error"), std::string::npos);

    std::uint64_t cancellations = 0;
    for (int i = 0; i < 50; ++i) {
        auto stats = cli.Get("/stats");
        if (stats && stats->status == 200) {
            const auto metrics = json::parse(stats->body);
            cancellations = metrics.value("stream_cancellations_total", 0u);
            if (cancellations > 0) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(cancellations, 1u);
}

TEST(OpenAIProxy, ModelRouterSelectsTierAndRewritesForwardedRequest) {
    ProxyHarness h;
    FakeUpstream default_upstream;
    FakeUpstream routed_upstream;

    preprocessor::ModelRouter router;
    router.add_tier({
        "frontier",
        "http://127.0.0.1:" + std::to_string(routed_upstream.port) + "/v1/chat/completions",
        "gpt-4o-mini",
        "tier-token",
        0
    });
    router.add_route({preprocessor::PromptBucket::CodeEdit, 0, 0, "frontier"});

    preprocessor::OpenAIProxyConfig cfg;
    cfg.model_router = &router;
    cfg.upstream_api_key = "default-token";
    cfg.forward_client_authorization = false;
    h.start("http://127.0.0.1:" + std::to_string(default_upstream.port) +
                "/v1/chat/completions",
            cfg);

    json body = {
        {"model", "gpt-original"},
        {"messages", json::array({{{"role", "user"}, {"content", "fix this bug"}}})}
    };

    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Post("/v1/chat/completions", body.dump(), "application/json");

    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 200);
    EXPECT_EQ(default_upstream.calls.load(), 0);
    EXPECT_EQ(routed_upstream.calls.load(), 1);
    EXPECT_EQ(routed_upstream.last_authorization, "Bearer tier-token");

    auto forwarded = json::parse(routed_upstream.last_body);
    EXPECT_EQ(forwarded["model"], "gpt-4o-mini");

    auto stats = cli.Get("/stats");
    ASSERT_TRUE(stats);
    auto metrics = json::parse(stats->body);
    ASSERT_TRUE(metrics["tokens_by_model_family"].contains("gpt-4o"));
}

TEST(OpenAIProxy, SyncCacheExportRequiresConfiguredAuth) {
    ProxyHarness h;
    FakeUpstream up;
    preprocessor::OpenAIProxyConfig cfg;
    cfg.auth.bearer_tokens = {"sync-token"};
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions",
            cfg);

    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Get("/sync/cache");

    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 401);
}

TEST(OpenAIProxy, SyncCacheImportAndExportWithAuth) {
    ProxyHarness h;
    FakeUpstream up;
    preprocessor::OpenAIProxyConfig cfg;
    cfg.auth.bearer_tokens = {"sync-token"};
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions",
            cfg);

    preprocessor::SyncEndpoint sync;
    preprocessor::SyncBundle bundle;
    bundle.cache.push_back({"team-key", "team-payload"});

    httplib::Client cli("127.0.0.1", h.port);
    httplib::Headers headers{{"X-Preprocessor-Authorization", "Bearer sync-token"}};
    auto imported = cli.Post("/sync/cache", headers, sync.to_json(bundle),
                             "application/json");

    ASSERT_TRUE(imported);
    EXPECT_EQ(imported->status, 200);
    EXPECT_EQ(h.cache.get("team-key").value_or(""), "team-payload");

    auto exported = cli.Get("/sync/cache", headers);
    ASSERT_TRUE(exported);
    EXPECT_EQ(exported->status, 200);
    auto roundtrip = sync.from_json(exported->body);
    ASSERT_EQ(roundtrip.cache.size(), 1u);
    EXPECT_EQ(roundtrip.cache[0].key, "team-key");
    EXPECT_EQ(roundtrip.cache[0].body, "team-payload");
}

TEST(OpenAIProxy, SyncVectorExportRequiresConfiguredAuth) {
    ProxyHarness h;
    FakeUpstream up;
    preprocessor::OpenAIProxyConfig cfg;
    cfg.auth.bearer_tokens = {"sync-token"};
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions",
            cfg);

    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Get("/sync/vectors");

    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 401);
}

TEST(OpenAIProxy, SyncVectorImportAndExportWithAuth) {
    ProxyHarness h;
    FakeUpstream up;
    preprocessor::OpenAIProxyConfig cfg;
    cfg.auth.bearer_tokens = {"sync-token"};
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions",
            cfg);

    preprocessor::SyncEndpoint sync;
    preprocessor::SyncBundle bundle;
    preprocessor::SyncVectorEntry vector;
    vector.chunk_id = 12345;
    vector.vec = std::vector<float>(16, 0.0f);
    vector.vec[0] = 1.0f;
    vector.source_path = "team/alpha.cpp";
    vector.text = "void team_alpha_symbol() {}";
    vector.start_line = 3;
    vector.end_line = 3;
    vector.symbol = "team_alpha_symbol";
    bundle.vectors.push_back(vector);

    httplib::Client cli("127.0.0.1", h.port);
    httplib::Headers headers{{"X-Preprocessor-Authorization", "Bearer sync-token"}};
    auto imported = cli.Post("/sync/vectors", headers, sync.to_json(bundle),
                             "application/json");

    ASSERT_TRUE(imported);
    EXPECT_EQ(imported->status, 200);
    auto imported_json = json::parse(imported->body);
    EXPECT_EQ(imported_json["applied_vector_entries"], 1u);
    EXPECT_EQ(imported_json["bundles_imported"], 1u);
    EXPECT_EQ(h.index.chunk_count(), 1u);

    auto hits = h.index.search("team_alpha_symbol", 3);
    ASSERT_FALSE(hits.empty());
    EXPECT_NE(hits[0].chunk.text.find("team_alpha_symbol"), std::string::npos);

    auto exported = cli.Get("/sync/vectors", headers);
    ASSERT_TRUE(exported);
    EXPECT_EQ(exported->status, 200);
    auto roundtrip = sync.from_json(exported->body);
    ASSERT_EQ(roundtrip.vectors.size(), 1u);
    EXPECT_EQ(roundtrip.vectors[0].chunk_id, 12345u);
    EXPECT_EQ(roundtrip.vectors[0].text, "void team_alpha_symbol() {}");
}

TEST(OpenAIProxy, StructuralFastPathHonorsStreamingSse) {
    ProxyHarness h;
    FakeUpstream up;
    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;
    h.index.attach_symbol_graph(&graph, &extractor);

    const auto dir = std::filesystem::temp_directory_path() /
        ("llm_pp_proxy_struct_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "math.cpp")
        << "int add(int a,int b){ return a+b; }\n";
    h.index.index_path(dir.string());

    preprocessor::StructuralQueryEngine structural(graph, h.index);
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions");
    h.proxy->set_structural_query_engine(&structural);

    json body = {
        {"model", "gpt-test"},
        {"stream", true},
        {"messages", json::array({{{"role", "user"}, {"content", "where is `add`?"}}})}
    };

    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Post("/v1/chat/completions", body.dump(), "application/json");

    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 200);
    EXPECT_EQ(r->get_header_value("Content-Type"), "text/event-stream");
    EXPECT_NE(r->body.find("data: {\"choices\""), std::string::npos);
    EXPECT_NE(r->body.find("data: [DONE]"), std::string::npos);
    EXPECT_EQ(up.calls.load(), 0);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST(OpenAIProxy, StructuralFastPathPreservesCaseSensitiveSymbolNames) {
    ProxyHarness h;
    FakeUpstream up;
    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;
    h.index.attach_symbol_graph(&graph, &extractor);

    const auto dir = std::filesystem::temp_directory_path() /
        ("llm_pp_proxy_case_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "server.cpp")
        << "class HTTPServer { public: void Listen(){} };\n";
    h.index.index_path(dir.string());

    preprocessor::StructuralQueryEngine structural(graph, h.index);
    h.start("http://127.0.0.1:" + std::to_string(up.port) + "/v1/chat/completions");
    h.proxy->set_structural_query_engine(&structural);

    json body = {
        {"model", "gpt-test"},
        {"messages", json::array({{
            {"role", "user"},
            {"content", "Where is `HTTPServer` defined?"}
        }})}
    };

    httplib::Client cli("127.0.0.1", h.port);
    auto r = cli.Post("/v1/chat/completions", body.dump(), "application/json");

    ASSERT_TRUE(r);
    EXPECT_EQ(r->status, 200);
    EXPECT_EQ(up.calls.load(), 0);
    auto parsed = json::parse(r->body);
    const std::string content =
        parsed["choices"][0]["message"]["content"].get<std::string>();
    EXPECT_NE(content.find("`HTTPServer` is defined"), std::string::npos);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}
