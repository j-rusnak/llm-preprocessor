#include "code_chunker.hpp"
#include "i_embedding_engine.hpp"
#include "llm_tokenizer.hpp"
#include "openai_proxy.hpp"
#include "prompt_cache.hpp"
#include "proxy_metrics.hpp"
#include "repo_index.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
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

    FakeUpstream() : server(std::make_shared<httplib::Server>()) {
        server->Post("/v1/chat/completions",
                     [this](const httplib::Request& req, httplib::Response& res) {
            calls.fetch_add(1);
            last_body = req.body;
            last_authorization = req.has_header("Authorization")
                ? req.get_header_value("Authorization")
                : std::string{};
            if (streaming_response) {
                res.set_chunked_content_provider(
                    "text/event-stream",
                    [](std::size_t, httplib::DataSink& sink) {
                        const std::string first =
                            "data: {\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n";
                        const std::string second =
                            "data: {\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n";
                        const std::string done = "data: [DONE]\n\n";
                        if (!sink.write(first.data(), first.size())) return false;
                        if (!sink.write(second.data(), second.size())) return false;
                        if (!sink.write(done.data(), done.size())) return false;
                        sink.done();
                        return true;
                    });
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
            res.set_content(out.dump(), "application/json");
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
    preprocessor::HeuristicLLMTokenizer tokenizer;
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
