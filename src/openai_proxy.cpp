#include "openai_proxy.hpp"

#include "graph_aware_retriever.hpp"
#include "llm_tokenizer.hpp"
#include "prompt_cache.hpp"
#include "prompt_optimizer.hpp"
#include "proxy_metrics.hpp"
#include "repo_index.hpp"
#include "structural_query_engine.hpp"
#include "symbol_graph.hpp"
#include "text_sanitizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <curl/curl.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace preprocessor {

namespace {

using nlohmann::json;

/// libcurl write callback that appends to a std::string.
std::size_t curl_write_cb(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

struct UpstreamResponse {
    long status = 0;
    std::string body;
};

/// Forward a JSON body to the upstream chat-completions endpoint via libcurl.
/// `incoming_auth` is the Authorization header from the client request (may be
/// empty). `fallback_key` is used when the client provided none.
UpstreamResponse forward_upstream(const std::string& url,
                                  const std::string& body,
                                  const std::string& incoming_auth,
                                  const std::string& fallback_key,
                                  long timeout_seconds) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    UpstreamResponse resp;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp.body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth;
    if (!incoming_auth.empty()) {
        auth = "Authorization: " + incoming_auth;
    } else if (!fallback_key.empty()) {
        auth = "Authorization: Bearer " + fallback_key;
    }
    if (!auth.empty()) headers = curl_slist_append(headers, auth.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        std::string err = curl_easy_strerror(rc);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        throw std::runtime_error("curl_easy_perform: " + err);
    }
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &resp.status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return resp;
}

/// Extract the last user message string from an OpenAI-style messages array.
std::string last_user_message(const json& messages) {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->is_object() && it->value("role", std::string{}) == "user") {
            auto c = it->find("content");
            if (c == it->end()) continue;
            if (c->is_string()) return c->get<std::string>();
            // OpenAI's multimodal content arrays — concatenate text parts.
            if (c->is_array()) {
                std::string acc;
                for (const auto& part : *c) {
                    if (part.is_object() && part.value("type", std::string{}) == "text") {
                        acc += part.value("text", std::string{});
                        acc += "\n";
                    }
                }
                return acc;
            }
        }
    }
    return {};
}

/// Build a context system message from retrieved chunks, capped by total
/// character budget. Earlier (higher-scoring) chunks win when the budget is
/// tight.
std::string build_context_block(const std::vector<RetrievedChunk>& chunks,
                                std::size_t max_chars) {
    std::ostringstream out;
    out << "Retrieved code context (most relevant first):\n";
    std::size_t used = 0;
    for (const auto& rc : chunks) {
        std::ostringstream entry;
        entry << "\n--- " << rc.chunk.file_path
              << " [L" << rc.chunk.start_line << "-L" << rc.chunk.end_line << "]";
        if (!rc.chunk.symbol.empty()) entry << " " << rc.chunk.symbol;
        entry << " ---\n" << rc.chunk.text << "\n";
        const std::string s = entry.str();
        if (max_chars > 0 && used + s.size() > max_chars) break;
        out << s;
        used += s.size();
    }
    return out.str();
}

} // namespace

OpenAIProxy::OpenAIProxy(RepoIndex& index,
                         PromptCache& cache,
                         ProxyMetrics& metrics,
                         ILLMTokenizer& tokenizer,
                         OpenAIProxyConfig config)
    : index_(index),
      cache_(cache),
      metrics_(metrics),
      tokenizer_(tokenizer),
      config_(std::move(config)),
      server_(std::make_unique<httplib::Server>()) {
    if (config_.upstream_url.empty()) {
        throw std::invalid_argument("OpenAIProxy: upstream_url must not be empty");
    }
    install_routes();
}

OpenAIProxy::~OpenAIProxy() {
    if (server_ && server_->is_running()) server_->stop();
}

void OpenAIProxy::set_prompt_optimizer(PromptOptimizer* optimiser) noexcept {
    optimiser_ = optimiser;
}

void OpenAIProxy::set_symbol_graph(SymbolGraph* graph) noexcept {
    symbol_graph_ = graph;
}

void OpenAIProxy::set_structural_query_engine(StructuralQueryEngine* engine) noexcept {
    structural_engine_ = engine;
}

int OpenAIProxy::bind_to_port(const std::string& host, int port) {
    // httplib 0.38 returns bool from bind_to_port; use bind_to_any_port for
    // ephemeral binding so we can discover the actually-chosen port (port=0).
    if (port == 0) {
        int bound = server_->bind_to_any_port(host.c_str());
        if (bound <= 0) {
            throw std::runtime_error("OpenAIProxy: bind_to_any_port failed on " + host);
        }
        return bound;
    }
    if (!server_->bind_to_port(host.c_str(), port)) {
        throw std::runtime_error("OpenAIProxy: bind_to_port failed on " +
                                 host + ":" + std::to_string(port));
    }
    return port;
}

void OpenAIProxy::listen_after_bind() {
    server_->listen_after_bind();
}

void OpenAIProxy::listen(const std::string& host, int port) {
    int bound = bind_to_port(host, port);
    (void)bound;
    listen_after_bind();
}

void OpenAIProxy::stop() {
    if (server_) server_->stop();
}

void OpenAIProxy::install_routes() {
    server_->Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });

    server_->Get("/stats", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(metrics_.snapshot().dump(2), "application/json");
    });

    server_->Post("/v1/chat/completions",
                  [this](const httplib::Request& req, httplib::Response& res) {
        metrics_.on_request();
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            metrics_.on_error();
            res.status = 400;
            res.set_content(json{{"error", std::string("invalid JSON: ") + e.what()}}.dump(),
                            "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            metrics_.on_error();
            res.status = 400;
            res.set_content(json{{"error", "missing messages[]"}}.dump(),
                            "application/json");
            return;
        }

        const std::string model = body.value("model", std::string{"unknown"});
        std::string user_msg = TextSanitizer::sanitize(last_user_message(body["messages"]));

        // Phase 3: zero-LLM fast path for purely structural questions.
        if (structural_engine_ && !user_msg.empty()) {
            try {
                auto answer = structural_engine_->try_answer(user_msg);
                if (answer) {
                    metrics_.on_cache_hit(); // counts as an upstream-avoided hit
                    auto now = std::chrono::system_clock::now().time_since_epoch();
                    long long ts = std::chrono::duration_cast<std::chrono::seconds>(now).count();
                    json synthetic = {
                        {"id", std::string("local-structural-") + std::to_string(ts)},
                        {"object", "chat.completion"},
                        {"created", ts},
                        {"model", model},
                        {"choices", json::array({
                            json{{"index", 0},
                                 {"message", json{{"role", "assistant"},
                                                  {"content", *answer}}},
                                 {"finish_reason", "stop"}}
                        })},
                        {"usage", json{{"prompt_tokens", 0},
                                       {"completion_tokens", 0},
                                       {"total_tokens", 0}}},
                        {"x_preprocessor", json{{"source", "structural_query_engine"}}}
                    };
                    res.set_content(synthetic.dump(), "application/json");
                    return;
                }
            } catch (const std::exception&) {
                // Fast-path failures are non-fatal: fall through to normal flow.
            }
        }

        // Retrieval.
        std::vector<RetrievedChunk> retrieved;
        if (!user_msg.empty()) {
            try {
                retrieved = index_.search(user_msg, config_.retrieval_k);
            } catch (const std::exception&) {
                // Retrieval failures are non-fatal: forward without context.
                retrieved.clear();
            }
        }

        // Phase 3: optional graph-aware expansion of retrieval results.
        if (symbol_graph_ && !retrieved.empty()) {
            try {
                retrieved = expand_with_graph(retrieved, *symbol_graph_, index_);
            } catch (const std::exception&) {
                // Expansion failures are non-fatal.
            }
        }

        std::vector<std::uint64_t> chunk_ids;
        chunk_ids.reserve(retrieved.size());
        for (const auto& r : retrieved) chunk_ids.push_back(r.chunk.id);

        // Cache lookup BEFORE mutating the body so the key is deterministic
        // across identical requests.
        const std::string cache_key = PromptCache::make_key(model, user_msg, chunk_ids);
        if (auto cached = cache_.get(cache_key)) {
            metrics_.on_cache_hit();
            res.set_content(*cached, "application/json");
            return;
        }

        // Inject retrieved context as a system message.
        const std::size_t original_tokens = tokenizer_.count_tokens(req.body);
        std::string sys_content;
        if (optimiser_) {
            auto opt = optimiser_->optimise(user_msg, retrieved);
            sys_content = std::move(opt.system_message);
        } else if (!retrieved.empty()) {
            sys_content = build_context_block(retrieved, config_.max_context_chars);
        }
        if (!sys_content.empty()) {
            json sys_msg = {{"role", "system"}, {"content", sys_content}};
            // Place context right before the last user message so the LLM
            // treats it as fresh grounding.
            auto& msgs = body["messages"];
            msgs.insert(msgs.end() - 1, sys_msg);
        }
        const std::string compiled = body.dump();
        const std::size_t compiled_tokens = tokenizer_.count_tokens(compiled);
        metrics_.observe_tokens(original_tokens, compiled_tokens);

        // Forward to upstream.
        UpstreamResponse up;
        try {
            metrics_.on_upstream_call();
            std::string incoming_auth;
            if (req.has_header("Authorization")) {
                incoming_auth = req.get_header_value("Authorization");
            }
            up = forward_upstream(config_.upstream_url, compiled, incoming_auth,
                                  config_.upstream_api_key,
                                  config_.upstream_timeout_seconds);
        } catch (const std::exception& e) {
            metrics_.on_error();
            res.status = 502;
            res.set_content(json{{"error", std::string("upstream: ") + e.what()}}.dump(),
                            "application/json");
            return;
        }

        // Only cache successful responses.
        if (up.status >= 200 && up.status < 300) {
            try { cache_.put(cache_key, up.body); }
            catch (...) { /* cache write failures are non-fatal */ }
        }
        res.status = static_cast<int>(up.status);
        res.set_content(up.body, "application/json");
    });
}

} // namespace preprocessor
