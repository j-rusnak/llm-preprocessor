#include "openai_proxy.hpp"

#include "graph_aware_retriever.hpp"
#include "intent_classifier.hpp"
#include "llm_tokenizer.hpp"
#include "model_router.hpp"
#include "prompt_cache.hpp"
#include "prompt_optimizer.hpp"
#include "prompt_rewriter.hpp"
#include "proxy_metrics.hpp"
#include "auth_middleware.hpp"
#include "rate_limiter.hpp"
#include "repo_index.hpp"
#include "structural_query_engine.hpp"
#include "symbol_graph.hpp"

#include <algorithm>
#include <cctype>
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
    std::string content_type = "application/json";
};

struct StreamingSink {
    httplib::DataSink* sink = nullptr;
};

std::size_t curl_stream_cb(char* ptr, std::size_t size, std::size_t nmemb, void* userdata) {
    auto* out = static_cast<StreamingSink*>(userdata);
    const std::size_t bytes = size * nmemb;
    if (!out || !out->sink) return 0;
    return out->sink->write(ptr, bytes) ? bytes : 0;
}

std::string trim_header_value(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' ||
                          s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    std::size_t first = 0;
    while (first < s.size() && (s[first] == ' ' || s[first] == '\t')) {
        ++first;
    }
    if (first > 0) s.erase(0, first);
    return s;
}

std::size_t curl_header_cb(char* buffer,
                           std::size_t size,
                           std::size_t nitems,
                           void* userdata) {
    auto* resp = static_cast<UpstreamResponse*>(userdata);
    const std::size_t bytes = size * nitems;
    if (!resp) return bytes;

    const std::string line(buffer, bytes);
    const auto colon = line.find(':');
    if (colon == std::string::npos) return bytes;

    std::string name = line.substr(0, colon);
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (name == "content-type") {
        resp->content_type = trim_header_value(line.substr(colon + 1));
        if (resp->content_type.empty()) {
            resp->content_type = "application/json";
        }
    }
    return bytes;
}

void set_json_error(httplib::Response& res, int status,
                    const std::string& type,
                    const std::string& message) {
    res.status = status;
    res.set_content(json{{"error", json{{"type", type}, {"message", message}}}}.dump(),
                    "application/json");
}

std::string first_header(const httplib::Request& req,
                         const char* primary,
                         const char* fallback = nullptr) {
    if (req.has_header(primary)) return req.get_header_value(primary);
    if (fallback && req.has_header(fallback)) return req.get_header_value(fallback);
    return {};
}

std::string rate_limit_key(const httplib::Request& req) {
    std::string auth = first_header(req, "X-Preprocessor-Authorization", "Authorization");
    if (!auth.empty()) return "auth:" + auth;
    if (!req.remote_addr.empty()) return "ip:" + req.remote_addr;
    return "anonymous";
}

std::string normalize_user_message(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    bool in_space = true;
    for (unsigned char c : input) {
        if (std::isspace(c)) {
            if (!in_space) {
                out.push_back(' ');
                in_space = true;
            }
            continue;
        }
        out.push_back(static_cast<char>(c));
        in_space = false;
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

bool enforce_proxy_controls(const httplib::Request& req,
                            httplib::Response& res,
                            const AuthMiddleware& auth,
                            RateLimiter& rate_limiter,
                            const std::string& body) {
    if (auth.enabled()) {
        const std::string auth_header =
            first_header(req, "X-Preprocessor-Authorization", "Authorization");
        const std::string signature =
            first_header(req, "X-Preprocessor-Signature", "X-Signature");
        const std::string timestamp =
            first_header(req, "X-Preprocessor-Timestamp", "X-Timestamp");
        if (!auth.verify(auth_header, signature, timestamp, body)) {
            set_json_error(res, 401, "unauthorized",
                           "missing or invalid proxy authentication");
            return false;
        }
    }
    if (rate_limiter.enabled() && !rate_limiter.try_acquire(rate_limit_key(req))) {
        set_json_error(res, 429, "rate_limited", "proxy rate limit exceeded");
        return false;
    }
    return true;
}

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
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, curl_header_cb);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &resp);
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

void write_sse_error(httplib::DataSink& sink, const std::string& message) {
    const std::string event =
        "event: error\n"
        "data: " + json{{"error", message}}.dump() + "\n\n";
    sink.write(event.data(), event.size());
}

void write_sse_done(httplib::DataSink& sink) {
    const std::string done = "data: [DONE]\n\n";
    sink.write(done.data(), done.size());
}

bool stream_synthetic_completion(const std::string& id,
                                 long long created,
                                 const std::string& model,
                                 const std::string& content,
                                 httplib::DataSink& sink) {
    const json first = {
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", json::array({
            json{{"index", 0},
                 {"delta", json{{"role", "assistant"}, {"content", content}}},
                 {"finish_reason", nullptr}}
        })},
        {"x_preprocessor", json{{"source", "structural_query_engine"}}}
    };
    const json last = {
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", created},
        {"model", model},
        {"choices", json::array({
            json{{"index", 0},
                 {"delta", json::object()},
                 {"finish_reason", "stop"}}
        })},
        {"x_preprocessor", json{{"source", "structural_query_engine"}}}
    };
    const std::string first_event = "data: " + first.dump() + "\n\n";
    const std::string last_event = "data: " + last.dump() + "\n\n";
    const std::string done = "data: [DONE]\n\n";
    if (!sink.write(first_event.data(), first_event.size())) return false;
    if (!sink.write(last_event.data(), last_event.size())) return false;
    if (!sink.write(done.data(), done.size())) return false;
    sink.done();
    return true;
}

bool stream_upstream(const std::string& url,
                     const std::string& body,
                     const std::string& incoming_auth,
                     const std::string& fallback_key,
                     long timeout_seconds,
                     httplib::DataSink& sink) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        write_sse_error(sink, "curl_easy_init failed");
        write_sse_done(sink);
        sink.done();
        return true;
    }

    StreamingSink stream{&sink};
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_stream_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: text/event-stream");
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
        write_sse_error(sink, std::string("upstream: ") + curl_easy_strerror(rc));
        write_sse_done(sink);
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    sink.done();
    return true;
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
      auth_(config_.auth),
      rate_limiter_(config_.rate_limit),
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

void OpenAIProxy::set_prompt_rewriter(IPromptRewriter* rewriter) noexcept {
    rewriter_ = rewriter;
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

    server_->Get("/stats", [this](const httplib::Request& req, httplib::Response& res) {
        if (!enforce_proxy_controls(req, res, auth_, rate_limiter_, "")) {
            return;
        }
        res.set_content(metrics_.snapshot().dump(2), "application/json");
    });

    server_->Get("/sync/cache", [this](const httplib::Request& req,
                                       httplib::Response& res) {
        if (!enforce_proxy_controls(req, res, auth_, rate_limiter_, "")) {
            return;
        }

        SyncBundle bundle;
        try {
            for (const auto& entry :
                 cache_.snapshot(config_.sync_cache_export_limit)) {
                bundle.cache.push_back({entry.key, entry.payload});
            }
            sync_.note_export();
            res.set_content(sync_.to_json(bundle), "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 500, "sync_export_failed", e.what());
        }
    });

    server_->Post("/sync/cache", [this](const httplib::Request& req,
                                        httplib::Response& res) {
        if (config_.max_request_bytes > 0 &&
            req.body.size() > config_.max_request_bytes) {
            set_json_error(res, 413, "request_too_large",
                           "request body exceeds proxy_max_request_bytes");
            return;
        }
        if (!enforce_proxy_controls(req, res, auth_, rate_limiter_, req.body)) {
            return;
        }

        auto parsed = json::parse(req.body, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            set_json_error(res, 400, "invalid_json",
                           "sync bundle must be a JSON object");
            return;
        }

        try {
            SyncBundle bundle = sync_.from_json(req.body);
            const std::size_t applied = sync_.apply_to_cache(bundle, &cache_);
            res.set_content(
                json{{"applied_cache_entries", applied},
                     {"bundles_imported", sync_.bundles_imported()}}.dump(),
                "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid_sync_bundle", e.what());
        }
    });

    server_->Get("/sync/vectors", [this](const httplib::Request& req,
                                         httplib::Response& res) {
        if (!enforce_proxy_controls(req, res, auth_, rate_limiter_, "")) {
            return;
        }

        try {
            SyncBundle bundle;
            bundle.vectors =
                index_.snapshot_vectors(config_.sync_vector_export_limit);
            sync_.note_export();
            res.set_content(sync_.to_json(bundle), "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 500, "sync_export_failed", e.what());
        }
    });

    server_->Post("/sync/vectors", [this](const httplib::Request& req,
                                          httplib::Response& res) {
        if (config_.max_request_bytes > 0 &&
            req.body.size() > config_.max_request_bytes) {
            set_json_error(res, 413, "request_too_large",
                           "request body exceeds proxy_max_request_bytes");
            return;
        }
        if (!enforce_proxy_controls(req, res, auth_, rate_limiter_, req.body)) {
            return;
        }

        auto parsed = json::parse(req.body, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            set_json_error(res, 400, "invalid_json",
                           "sync bundle must be a JSON object");
            return;
        }

        try {
            SyncBundle bundle = sync_.from_json(req.body);
            const std::size_t applied =
                index_.apply_synced_vectors(bundle.vectors);
            sync_.note_import();
            res.set_content(
                json{{"applied_vector_entries", applied},
                     {"bundles_imported", sync_.bundles_imported()}}.dump(),
                "application/json");
        } catch (const std::exception& e) {
            set_json_error(res, 400, "invalid_sync_bundle", e.what());
        }
    });

    server_->Post("/v1/chat/completions",
                  [this](const httplib::Request& req, httplib::Response& res) {
        metrics_.on_request();
        if (config_.max_request_bytes > 0 &&
            req.body.size() > config_.max_request_bytes) {
            metrics_.on_error();
            set_json_error(res, 413, "request_too_large",
                           "request body exceeds proxy_max_request_bytes");
            return;
        }
        if (auth_.enabled()) {
            const std::string auth_header =
                first_header(req, "X-Preprocessor-Authorization", "Authorization");
            const std::string signature =
                first_header(req, "X-Preprocessor-Signature", "X-Signature");
            const std::string timestamp =
                first_header(req, "X-Preprocessor-Timestamp", "X-Timestamp");
            if (!auth_.verify(auth_header, signature, timestamp, req.body)) {
                metrics_.on_error();
                set_json_error(res, 401, "unauthorized",
                               "missing or invalid proxy authentication");
                return;
            }
        }
        if (rate_limiter_.enabled() && !rate_limiter_.try_acquire(rate_limit_key(req))) {
            metrics_.on_error();
            set_json_error(res, 429, "rate_limited",
                           "proxy rate limit exceeded");
            return;
        }
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception& e) {
            metrics_.on_error();
            res.status = 400;
            set_json_error(res, 400, "invalid_json",
                           std::string("invalid JSON: ") + e.what());
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            metrics_.on_error();
            set_json_error(res, 400, "invalid_request", "missing messages[]");
            return;
        }

        const bool stream = body.value("stream", false);
        std::string user_msg = normalize_user_message(last_user_message(body["messages"]));
        std::string effective_model = body.value("model", std::string{"unknown"});
        std::string upstream_url = config_.upstream_url;
        std::string upstream_api_key = config_.upstream_api_key;
        std::size_t max_context_chars = config_.max_context_chars;

        if (config_.model_router) {
            HeuristicIntentClassifier classifier;
            const PromptBucket bucket = classifier.classify(user_msg);
            if (const ModelTier* tier =
                    config_.model_router->route(bucket, req.body.size())) {
                if (!tier->upstream_url.empty()) {
                    upstream_url = tier->upstream_url;
                }
                if (!tier->api_key.empty()) {
                    upstream_api_key = tier->api_key;
                }
                if (!tier->model_name.empty()) {
                    effective_model = tier->model_name;
                    body["model"] = tier->model_name;
                }
                if (tier->max_context > 0) {
                    max_context_chars =
                        max_context_chars == 0
                            ? tier->max_context
                            : (std::min)(max_context_chars, tier->max_context);
                }
            }
        }

        // Phase 3: zero-LLM fast path for purely structural questions.
        if (structural_engine_ && !user_msg.empty()) {
            try {
                auto answer = structural_engine_->try_answer(user_msg);
                if (answer) {
                    metrics_.on_cache_hit(); // counts as an upstream-avoided hit
                    auto now = std::chrono::system_clock::now().time_since_epoch();
                    long long ts = std::chrono::duration_cast<std::chrono::seconds>(now).count();
                    const std::string id =
                        std::string("local-structural-") + std::to_string(ts);
                    if (stream) {
                        const std::string answer_text = *answer;
                        const std::string model = effective_model;
                        res.set_header("Cache-Control", "no-cache");
                        res.set_chunked_content_provider(
                            "text/event-stream",
                            [id, ts, model, answer_text]
                            (std::size_t, httplib::DataSink& sink) {
                                return stream_synthetic_completion(
                                    id, ts, model, answer_text, sink);
                            });
                        return;
                    }
                    json synthetic = {
                        {"id", id},
                        {"object", "chat.completion"},
                        {"created", ts},
                        {"model", effective_model},
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
                GraphExpansionConfig graph_cfg;
                graph_cfg.query_text = user_msg;
                retrieved = expand_with_graph(retrieved, *symbol_graph_,
                                              index_, graph_cfg);
            } catch (const std::exception&) {
                // Expansion failures are non-fatal.
            }
        }

        std::vector<std::uint64_t> chunk_ids;
        chunk_ids.reserve(retrieved.size());
        for (const auto& r : retrieved) chunk_ids.push_back(r.chunk.id);

        // Inject retrieved context as a system message.
        const std::size_t original_tokens =
            tokenizer_.count_tokens_for_model(effective_model, req.body);
        std::string sys_content;
        if (optimiser_) {
            auto opt = optimiser_->optimise(user_msg, retrieved);
            sys_content = std::move(opt.system_message);
        } else if (!retrieved.empty()) {
            sys_content = build_context_block(retrieved, max_context_chars);
        }
        if (rewriter_ && !sys_content.empty()) {
            try {
                sys_content = rewriter_->rewrite(sys_content, max_context_chars);
            } catch (...) {
                // Rewriter failures are non-fatal: keep the pre-rewrite block.
            }
        }
        if (!sys_content.empty()) {
            json sys_msg = {{"role", "system"}, {"content", sys_content}};
            // Place context right before the last user message so the LLM
            // treats it as fresh grounding.
            auto& msgs = body["messages"];
            msgs.insert(msgs.end() - 1, sys_msg);
        }
        const std::string compiled = body.dump();
        // Cache the fully compiled upstream request, not just the user text.
        // Request parameters such as temperature, tools, existing system
        // messages, optimiser output, and rewritten context all affect the
        // response and must participate in cache identity.
        const std::string cache_key =
            PromptCache::make_key(effective_model, compiled, chunk_ids);
        if (!stream) {
            if (auto cached = cache_.get(cache_key)) {
                metrics_.on_cache_hit();
                res.set_content(*cached, "application/json");
                return;
            }
        }

        const std::size_t compiled_tokens =
            tokenizer_.count_tokens_for_model(effective_model, compiled);
        metrics_.observe_tokens(tokenizer_.model_family(effective_model),
                                original_tokens, compiled_tokens);

        std::string incoming_auth;
        if (config_.forward_client_authorization &&
            req.has_header("Authorization")) {
            incoming_auth = req.get_header_value("Authorization");
        }

        if (stream) {
            metrics_.on_upstream_call();
            const auto target_url = upstream_url;
            const auto fallback_key = upstream_api_key;
            const auto timeout_seconds = config_.upstream_timeout_seconds;
            res.set_header("Cache-Control", "no-cache");
            res.set_chunked_content_provider(
                "text/event-stream",
                [target_url, compiled, incoming_auth, fallback_key, timeout_seconds]
                (std::size_t, httplib::DataSink& sink) {
                    return stream_upstream(target_url, compiled, incoming_auth,
                                           fallback_key, timeout_seconds, sink);
                });
            return;
        }

        // Forward to upstream.
        UpstreamResponse up;
        try {
            metrics_.on_upstream_call();
            up = forward_upstream(upstream_url, compiled, incoming_auth,
                                  upstream_api_key,
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
        res.set_content(up.body, up.content_type.c_str());
    });
}

} // namespace preprocessor
