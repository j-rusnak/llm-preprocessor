// End-to-end smoke runner for the coding-agent preprocessor pipeline.
//
// Drives every new module against an in-memory synthetic "repo" and prints
// PASS/FAIL lines. This is the manual integration framework the user can run
// to validate the local middleware pipeline (chunk -> hash -> index -> retrieve
// -> tokenize -> watch).
//
// It does NOT depend on the ONNX model — embeddings are simulated with a
// deterministic hash-based projection so the smoke runner stays fast and
// hermetic. Real-embedding tests live in test_embedding_engine.cpp.
//
// Usage:
//   ./smoke_runner             # run all
//   ./smoke_runner --verbose   # extra detail per stage

#include "code_chunker.hpp"
#include "file_watcher.hpp"
#include "llm_tokenizer.hpp"
#include "vector_store.hpp"
// Phase 1:
#include "bm25_index.hpp"
#include "hybrid_retriever.hpp"
#include "i_embedding_engine.hpp"
#include "openai_proxy.hpp"
#include "prompt_cache.hpp"
#include "proxy_metrics.hpp"
#include "repo_index.hpp"
// Phase 2:
#include "intent_classifier.hpp"
#include "project_card.hpp"
#include "prompt_optimizer.hpp"
#include "prompt_templates.hpp"
// Phase 3:
#include "graph_aware_retriever.hpp"
#include "mcp_server.hpp"
#include "prompt_rewriter.hpp"
#include "diff_patcher.hpp"
#include "embedding_cache.hpp"
#include "model_router.hpp"
#include "ab_harness.hpp"
#include "sync_endpoint.hpp"
#include "streaming_compactor.hpp"
#include "auth_middleware.hpp"
#include "rate_limiter.hpp"
#include "structural_query_engine.hpp"
#include "symbol_graph.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <httplib.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Stats {
    int passed = 0;
    int failed = 0;
};

void report(Stats& s, const std::string& name, bool ok, const std::string& detail = "") {
    if (ok) ++s.passed; else ++s.failed;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
    if (!detail.empty()) std::cout << " - " << detail;
    std::cout << "\n";
}

// Deterministic 256-dim projection from chunk-id -> unit vector. Stands in
// for a real embedding model so the smoke runner is hermetic.
std::vector<float> fake_embed(std::uint64_t seed, std::size_t dim = 64) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(dim);
    float sq = 0.0f;
    for (auto& x : v) { x = d(rng); sq += x * x; }
    const float n = std::sqrt(sq);
    if (n > 0.0f) for (auto& x : v) x /= n;
    return v;
}

std::vector<std::pair<std::string, std::string>> make_fake_repo() {
    return {
        {"src/math.cpp",
         "int add(int a, int b) { return a + b; }\n"
         "int sub(int a, int b) { return a - b; }\n"
         "int mul(int a, int b) { return a * b; }\n"},
        {"src/io.cpp",
         "void read_file(const std::string& p) { /* ... */ }\n"
         "void write_file(const std::string& p) { /* ... */ }\n"},
        {"include/math.hpp",
         "#pragma once\nint add(int, int);\nint sub(int, int);\nint mul(int, int);\n"},
    };
}

void stage_chunker(Stats& s, bool verbose) {
    std::cout << "[chunker]\n";
    preprocessor::LineWindowChunker chunker(/*window*/ 3, /*overlap*/ 1);
    std::size_t total = 0;
    for (const auto& [path, src] : make_fake_repo()) {
        auto chunks = chunker.chunk(path, src);
        if (verbose) {
            std::cout << "    " << path << " -> " << chunks.size() << " chunks\n";
        }
        total += chunks.size();
    }
    report(s, "chunker produced chunks", total > 0,
           std::to_string(total) + " total");
}

void stage_vector_index_and_search(Stats& s, bool verbose) {
    std::cout << "[vector_store + chunker]\n";
    preprocessor::LineWindowChunker chunker(3, 1);
    preprocessor::VectorStore store(64, 1024);

    std::vector<preprocessor::CodeChunk> all;
    for (const auto& [path, src] : make_fake_repo()) {
        auto cs = chunker.chunk(path, src);
        for (auto& c : cs) {
            store.add(c.id, fake_embed(c.id));
        }
        all.insert(all.end(), cs.begin(), cs.end());
    }

    report(s, "indexed all chunks", store.size() == all.size(),
           "indexed " + std::to_string(store.size()));

    // Self-recall: query each chunk -> top hit should be itself.
    int recall = 0;
    for (const auto& c : all) {
        auto hits = store.search(fake_embed(c.id), 1);
        if (!hits.empty() && hits.front().id == c.id) ++recall;
    }
    report(s, "self-recall @ 1", recall == static_cast<int>(all.size()),
           std::to_string(recall) + "/" + std::to_string(all.size()));

    // Persistence round-trip.
    auto tmp = fs::temp_directory_path() / "llm_pp_smoke_index.bin";
    store.save(tmp.string());
    try {
        preprocessor::VectorStore reloaded(64, 1024, tmp.string());
        bool ok = reloaded.size() == store.size();
        if (ok && !all.empty()) {
            auto hits = reloaded.search(fake_embed(all.front().id), 1);
            ok = !hits.empty() && hits.front().id == all.front().id;
        }
        report(s, "persisted index round-trips", ok);
    } catch (const std::exception& e) {
        report(s, "persisted index round-trips", false, e.what());
    }
    std::error_code ec;
    fs::remove(tmp, ec);
    (void)verbose;
}

void stage_tokenizer(Stats& s, bool) {
    std::cout << "[llm_tokenizer]\n";
    preprocessor::HeuristicLLMTokenizer tk;
    auto few = tk.count_tokens("hello world");
    auto many = tk.count_tokens(std::string(4000, 'x'));
    report(s, "tokenizer monotonic", many > few,
           "few=" + std::to_string(few) + " many=" + std::to_string(many));
}

void stage_file_watcher(Stats& s, bool verbose) {
    std::cout << "[file_watcher]\n";
    auto dir = fs::temp_directory_path() /
               ("llm_pp_smoke_fw_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);

    std::atomic<int> events{0};
    preprocessor::FileWatcher watcher;
    long id = watcher.add_watch(dir.string(), [&](const preprocessor::FileEvent& ev) {
        events.fetch_add(1, std::memory_order_relaxed);
        if (verbose) std::cout << "    event: " << ev.path << "\n";
    });
    if (id < 0) {
        report(s, "file_watcher attaches", false, "add_watch returned " + std::to_string(id));
        return;
    }

    for (int i = 0; i < 3; ++i) {
        std::ofstream(dir / ("f" + std::to_string(i) + ".txt")) << i;
    }
    for (int i = 0; i < 30 && events.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    watcher.remove_watch(id);
    report(s, "file_watcher fires", events.load() > 0,
           std::to_string(events.load()) + " events");

    std::error_code ec;
    fs::remove_all(dir, ec);
}

// ----------------------------- Phase 1 stages -----------------------------

namespace p1 {

class HashEmbedder : public preprocessor::IEmbeddingEngine {
public:
    explicit HashEmbedder(std::size_t dim) : dim_(dim) {}
    std::vector<float> generate_embedding(const std::string& text) override {
        std::vector<float> v(dim_, 0.0f);
        std::size_t h = std::hash<std::string>{}(text);
        for (std::size_t i = 0; i < dim_; ++i)
            v[i] = static_cast<float>(((h >> (i % 32)) & 0xFF) / 255.0);
        float n = 0.0f; for (float x : v) n += x*x;
        n = n > 0 ? std::sqrt(n) : 1.0f;
        for (auto& x : v) x /= n;
        return v;
    }
private:
    std::size_t dim_;
};

} // namespace p1

void stage_bm25(Stats& s, bool) {
    std::cout << "[bm25_index]\n";
    preprocessor::BM25Index idx;
    idx.add(1, "void compute_hash(int x) { return x * 31; }");
    idx.add(2, "void render_screen() { draw_frame(); }");
    idx.add(3, "int main() { return 0; }");
    auto hits = idx.search("compute hash", 5);
    bool ok = !hits.empty() && hits[0].id == 1;
    report(s, "BM25 ranks identifier match first", ok);
    idx.remove(1);
    auto h2 = idx.search("compute", 5);
    bool removed = h2.empty() || h2[0].id != 1;
    report(s, "BM25 remove() drops doc", removed);
}

void stage_hybrid(Stats& s, bool) {
    std::cout << "[hybrid_retriever]\n";
    preprocessor::VectorStore vec(8, 64);
    preprocessor::BM25Index bm;
    auto axis = [](int a){ std::vector<float> v(8,0); v[a%8]=1; return v; };
    vec.add(1, axis(0)); bm.add(1, "alpha");
    vec.add(2, axis(1)); bm.add(2, "beta gamma");
    vec.add(3, axis(2)); bm.add(3, "delta epsilon");
    preprocessor::HybridRetriever h(vec, bm);
    auto hits = h.search("beta", axis(0), 3);
    bool saw1 = false, saw2 = false;
    for (auto& r : hits) { if (r.id == 1) saw1 = true; if (r.id == 2) saw2 = true; }
    report(s, "RRF fuses vector + keyword hits", saw1 && saw2);
}

void stage_prompt_cache(Stats& s, bool) {
    std::cout << "[prompt_cache]\n";
    try {
        preprocessor::PromptCache c(":memory:");
        auto k = preprocessor::PromptCache::make_key("m", "p", {3,1,2});
        c.put(k, "payload-1");
        auto got = c.get(k);
        report(s, "put/get round trip", got.has_value() && *got == "payload-1");
        auto k2 = preprocessor::PromptCache::make_key("m", "p", {1,2,3});
        report(s, "key is order-invariant in chunk ids", k == k2);
    } catch (const std::exception& e) {
        report(s, "prompt_cache stage", false, e.what());
    }
}

void stage_repo_index(Stats& s, bool verbose) {
    std::cout << "[repo_index]\n";
    auto dir = fs::temp_directory_path() /
               ("llm_pp_smoke_repo_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    {
        std::ofstream(dir / "a.cpp")
            << "int compute_hash(int x) { return x * 31; }\n";
        std::ofstream(dir / "b.cpp")
            << "void render_screen() { draw_frame(); }\n";
    }

    try {
        auto embedder = std::make_shared<p1::HashEmbedder>(16);
        auto chunker  = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
        preprocessor::RepoIndexConfig cfg;
        cfg.embedding_dim = 16;
        cfg.watch_for_changes = false;
        preprocessor::RepoIndex idx(embedder, chunker, cfg);
        idx.index_path(dir.string());
        if (verbose) std::cout << "    files=" << idx.file_count()
                               << " chunks=" << idx.chunk_count() << "\n";
        report(s, "indexed synthetic repo", idx.chunk_count() > 0);

        auto hits = idx.search("compute_hash", 3);
        bool ok = !hits.empty() &&
                  hits[0].chunk.file_path.find("a.cpp") != std::string::npos;
        report(s, "hybrid search returns relevant chunk", ok);
    } catch (const std::exception& e) {
        report(s, "repo_index stage", false, e.what());
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void stage_proxy(Stats& s, bool) {
    std::cout << "[openai_proxy]\n";
    using nlohmann::json;

    // Fake upstream.
    auto upstream = std::make_shared<httplib::Server>();
    std::atomic<int> upstream_calls{0};
    upstream->Post("/v1/chat/completions",
                   [&](const httplib::Request&, httplib::Response& res) {
        upstream_calls.fetch_add(1);
        json out = {
            {"id","fake"},
            {"choices", json::array({
                {{"message", {{"role","assistant"},{"content","ok"}}}}
            })}
        };
        res.set_content(out.dump(), "application/json");
    });
    int up_port = upstream->bind_to_any_port("127.0.0.1");
    std::thread up_thr([&]{ upstream->listen_after_bind(); });

    try {
        auto embedder = std::make_shared<p1::HashEmbedder>(16);
        auto chunker  = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
        preprocessor::RepoIndexConfig icfg;
        icfg.embedding_dim = 16;
        icfg.watch_for_changes = false;
        preprocessor::RepoIndex idx(embedder, chunker, icfg);
        preprocessor::PromptCache cache(":memory:");
        preprocessor::ProxyMetrics metrics;
        preprocessor::HeuristicLLMTokenizer tk;
        preprocessor::OpenAIProxyConfig pcfg;
        pcfg.upstream_url = "http://127.0.0.1:" + std::to_string(up_port)
                          + "/v1/chat/completions";
        preprocessor::OpenAIProxy proxy(idx, cache, metrics, tk, pcfg);
        int p_port = proxy.bind_to_port("127.0.0.1", 0);
        std::thread p_thr([&]{ proxy.listen_after_bind(); });

        // wait for health
        for (int i = 0; i < 100; ++i) {
            httplib::Client cli("127.0.0.1", p_port);
            auto r = cli.Get("/healthz");
            if (r && r->status == 200) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        httplib::Client cli("127.0.0.1", p_port);
        cli.set_read_timeout(5, 0);
        json body = {
            {"model","gpt-test"},
            {"messages", json::array({
                {{"role","user"},{"content","hello world"}}
            })}
        };
        auto r1 = cli.Post("/v1/chat/completions", body.dump(), "application/json");
        report(s, "proxy forwards request", r1 && r1->status == 200);

        auto r2 = cli.Post("/v1/chat/completions", body.dump(), "application/json");
        report(s, "proxy serves second hit from cache",
               r2 && r2->status == 200 && upstream_calls.load() == 1);

        auto stats = cli.Get("/stats");
        bool stats_ok = false;
        if (stats && stats->status == 200) {
            auto j = json::parse(stats->body);
            stats_ok = j["requests_total"] == 2u && j["cache_hits"] == 1u;
        }
        report(s, "stats endpoint reports counters", stats_ok);

        proxy.stop();
        p_thr.join();
    } catch (const std::exception& e) {
        report(s, "openai_proxy stage", false, e.what());
    }

    upstream->stop();
    up_thr.join();
}

// ----------------------------- Phase 2 stages -----------------------------

void stage_intent_classifier(Stats& s, bool) {
    std::cout << "[intent_classifier]\n";
    preprocessor::HeuristicIntentClassifier c;
    bool ok = c.classify("refactor build_context_block") == preprocessor::PromptBucket::CodeEdit
           && c.classify("explain why this loop terminates") == preprocessor::PromptBucket::CodeExplain
           && c.classify("write a parser for json") == preprocessor::PromptBucket::CodeGenerate
           && c.classify("which files import sqlite") == preprocessor::PromptBucket::MetaQuery;
    report(s, "heuristic classifier routes 4 archetypes", ok);
}

void stage_project_card(Stats& s, bool verbose) {
    std::cout << "[project_card]\n";
    auto dir = fs::temp_directory_path() /
               ("llm_pp_smoke_card_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    {
        std::ofstream(dir / "a.cpp") << "int compute_thing(){return 1;}\n";
        std::ofstream(dir / "b.cpp") << "void do_stuff(){}\n";
        std::ofstream(dir / "README.md") << "My project README\n";
    }
    try {
        auto emb = std::make_shared<p1::HashEmbedder>(16);
        auto chunker = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
        preprocessor::RepoIndexConfig cfg;
        cfg.embedding_dim = 16;
        cfg.watch_for_changes = false;
        preprocessor::RepoIndex idx(emb, chunker, cfg);
        idx.index_path(dir.string());
        auto card = preprocessor::ProjectCardBuilder::build(idx, dir.string(), 30, 64);
        if (verbose) std::cout << "    files=" << card.total_files
                               << " chunks=" << card.total_chunks << "\n";
        bool ok = card.total_files >= 2 &&
                  card.files_by_extension[".cpp"] == 2 &&
                  card.readme_excerpt.find("My project") != std::string::npos;
        report(s, "card aggregates extensions + README", ok);
        auto md = card.to_markdown();
        report(s, "markdown rendering non-empty",
               md.find(".cpp") != std::string::npos);
    } catch (const std::exception& e) {
        report(s, "project_card stage", false, e.what());
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void stage_prompt_optimizer(Stats& s, bool) {
    std::cout << "[prompt_optimizer]\n";
    try {
        auto cls = std::make_shared<preprocessor::HeuristicIntentClassifier>();
        preprocessor::PromptOptimizerConfig cfg;
        cfg.enabled = true;
        cfg.include_project_card = false;
        preprocessor::PromptOptimizer opt(preprocessor::PromptTemplates{}, cls, cfg);

        preprocessor::RetrievedChunk rc;
        rc.score = 1.0f;
        rc.chunk.id = 7;
        rc.chunk.text = "int add(int a,int b){return a+b;}";
        rc.chunk.file_path = "src/math.cpp";
        rc.chunk.symbol = "add";
        rc.chunk.start_line = 1;
        rc.chunk.end_line = 1;

        auto r = opt.optimise("explain why this function works", {rc});
        bool ok = r.used_template &&
                  r.bucket == preprocessor::PromptBucket::CodeExplain &&
                  r.system_message.find("src/math.cpp") != std::string::npos;
        report(s, "enabled optimiser renders template with context", ok);

        opt.set_enabled(false);
        auto r2 = opt.optimise("explain why this function works", {rc});
        report(s, "disabled optimiser falls back to plain block",
               !r2.used_template &&
               r2.system_message.find("Retrieved code context") != std::string::npos);

        auto r3 = opt.optimise("hi", {});
        report(s, "empty chunks + disabled card -> empty message", r3.system_message.empty());
    } catch (const std::exception& e) {
        report(s, "prompt_optimizer stage", false, e.what());
    }
}

} // namespace

namespace {

void stage_symbol_graph(Stats& s, bool) {
    std::cout << "[symbol_graph]\n";
    try {
        preprocessor::SymbolGraph g;
        preprocessor::RegexSymbolExtractor ex;
        preprocessor::CodeChunk defc;
        defc.id = 1; defc.file_path = "src/sg.cpp"; defc.start_line = 1; defc.end_line = 1;
        defc.text = "void target(){}\n";
        preprocessor::CodeChunk callc;
        callc.id = 2; callc.file_path = "src/sg2.cpp"; callc.start_line = 1; callc.end_line = 1;
        callc.text = "void caller(){ target(); }\n";
        g.update_chunk(defc, ex.extract(defc));
        g.update_chunk(callc, ex.extract(callc));
        report(s, "extractor finds defs", g.definition_count() >= 2);
        report(s, "extractor finds refs", g.reference_count() >= 1);
        report(s, "definition lookup", !g.find_definitions("target").empty());
        report(s, "reference lookup", !g.find_references("target").empty());
        g.remove_file("src/sg.cpp");
        g.remove_file("src/sg2.cpp");
        report(s, "remove_file empties graph", g.definition_count() == 0 && g.reference_count() == 0);
    } catch (const std::exception& e) {
        report(s, "symbol_graph stage", false, e.what());
    }
}

void stage_graph_expansion(Stats& s, bool) {
    std::cout << "[graph_expansion]\n";
    try {
        auto emb = std::make_shared<p1::HashEmbedder>(16);
        auto chunker = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
        preprocessor::RepoIndexConfig cfg;
        cfg.embedding_dim = 16;
        cfg.watch_for_changes = false;
        preprocessor::RepoIndex index(emb, chunker, cfg);
        preprocessor::SymbolGraph graph;
        preprocessor::RegexSymbolExtractor ex;
        index.attach_symbol_graph(&graph, &ex);

        auto tmp = fs::temp_directory_path() / ("sg_smoke_" + std::to_string(
                       std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(tmp);
        std::ofstream(tmp / "h.cpp") << "void shared_helper(){}\n";
        std::ofstream(tmp / "c.cpp") << "void caller(){ shared_helper(); }\n";
        index.index_path(tmp.string());

        auto seeds = index.search("caller", 1);
        report(s, "seeds non-empty", !seeds.empty());
        auto expanded = preprocessor::expand_with_graph(seeds, graph, index);
        bool got_helper = false;
        for (const auto& r : expanded)
            if (r.chunk.file_path.find("h.cpp") != std::string::npos) got_helper = true;
        report(s, "graph expansion appends neighbour chunk", got_helper);
        std::error_code ec; fs::remove_all(tmp, ec);
    } catch (const std::exception& e) {
        report(s, "graph_expansion stage", false, e.what());
    }
}

void stage_structural_query(Stats& s, bool) {
    std::cout << "[structural_query]\n";
    try {
        auto emb = std::make_shared<p1::HashEmbedder>(16);
        auto chunker = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
        preprocessor::RepoIndexConfig cfg;
        cfg.embedding_dim = 16;
        cfg.watch_for_changes = false;
        preprocessor::RepoIndex index(emb, chunker, cfg);
        preprocessor::SymbolGraph graph;
        preprocessor::RegexSymbolExtractor ex;
        index.attach_symbol_graph(&graph, &ex);

        auto tmp = fs::temp_directory_path() / ("sq_smoke_" + std::to_string(
                       std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(tmp);
        std::ofstream(tmp / "math.cpp")
            << "int add(int a,int b){return a+b;}\nint caller(){return add(1,2);}\n";
        index.index_path(tmp.string());

        preprocessor::StructuralQueryEngine eng(graph, index);
        auto def = eng.try_answer("where is `add`?");
        report(s, "definition fast-path returns an answer", def.has_value());
        auto cal = eng.try_answer("what calls `add`");
        report(s, "callers fast-path returns an answer", cal.has_value());
        auto stats = eng.try_answer("how many chunks are in the repo?");
        report(s, "repo stats fast-path returns an answer", stats.has_value());
        auto none = eng.try_answer("write me a haiku about pointers");
        report(s, "freeform message returns nullopt", !none.has_value());
        std::error_code ec; fs::remove_all(tmp, ec);
    } catch (const std::exception& e) {
        report(s, "structural_query stage", false, e.what());
    }
}

void stage_mcp_server(Stats& s, bool) {
    std::cout << "[mcp_server]\n";
    try {
        auto emb = std::make_shared<p1::HashEmbedder>(16);
        auto chunker = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
        preprocessor::RepoIndexConfig cfg;
        cfg.embedding_dim = 16;
        cfg.watch_for_changes = false;
        preprocessor::RepoIndex index(emb, chunker, cfg);

        auto tmp = fs::temp_directory_path() / ("mcp_smoke_" + std::to_string(
                       std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(tmp);
        std::ofstream(tmp / "math.cpp")
            << "int add(int a,int b){return a+b;}\nint caller(){return add(1,2);}\n";
        index.index_path(tmp.string());

        preprocessor::McpServer server(index);
        nlohmann::json init = {
            {"jsonrpc", "2.0"}, {"id", 1}, {"method", "initialize"}};
        auto init_rep = nlohmann::json::parse(server.handle_message(init.dump()));
        report(s, "initialize advertises serverInfo",
               init_rep["result"]["serverInfo"]["name"] == "llm-preprocessor");

        nlohmann::json tlist = {
            {"jsonrpc", "2.0"}, {"id", 2}, {"method", "tools/list"}};
        auto tlist_rep = nlohmann::json::parse(server.handle_message(tlist.dump()));
        report(s, "tools/list returns 3 tools",
               tlist_rep["result"]["tools"].size() == 3);

        nlohmann::json call = {
            {"jsonrpc", "2.0"}, {"id", 3}, {"method", "tools/call"},
            {"params", {{"name", "search_repo"},
                        {"arguments", {{"query", "add"}, {"k", 2}}}}}};
        auto call_rep = nlohmann::json::parse(server.handle_message(call.dump()));
        report(s, "search_repo returns content",
               !call_rep["result"]["content"].empty());

        nlohmann::json rlist = {
            {"jsonrpc", "2.0"}, {"id", 4}, {"method", "resources/list"}};
        auto rlist_rep = nlohmann::json::parse(server.handle_message(rlist.dump()));
        report(s, "resources/list returns 2 resources",
               rlist_rep["result"]["resources"].size() == 2);

        nlohmann::json note = {
            {"jsonrpc", "2.0"}, {"method", "notifications/initialized"}};
        report(s, "notifications yield empty reply",
               server.handle_message(note.dump()).empty());

        std::error_code ec; fs::remove_all(tmp, ec);
    } catch (const std::exception& e) {
        report(s, "mcp_server stage", false, e.what());
    }
}

void stage_prompt_rewriter(Stats& s, bool) {
    std::cout << "[prompt_rewriter]\n";
    try {
        preprocessor::HeuristicCompressionRewriter r;
        std::string in =
            "// File: src/math.cpp:1-6  (chunk_id=1)\n"
            "/* header */\n"
            "#include <cstdint>\n\n\n\n"
            "// returns sum\n"
            "int add(int a, int b) {\n"
            "    // trivial\n"
            "    return a + b;   \n"
            "}\n";
        auto out = r.rewrite(in, 0);
        report(s, "heuristic compressor reduces size", out.size() < in.size());
        report(s, "keeps include directive", out.find("#include <cstdint>") != std::string::npos);
        report(s, "strips inline comments", out.find("trivial") == std::string::npos);
        report(s, "keeps code body", out.find("int add") != std::string::npos);

        auto truncated = r.rewrite(std::string(400, 'x'), 64);
        report(s, "max_chars hint truncates", truncated.size() <= 96);

#ifndef LLM_PREPROCESSOR_WITH_LLAMA_CPP
        bool threw = false;
        try {
            preprocessor::LlamaCppRewriterConfig cfg;
            cfg.model_path = "none.gguf";
            preprocessor::LlamaCppRewriter rr(cfg);
            (void)rr;
        } catch (const std::exception&) { threw = true; }
        report(s, "LlamaCppRewriter throws without build flag", threw);
#else
        report(s, "LlamaCppRewriter compiled in", preprocessor::LlamaCppRewriter::is_available());
#endif
    } catch (const std::exception& e) {
        report(s, "prompt_rewriter stage", false, e.what());
    }
}

void stage_phase6_through_12(Stats& s, bool) {
    std::cout << "[phase6_12]\n";
    try {
        // Phase 6: DiffPatcher
        preprocessor::DiffPatcher dp;
        std::string diff =
            "--- a/x.txt\n"
            "+++ b/x.txt\n"
            "@@ -1,2 +1,2 @@\n"
            " hello\n"
            "-world\n"
            "+earth\n";
        std::unordered_map<std::string, std::string> contents{{"x.txt", "hello\nworld\n"}};
        auto patched = dp.apply(diff, contents);
        report(s, "diff parses & applies", patched.ok && !patched.files.empty() &&
               patched.files[0].patched_content.find("earth") != std::string::npos);

        // Phase 7: EmbeddingCache
        auto db = std::filesystem::temp_directory_path() / "smoke_emb.db";
        std::error_code ec; std::filesystem::remove(db, ec);
        preprocessor::EmbeddingCache cache(db.string(), "m");
        std::vector<float> v{1.0f, 2.0f, 3.0f};
        cache.put("code", v);
        auto got = cache.get("code");
        report(s, "embedding cache roundtrips", got.has_value() && got->size() == 3);

        // Phase 8: ModelRouter
        preprocessor::ModelRouter router;
        preprocessor::ModelTier t; t.name = "cheap"; t.upstream_url = "http://x"; t.model_name = "m";
        router.add_tier(t);
        preprocessor::ModelRoute r; r.bucket = preprocessor::PromptBucket::CodeExplain; r.tier = "cheap";
        router.add_route(r);
        auto* sel = router.route(preprocessor::PromptBucket::CodeExplain, 100);
        report(s, "router selects tier", sel && sel->name == "cheap");

        // Phase 9: AbHarness
        preprocessor::AbHarness ab;
        preprocessor::AbExperiment exp; exp.id = "e1";
        exp.variants.push_back({"a", 1.0});
        exp.variants.push_back({"b", 1.0});
        ab.define(exp);
        auto v1 = ab.assign("e1", "user-1");
        auto v2 = ab.assign("e1", "user-1");
        report(s, "ab harness is sticky", v1 == v2 && !v1.empty());

        // Phase 10: SyncEndpoint
        preprocessor::SyncEndpoint sync;
        preprocessor::SyncBundle b;
        b.cache.push_back({"k", "val"});
        auto j = sync.to_json(b);
        auto back = sync.from_json(j);
        report(s, "sync json roundtrip", back.cache.size() == 1 && back.cache[0].key == "k");

        // Phase 11: StreamingCompactor
        preprocessor::StreamingCompactor::Config cc;
        cc.max_total_chars = 50;
        cc.keep_recent = 1;
        preprocessor::StreamingCompactor comp(cc);
        std::vector<preprocessor::ChatTurn> h;
        for (int i = 0; i < 8; ++i) h.push_back({"user", "message " + std::to_string(i)});
        auto cr = comp.compact(h);
        report(s, "compactor rolls older turns", cr.rolled > 0 && !cr.rolled_summary.empty());

        // Phase 12: AuthMiddleware + RateLimiter
        preprocessor::AuthMiddleware::Config acfg; acfg.hmac_secret = "shh";
        preprocessor::AuthMiddleware auth(acfg);
        auto sig = preprocessor::AuthMiddleware::sign("shh", 100, "body");
        report(s, "auth hmac verifies", auth.verify("", sig, "100", "body", 100));
        report(s, "auth hmac rejects wrong sig", !auth.verify("", "deadbeef", "100", "body", 100));

        preprocessor::RateLimiter::Config rcfg;
        rcfg.tokens_per_second = 1.0; rcfg.burst = 1.0;
        preprocessor::RateLimiter rl(rcfg);
        bool a1 = rl.try_acquire("k", 1.0);
        bool a2 = rl.try_acquire("k", 1.0);
        report(s, "rate limiter throttles", a1 && !a2);
    } catch (const std::exception& e) {
        report(s, "phase6_12 stage", false, e.what());
    }
}

} // namespace

int main(int argc, char** argv) {
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
    }

    std::cout << "=== LLM Preprocessor :: Middleware Smoke Runner ===\n\n";

    Stats s;
    stage_chunker(s, verbose);
    stage_vector_index_and_search(s, verbose);
    stage_tokenizer(s, verbose);
    stage_file_watcher(s, verbose);
    stage_bm25(s, verbose);
    stage_hybrid(s, verbose);
    stage_prompt_cache(s, verbose);
    stage_repo_index(s, verbose);
    stage_proxy(s, verbose);
    stage_intent_classifier(s, verbose);
    stage_project_card(s, verbose);
    stage_prompt_optimizer(s, verbose);
    stage_symbol_graph(s, verbose);
    stage_graph_expansion(s, verbose);
    stage_structural_query(s, verbose);
    stage_mcp_server(s, verbose);
    stage_prompt_rewriter(s, verbose);
    stage_phase6_through_12(s, verbose);

    std::cout << "\nSummary: " << s.passed << " passed, " << s.failed << " failed.\n";
    return s.failed == 0 ? 0 : 1;
}
