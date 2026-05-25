// LLM Preprocessor - Effectiveness Runner
//
// Measures the *real* value each subsystem delivers. Unlike `benchmark_runner`
// (which focuses on the legacy semantic router and requires the ONNX model),
// this runner exercises every Phase 5-12 module against deterministic
// synthetic inputs and reports quantitative effectiveness numbers:
//
//   - PromptCache:        hit speedup, hit-rate under realistic traffic
//   - HeuristicCompressionRewriter: char / token reduction on C/C++ context
//   - StreamingCompactor: char reduction across long chat histories
//   - EmbeddingCache:     persistent hit vs cold recompute speedup
//   - DiffPatcher:        bytes-on-wire savings (diff vs whole-file)
//   - BM25Index:          retrieval latency + top-k accuracy at scale
//   - HybridRetriever-style fusion check (BM25 alone, qualitative)
//   - ModelRouter:        routing decision correctness + latency
//   - AbHarness:          sticky-hash distribution chi-square check
//   - AuthMiddleware:     HMAC verify throughput
//   - RateLimiter:        token-bucket throttle decisions / sec
//
// Output: machine-readable JSON to stdout + a human summary table to stderr.
// Build:  CMake target `effectiveness_runner`.
// Run:    .\build\effectiveness_runner.exe > benchmarks\results\effectiveness.json

#include "ab_harness.hpp"
#include "auth_middleware.hpp"
#include "bm25_index.hpp"
#include "diff_patcher.hpp"
#include "embedding_cache.hpp"
#include "intent_classifier.hpp"
#include "llm_tokenizer.hpp"
#include "model_router.hpp"
#include "prompt_cache.hpp"
#include "prompt_rewriter.hpp"
#include "rate_limiter.hpp"
#include "streaming_compactor.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;
using Clock  = std::chrono::high_resolution_clock;

namespace {

double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

std::string make_tmp_db(const std::string& tag) {
    static std::mt19937_64 rng{std::random_device{}()};
    auto p = fs::temp_directory_path() /
             ("llm_pre_eff_" + tag + "_" + std::to_string(rng()) + ".db");
    fs::remove(p);
    return p.string();
}

// A representative C++-flavoured retrieval snippet with comments + blanks +
// duplicate lines that the heuristic compressor should remove.
std::string sample_cpp_context() {
    return R"CPP(
// ============================================================
// vector_store.cpp - HNSW ANN backend
// Author: someone
// ============================================================

#include "vector_store.hpp"   // public interface
#include <hnswlib/hnswlib.h>  // third-party
#include <stdexcept>


// Build the index with sensible defaults.
namespace preprocessor {

VectorStore::VectorStore(std::size_t dim) : dim_(dim) {
    // initialise the HNSW space
    space_ = std::make_unique<hnswlib::L2Space>(dim);
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), 100000);
}

VectorStore::VectorStore(std::size_t dim) : dim_(dim) {
    // initialise the HNSW space
    space_ = std::make_unique<hnswlib::L2Space>(dim);
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), 100000);
}


/* ---------------- search ---------------- */
std::vector<Hit> VectorStore::search(const float* q, std::size_t k) const {
    // run knn
    auto raw = index_->searchKnn(q, k);
    std::vector<Hit> out;
    while (!raw.empty()) {
        out.push_back({raw.top().second, raw.top().first});
        raw.pop();
    }
    return out;
}

}  // namespace preprocessor
)CPP";
}

std::vector<std::string> sample_doc_corpus() {
    // Synthetic micro-corpus for BM25. Document 7 is the canonical hit for
    // the test query "HNSW vector store nearest neighbour".
    return {
        "Configuration loader reads JSON files and validates required keys.",
        "Tokenizer wraps WordPiece encoding for BERT-class embedding models.",
        "Embedding engine runs ONNX inference and pools the last hidden state.",
        "Intent router computes cosine similarity between query and intents.",
        "Chat history store persists conversation turns in SQLite.",
        "Prompt compiler assembles the final JSON payload for upstream models.",
        "BM25 keyword index ranks documents by Okapi term frequency.",
        "Vector store uses HNSW for fast approximate nearest neighbour search "
        "over chunk embeddings; supports persistent save and load.",
        "Hybrid retriever fuses BM25 and dense vector hits via reciprocal rank fusion.",
        "File watcher uses efsw to react to filesystem changes incrementally.",
        "Symbol graph indexes definitions and references for fast structural queries.",
        "Project card summarises the repository with extension histogram and top symbols.",
        "Prompt cache memoises upstream responses keyed by model, prompt, and chunk ids.",
        "Proxy metrics tracks requests, cache hits, upstream calls, and tokens saved.",
        "Streaming compactor folds older chat turns into a rolling summary string.",
        "Auth middleware supports bearer token allow-lists and HMAC SHA-256 signatures.",
        "Rate limiter implements a per-key token-bucket throttle.",
    };
}

// --------------------------------------------------------------------------
// PromptCache effectiveness: how much faster is a cache hit vs a fresh
// "compute the response" path (here: a deterministic mock payload)?
// --------------------------------------------------------------------------
json measure_prompt_cache() {
    const std::string dbp = make_tmp_db("cache");
    const int kPrompts = 200;
    const int kRequests = 1000;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> pick(0, kPrompts - 1);
    std::vector<std::string> keys;
    keys.reserve(kPrompts);
    for (int i = 0; i < kPrompts; ++i) {
        keys.push_back(preprocessor::PromptCache::make_key(
            "gpt-4", "prompt #" + std::to_string(i),
            {static_cast<std::uint64_t>(i * 7 + 1),
             static_cast<std::uint64_t>(i * 11 + 3)}));
    }

    auto fake_upstream = [](int i) {
        // Simulate the work a real upstream call avoids: hash + small busywork.
        std::string s = "response body " + std::to_string(i);
        for (int k = 0; k < 50; ++k) s += std::to_string(k);
        return s;
    };

    double cold_us = 0, warm_us = 0;
    std::size_t cold_bytes = 0, warm_bytes = 0;
    int hits = 0;
    {
        preprocessor::PromptCache cache(dbp);
        for (int i = 0; i < kPrompts; ++i) cache.put(keys[i], fake_upstream(i));

        auto t0 = Clock::now();
        for (int r = 0; r < kRequests; ++r) cold_bytes += fake_upstream(pick(rng)).size();
        cold_us = us_since(t0);

        t0 = Clock::now();
        for (int r = 0; r < kRequests; ++r) {
            auto v = cache.get(keys[pick(rng)]);
            if (v) { ++hits; warm_bytes += v->size(); }
        }
        warm_us = us_since(t0);
    }
    std::error_code ec; fs::remove(dbp, ec);

    return {
        {"requests", kRequests},
        {"hits", hits},
        {"hit_rate", double(hits) / kRequests},
        {"cold_total_us", cold_us},
        {"warm_total_us", warm_us},
        {"speedup_x", cold_us / std::max(warm_us, 1e-6)},
        {"avg_cold_us", cold_us / kRequests},
        {"avg_warm_us", warm_us / kRequests},
        {"cold_bytes", cold_bytes},
        {"warm_bytes", warm_bytes},
    };
}

// --------------------------------------------------------------------------
// HeuristicCompressionRewriter: char + token reduction on real C++.
// --------------------------------------------------------------------------
json measure_prompt_rewriter() {
    preprocessor::HeuristicCompressionRewriter rw{};
    preprocessor::HeuristicLLMTokenizer tok{};

    const std::string ctx = sample_cpp_context();
    const std::string out = rw.rewrite(ctx, 0);

    const auto tin  = tok.count_tokens(ctx);
    const auto tout = tok.count_tokens(out);

    return {
        {"chars_in", ctx.size()},
        {"chars_out", out.size()},
        {"char_reduction_pct", 100.0 * (1.0 - double(out.size()) / ctx.size())},
        {"tokens_in", tin},
        {"tokens_out", tout},
        {"token_reduction_pct", 100.0 * (1.0 - double(tout) / std::max<std::size_t>(tin, 1))},
        {"tokens_saved", tin > tout ? tin - tout : 0},
    };
}

// --------------------------------------------------------------------------
// StreamingCompactor: char reduction across a long chat.
// --------------------------------------------------------------------------
json measure_streaming_compactor() {
    preprocessor::StreamingCompactor::Config cfg;
    cfg.max_total_chars = 2000;
    cfg.summary_chars_per_turn = 80;
    cfg.keep_recent = 4;
    preprocessor::StreamingCompactor sc(cfg);

    std::vector<preprocessor::ChatTurn> hist;
    for (int i = 0; i < 40; ++i) {
        std::string body = "Turn #" + std::to_string(i) +
                           " - " + std::string(200, char('a' + (i % 26)));
        hist.push_back({i % 2 ? "assistant" : "user", body});
    }

    std::size_t before = 0;
    for (auto& t : hist) before += t.role.size() + t.content.size();

    auto r = sc.compact(hist);
    std::size_t after = r.rolled_summary.size();
    for (auto& t : r.kept) after += t.role.size() + t.content.size();

    return {
        {"turns_in", hist.size()},
        {"turns_kept", r.kept.size()},
        {"turns_rolled", r.rolled},
        {"chars_in", before},
        {"chars_out", after},
        {"char_reduction_pct", 100.0 * (1.0 - double(after) / std::max<std::size_t>(before, 1))},
    };
}

// --------------------------------------------------------------------------
// EmbeddingCache: persistent hit vs fresh "embed".
// --------------------------------------------------------------------------
json measure_embedding_cache() {
    const std::string dbp = make_tmp_db("emb");

    const int kVecs = 500;
    const int kDim  = 384;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    auto fake_embed = [&](const std::string& s) {
        std::vector<float> v(kDim);
        std::seed_seq sd(s.begin(), s.end());
        std::mt19937 g(sd);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (auto& x : v) x = d(g);
        return v;
    };

    std::vector<std::string> docs;
    for (int i = 0; i < kVecs; ++i)
        docs.push_back("chunk #" + std::to_string(i) +
                       " body body body " + std::to_string(u(rng)));

    double cold_us = 0, warm_us = 0;
    int hits = 0;
    std::size_t sz = 0;
    {
        preprocessor::EmbeddingCache cache(dbp, "all-MiniLM-L6-v2");
        auto t0 = Clock::now();
        for (auto& d : docs) {
            auto v = fake_embed(d);
            cache.put(d, v);
        }
        cold_us = us_since(t0);

        t0 = Clock::now();
        for (auto& d : docs) {
            auto v = cache.get(d);
            if (v) ++hits;
        }
        warm_us = us_since(t0);
        sz = cache.size();
    }
    std::error_code ec; fs::remove(dbp, ec);

    return {
        {"vectors", kVecs},
        {"dim", kDim},
        {"hits", hits},
        {"cache_size", sz},
        {"cold_total_us", cold_us},
        {"warm_total_us", warm_us},
        {"speedup_x", cold_us / std::max(warm_us, 1e-6)},
    };
}

// --------------------------------------------------------------------------
// DiffPatcher: bytes-on-wire savings of returning a diff vs the whole file.
// --------------------------------------------------------------------------
json measure_diff_patcher() {
    // Build a moderately large "current" file and a small edit.
    std::string original;
    for (int i = 0; i < 200; ++i) {
        original += "int line_" + std::to_string(i) + "() { return " +
                    std::to_string(i) + "; }\n";
    }
    std::string edited = original;
    auto pos = edited.find("line_42");
    edited.replace(pos, std::string("line_42").size(), "line_42_renamed");

    // Hand-built minimal unified diff.
    std::string diff =
        "--- a/file.cpp\n"
        "+++ b/file.cpp\n"
        "@@ -42,3 +42,3 @@\n"
        " int line_41() { return 41; }\n"
        "-int line_42() { return 42; }\n"
        "+int line_42_renamed() { return 42; }\n"
        " int line_43() { return 43; }\n";

    preprocessor::DiffPatcher patcher;
    std::unordered_map<std::string, std::string> files{{"file.cpp", original}};
    auto result = patcher.apply(diff, files);

    bool applied = result.ok && !result.files.empty() && result.files[0].applied;
    std::size_t full_bytes = edited.size();
    std::size_t diff_bytes = diff.size();

    return {
        {"applied", applied},
        {"full_file_bytes", full_bytes},
        {"diff_bytes", diff_bytes},
        {"wire_savings_pct", 100.0 * (1.0 - double(diff_bytes) / full_bytes)},
    };
}

// --------------------------------------------------------------------------
// BM25Index: top-k accuracy + latency.
// --------------------------------------------------------------------------
json measure_bm25() {
    preprocessor::BM25Index idx;
    auto corpus = sample_doc_corpus();
    for (std::size_t i = 0; i < corpus.size(); ++i) idx.add(i + 1, corpus[i]);

    struct Q { std::string q; std::uint64_t expected; };
    std::vector<Q> queries = {
        {"HNSW vector store nearest neighbour", 8},
        {"BM25 keyword okapi", 7},
        {"reciprocal rank fusion", 9},
        {"token bucket throttle", 17},
        {"HMAC signature bearer", 16},
        {"summarise rolling chat", 15},
        {"prompt cache memoise", 13},
        {"file watcher efsw", 10},
    };

    int top1 = 0, top3 = 0;
    double total_us = 0;
    for (auto& q : queries) {
        auto t0 = Clock::now();
        auto hits = idx.search(q.q, 5);
        total_us += us_since(t0);
        if (!hits.empty() && hits[0].id == q.expected) ++top1;
        for (std::size_t i = 0; i < hits.size() && i < 3; ++i) {
            if (hits[i].id == q.expected) { ++top3; break; }
        }
    }

    return {
        {"docs", corpus.size()},
        {"queries", queries.size()},
        {"top1_correct", top1},
        {"top3_correct", top3},
        {"top1_pct", 100.0 * top1 / queries.size()},
        {"top3_pct", 100.0 * top3 / queries.size()},
        {"avg_query_us", total_us / queries.size()},
    };
}

// --------------------------------------------------------------------------
// ModelRouter: pick the right tier per (bucket, request size).
// --------------------------------------------------------------------------
json measure_model_router() {
    preprocessor::ModelRouter r;
    r.add_tier({"cheap",    "http://up/cheap",    "gpt-cheap",  "", 4096});
    r.add_tier({"medium",   "http://up/medium",   "gpt-medium", "", 16384});
    r.add_tier({"frontier", "http://up/frontier", "gpt-front",  "", 128000});

    // tiny CodeEdit -> cheap
    r.add_route({preprocessor::PromptBucket::CodeEdit, 0,    2000,  "cheap"});
    // medium CodeEdit -> medium
    r.add_route({preprocessor::PromptBucket::CodeEdit, 2000, 16000, "medium"});
    // large CodeGenerate -> frontier
    r.add_route({preprocessor::PromptBucket::CodeGenerate, 8000, 0, "frontier"});

    struct C { preprocessor::PromptBucket b; std::size_t n; std::string expect; };
    std::vector<C> cases = {
        {preprocessor::PromptBucket::CodeEdit,     500,    "cheap"},
        {preprocessor::PromptBucket::CodeEdit,     5000,   "medium"},
        {preprocessor::PromptBucket::CodeGenerate, 50000,  "frontier"},
        {preprocessor::PromptBucket::Freeform,     100,    ""},        // no match
    };

    int correct = 0;
    double total_us = 0;
    for (auto& c : cases) {
        auto t0 = Clock::now();
        const auto* t = r.route(c.b, c.n);
        total_us += us_since(t0);
        std::string got = t ? t->name : "";
        if (got == c.expect) ++correct;
    }

    return {
        {"cases", cases.size()},
        {"correct", correct},
        {"correct_pct", 100.0 * correct / cases.size()},
        {"avg_route_us", total_us / cases.size()},
    };
}

// --------------------------------------------------------------------------
// AbHarness: sticky-hash variant distribution chi-square.
// --------------------------------------------------------------------------
json measure_ab_harness() {
    preprocessor::AbHarness ab;
    preprocessor::AbExperiment exp;
    exp.id = "prompt-style";
    exp.variants = {{"A", 1.0}, {"B", 1.0}, {"C", 2.0}};
    ab.define(exp);

    const int N = 10000;
    int a = 0, b = 0, c = 0;
    for (int i = 0; i < N; ++i) {
        auto v = ab.assign("prompt-style", "client-" + std::to_string(i));
        if (v == "A") ++a;
        else if (v == "B") ++b;
        else if (v == "C") ++c;
    }

    // Chi-square against expected 0.25 / 0.25 / 0.50.
    double exp_a = N * 0.25, exp_b = N * 0.25, exp_c = N * 0.5;
    double chi2 = (a - exp_a) * (a - exp_a) / exp_a +
                  (b - exp_b) * (b - exp_b) / exp_b +
                  (c - exp_c) * (c - exp_c) / exp_c;

    // Stickiness: same key -> same variant.
    int sticky_ok = 0;
    for (int i = 0; i < 100; ++i) {
        auto v1 = ab.assign("prompt-style", "client-" + std::to_string(i));
        auto v2 = ab.assign("prompt-style", "client-" + std::to_string(i));
        if (v1 == v2) ++sticky_ok;
    }

    return {
        {"samples", N},
        {"a", a}, {"b", b}, {"c", c},
        {"chi_square", chi2},   // critical at df=2, p=0.01 is 9.21
        {"sticky_checks", 100},
        {"sticky_ok", sticky_ok},
    };
}

// --------------------------------------------------------------------------
// AuthMiddleware: HMAC verify throughput.
// --------------------------------------------------------------------------
json measure_auth() {
    preprocessor::AuthMiddleware::Config cfg;
    cfg.hmac_secret = "supersecret";
    preprocessor::AuthMiddleware mw(cfg);

    const std::int64_t ts = 1700000000;
    const std::string body = R"({"model":"gpt-4","messages":[]})";
    const std::string sig = preprocessor::AuthMiddleware::sign(cfg.hmac_secret, ts, body);

    const int N = 50000;
    auto t0 = Clock::now();
    int ok = 0;
    for (int i = 0; i < N; ++i) {
        if (mw.verify("", sig, std::to_string(ts), body, ts)) ++ok;
    }
    const double us = us_since(t0);

    // Bad signature must reject.
    bool reject = !mw.verify("", std::string(sig.size(), '0'),
                             std::to_string(ts), body, ts);

    return {
        {"verifications", N},
        {"accepts", ok},
        {"rejects_bad_sig", reject},
        {"total_us", us},
        {"avg_verify_us", us / N},
        {"verifies_per_sec", 1e6 * N / std::max(us, 1.0)},
    };
}

// --------------------------------------------------------------------------
// RateLimiter: bucket decisions per second.
// --------------------------------------------------------------------------
json measure_rate_limiter() {
    preprocessor::RateLimiter rl({/*tps*/100.0, /*burst*/10.0});
    int allowed = 0, denied = 0;
    // 50 requests at t=0 - only 10 should pass (burst).
    for (int i = 0; i < 50; ++i) {
        if (rl.try_acquire("client-A", 0.0)) ++allowed; else ++denied;
    }
    // After 1s -> 100 more tokens available -> cap at burst=10.
    int allowed_t1 = 0;
    for (int i = 0; i < 50; ++i) {
        if (rl.try_acquire("client-A", 1.0)) ++allowed_t1;
    }
    // Different key has its own bucket.
    bool other_ok = rl.try_acquire("client-B", 0.0);

    return {
        {"phase1_allowed", allowed},
        {"phase1_denied",  denied},
        {"phase1_expected_allowed", 10},
        {"phase2_allowed_after_refill", allowed_t1},
        {"different_key_independent", other_ok},
        {"rejected_total", rl.rejected_count()},
    };
}

// --------------------------------------------------------------------------
// Pretty-print summary table to stderr.
// --------------------------------------------------------------------------
void print_summary(const json& report) {
    auto& s = std::cerr;
    s << "\n=== LLM Preprocessor Effectiveness Report ===\n";

    auto pct  = [](double v) { std::ostringstream o; o<<std::fixed<<std::setprecision(1)<<v<<"%"; return o.str(); };
    auto x    = [](double v) { std::ostringstream o; o<<std::fixed<<std::setprecision(1)<<v<<"x"; return o.str(); };
    auto us   = [](double v) { std::ostringstream o; o<<std::fixed<<std::setprecision(2)<<v<<" us"; return o.str(); };

    const auto& pc = report["prompt_cache"];
    s << "\n[PromptCache]          hit rate "<<pct(double(pc["hit_rate"])*100)
      <<" | speedup "<<x(pc["speedup_x"])
      <<" | warm "<<us(pc["avg_warm_us"])
      <<" vs cold "<<us(pc["avg_cold_us"])<<"\n";

    const auto& rw = report["prompt_rewriter"];
    s << "[PromptRewriter]       chars "<<rw["chars_in"]<<" -> "<<rw["chars_out"]
      <<" ("<<pct(rw["char_reduction_pct"])
      <<") | tokens "<<rw["tokens_in"]<<" -> "<<rw["tokens_out"]
      <<" ("<<pct(rw["token_reduction_pct"])<<")\n";

    const auto& sc = report["streaming_compactor"];
    s << "[StreamingCompactor]   "<<sc["turns_in"]<<" turns -> "
      <<sc["turns_kept"]<<" kept + "<<sc["turns_rolled"]<<" rolled | "
      <<sc["chars_in"]<<" -> "<<sc["chars_out"]<<" chars ("<<pct(sc["char_reduction_pct"])<<")\n";

    const auto& ec = report["embedding_cache"];
    s << "[EmbeddingCache]       "<<ec["vectors"]<<" vectors | speedup "
      <<x(ec["speedup_x"])<<" (cold "<<us(double(ec["cold_total_us"])/double(ec["vectors"]))
      <<" -> warm "<<us(double(ec["warm_total_us"])/double(ec["vectors"]))<<")\n";

    const auto& dp = report["diff_patcher"];
    s << "[DiffPatcher]          "<<dp["full_file_bytes"]<<" B full vs "
      <<dp["diff_bytes"]<<" B diff ("<<pct(dp["wire_savings_pct"])<<" saved)"
      <<" | applied=" << (dp["applied"] ? "yes" : "no") << "\n";

    const auto& bm = report["bm25_index"];
    s << "[BM25Index]            "<<bm["docs"]<<" docs / "<<bm["queries"]
      <<" queries | top-1 "<<pct(bm["top1_pct"])<<" | top-3 "<<pct(bm["top3_pct"])
      <<" | "<<us(bm["avg_query_us"])<<"/query\n";

    const auto& mr = report["model_router"];
    s << "[ModelRouter]          "<<mr["correct"]<<"/"<<mr["cases"]
      <<" correct ("<<pct(mr["correct_pct"])<<") | "<<us(mr["avg_route_us"])<<"/route\n";

    const auto& ab = report["ab_harness"];
    s << "[AbHarness]            "<<ab["samples"]<<" assigns | A="<<ab["a"]
      <<" B="<<ab["b"]<<" C="<<ab["c"]<<" | chi^2="
      <<std::fixed<<std::setprecision(2)<<double(ab["chi_square"])
      <<" (crit 9.21) | sticky="<<ab["sticky_ok"]<<"/"<<ab["sticky_checks"]<<"\n";

    const auto& au = report["auth_middleware"];
    s << "[AuthMiddleware]       "<<au["verifications"]<<" HMAC verifies | "
      <<us(au["avg_verify_us"])<<"/op | "
      <<std::fixed<<std::setprecision(0)<<double(au["verifies_per_sec"])<<"/s\n";

    const auto& rl = report["rate_limiter"];
    s << "[RateLimiter]          phase1 allowed "<<rl["phase1_allowed"]
      <<"/50 (expected 10) | phase2 allowed "<<rl["phase2_allowed_after_refill"]
      <<" | independent-key="<<(rl["different_key_independent"] ? "yes" : "no")<<"\n";

    s << "\n(JSON written to stdout)\n";
}

}  // namespace

int main() {
    json report;
    try {
        report["prompt_cache"]        = measure_prompt_cache();
        report["prompt_rewriter"]     = measure_prompt_rewriter();
        report["streaming_compactor"] = measure_streaming_compactor();
        report["embedding_cache"]     = measure_embedding_cache();
        report["diff_patcher"]        = measure_diff_patcher();
        report["bm25_index"]          = measure_bm25();
        report["model_router"]        = measure_model_router();
        report["ab_harness"]          = measure_ab_harness();
        report["auth_middleware"]     = measure_auth();
        report["rate_limiter"]        = measure_rate_limiter();
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] effectiveness runner failed: " << e.what() << "\n";
        return 1;
    }

    print_summary(report);
    std::cout << report.dump(2) << "\n";
    return 0;
}
