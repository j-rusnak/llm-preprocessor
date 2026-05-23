// Effectiveness regression suite.
//
// Locks in *minimum* effectiveness thresholds for every subsystem so that
// a regression in compression ratio, retrieval quality, or routing latency
// fails CI. These thresholds are deliberately conservative; the
// `effectiveness_runner` benchmark reports the actual numbers.

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

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

namespace {

double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

std::string tmp_db(const char* tag) {
    static std::mt19937_64 rng{std::random_device{}()};
    auto p = fs::temp_directory_path() /
             ("llm_eff_" + std::string(tag) + "_" + std::to_string(rng()) + ".db");
    fs::remove(p);
    return p.string();
}

// The same C++-like context used by effectiveness_runner so the gtest
// thresholds and the benchmark numbers stay aligned.
std::string sample_context() {
    return R"CPP(
// header comment block
// another comment line

#include <vector>   // standard header
#include <string>   // standard header

namespace x {

// duplicated definition block (deliberate)
int compute(int a, int b) { return a + b; }
int compute(int a, int b) { return a + b; }


/* multi-line
   block comment that
   should be stripped */
int product(int a, int b) {
    return a * b; // trailing
}

}  // namespace x
)CPP";
}

}  // namespace

// ---------- PromptRewriter ----------
TEST(Effectiveness_PromptRewriter, ReducesCharsAtLeast15Percent) {
    preprocessor::HeuristicCompressionRewriter r;
    auto in  = sample_context();
    auto out = r.rewrite(in, 0);
    double reduction = 1.0 - double(out.size()) / double(in.size());
    EXPECT_GE(reduction, 0.15) << "out=" << out.size() << " in=" << in.size();
}

TEST(Effectiveness_PromptRewriter, ReducesTokensWhenCompressing) {
    preprocessor::HeuristicCompressionRewriter r;
    preprocessor::HeuristicLLMTokenizer tok;
    auto in  = sample_context();
    auto out = r.rewrite(in, 0);
    EXPECT_LT(tok.count_tokens(out), tok.count_tokens(in));
}

// ---------- StreamingCompactor ----------
TEST(Effectiveness_StreamingCompactor, StaysUnderHardCap) {
    preprocessor::StreamingCompactor::Config cfg;
    cfg.max_total_chars = 1500;
    cfg.summary_chars_per_turn = 60;
    cfg.keep_recent = 3;
    preprocessor::StreamingCompactor sc(cfg);

    std::vector<preprocessor::ChatTurn> hist;
    for (int i = 0; i < 30; ++i)
        hist.push_back({i % 2 ? "user" : "assistant", std::string(300, 'x')});

    auto r = sc.compact(hist);
    std::size_t kept_chars = 0;
    for (auto& t : r.kept) kept_chars += t.content.size() + t.role.size() + 4;
    EXPECT_LE(kept_chars, cfg.max_total_chars)
        << "compactor kept window exceeded budget";
    EXPECT_GT(r.rolled, 0u) << "long history should roll some turns";
    EXPECT_LE(r.kept.size(), 30u);
}

// ---------- PromptCache ----------
TEST(Effectiveness_PromptCache, HitsAreFasterThanFreshComputeBy3x) {
    auto db = tmp_db("cache");
    double cold_us = 0, warm_us = 0;
    int hits = 0;
    const int N = 2000;
    auto cold = []() {
        // Approximate the work a real upstream call avoids: serialize, hash,
        // and concatenate a non-trivial response body.
        std::string s;
        s.reserve(8192);
        for (int i = 0; i < 400; ++i) s += std::to_string(i * 31 + 7);
        std::uint64_t h = 1469598103934665603ull;
        for (char c : s) { h ^= (unsigned char)c; h *= 1099511628211ull; }
        s += std::to_string(h);
        for (int i = 0; i < 200; ++i) s += "padding-block-";
        return s;
    };
    {
        preprocessor::PromptCache cache(db);
        auto k = preprocessor::PromptCache::make_key("m", "p", {1, 2, 3});
        cache.put(k, std::string(2048, 'a'));

        auto t0 = Clock::now();
        for (int i = 0; i < N; ++i) (void)cold();
        cold_us = us_since(t0);

        t0 = Clock::now();
        for (int i = 0; i < N; ++i) if (cache.get(k)) ++hits;
        warm_us = us_since(t0);
    }
    std::error_code ec; fs::remove(db, ec);
    EXPECT_EQ(hits, N);
    EXPECT_GT(cold_us, warm_us * 3.0)
        << "cold=" << cold_us << "us warm=" << warm_us << "us";
}

// ---------- EmbeddingCache ----------
TEST(Effectiveness_EmbeddingCache, RoundTripsVectorsByModelId) {
    auto db = tmp_db("emb");
    {
        preprocessor::EmbeddingCache c1(db, "model-A");
        std::vector<float> v(8, 0.5f);
        c1.put("hello", v);
        auto got = c1.get("hello");
        ASSERT_TRUE(got.has_value());
        EXPECT_EQ(got->size(), 8u);
    }
    {
        preprocessor::EmbeddingCache c2(db, "model-B");
        EXPECT_FALSE(c2.get("hello").has_value());
    }
    std::error_code ec; fs::remove(db, ec);
}

// ---------- DiffPatcher ----------
TEST(Effectiveness_DiffPatcher, DiffIsSmallerThanFullFile) {
    std::string original;
    for (int i = 0; i < 100; ++i)
        original += "int f_" + std::to_string(i) + "() { return " + std::to_string(i) + "; }\n";
    std::string diff =
        "--- a/x.cpp\n+++ b/x.cpp\n@@ -50,3 +50,3 @@\n"
        " int f_49() { return 49; }\n"
        "-int f_50() { return 50; }\n"
        "+int f_50() { return 5000; }\n"
        " int f_51() { return 51; }\n";

    preprocessor::DiffPatcher p;
    std::unordered_map<std::string, std::string> files{{"x.cpp", original}};
    auto r = p.apply(diff, files);
    ASSERT_TRUE(r.ok);
    ASSERT_FALSE(r.files.empty());
    EXPECT_TRUE(r.files[0].applied);
    EXPECT_LT(diff.size(), r.files[0].patched_content.size())
        << "diff transport should be smaller than whole-file";
}

// ---------- BM25Index ----------
TEST(Effectiveness_BM25, RetrievesCorrectDocInTop3) {
    preprocessor::BM25Index idx;
    idx.add(1, "configuration loader reads JSON files");
    idx.add(2, "vector store HNSW nearest neighbour search");
    idx.add(3, "token bucket rate limiter throttle");
    idx.add(4, "HMAC SHA-256 bearer authentication middleware");
    idx.add(5, "rolling summariser for chat history");

    auto h = idx.search("HNSW nearest neighbour", 3);
    ASSERT_FALSE(h.empty());
    EXPECT_EQ(h[0].id, 2u);

    auto h2 = idx.search("HMAC authentication", 3);
    ASSERT_FALSE(h2.empty());
    EXPECT_EQ(h2[0].id, 4u);
}

// ---------- ModelRouter ----------
TEST(Effectiveness_ModelRouter, PicksCheapForSmallEdits) {
    preprocessor::ModelRouter r;
    r.add_tier({"cheap", "u", "m1", "", 0});
    r.add_tier({"frontier", "u", "m2", "", 0});
    r.add_route({preprocessor::PromptBucket::CodeEdit, 0, 1000, "cheap"});
    r.add_route({preprocessor::PromptBucket::CodeGenerate, 4000, 0, "frontier"});

    auto a = r.route(preprocessor::PromptBucket::CodeEdit, 500);
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->name, "cheap");

    auto b = r.route(preprocessor::PromptBucket::CodeGenerate, 10000);
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->name, "frontier");

    EXPECT_EQ(r.route(preprocessor::PromptBucket::Freeform, 100), nullptr);
}

// ---------- AbHarness ----------
TEST(Effectiveness_AbHarness, IsSticky) {
    preprocessor::AbHarness ab;
    preprocessor::AbExperiment e{"exp", {{"x", 1}, {"y", 1}}};
    ab.define(e);
    for (int i = 0; i < 200; ++i) {
        auto key = "k-" + std::to_string(i);
        EXPECT_EQ(ab.assign("exp", key), ab.assign("exp", key));
    }
}

TEST(Effectiveness_AbHarness, RoughlyBalancesWeights) {
    preprocessor::AbHarness ab;
    preprocessor::AbExperiment e{"e2", {{"x", 1}, {"y", 1}}};
    ab.define(e);
    int x = 0, y = 0;
    for (int i = 0; i < 5000; ++i) {
        auto v = ab.assign("e2", "k-" + std::to_string(i));
        if (v == "x") ++x; else if (v == "y") ++y;
    }
    // Allow 10% slack around expected 50/50.
    EXPECT_NEAR(double(x) / 5000.0, 0.5, 0.05);
    EXPECT_NEAR(double(y) / 5000.0, 0.5, 0.05);
}

// ---------- AuthMiddleware ----------
TEST(Effectiveness_AuthMiddleware, AcceptsValidAndRejectsInvalid) {
    preprocessor::AuthMiddleware::Config cfg;
    cfg.hmac_secret = "topsecret";
    preprocessor::AuthMiddleware mw(cfg);
    const std::int64_t ts = 1700000000;
    const std::string body = "hello";
    auto sig = preprocessor::AuthMiddleware::sign(cfg.hmac_secret, ts, body);

    EXPECT_TRUE(mw.verify("", sig, std::to_string(ts), body, ts));
    EXPECT_FALSE(mw.verify("", std::string(sig.size(), '0'), std::to_string(ts), body, ts));
    // Stale timestamp.
    EXPECT_FALSE(mw.verify("", sig, std::to_string(ts), body, ts + 100000));
}

// ---------- RateLimiter ----------
TEST(Effectiveness_RateLimiter, BurstThenRefill) {
    preprocessor::RateLimiter rl({/*tps*/10.0, /*burst*/5.0});
    int p1 = 0;
    for (int i = 0; i < 20; ++i) if (rl.try_acquire("k", 100.0)) ++p1;
    EXPECT_EQ(p1, 5);
    int p2 = 0;
    // 1 second later -> should refill 10 tokens (capped at burst=5).
    for (int i = 0; i < 20; ++i) if (rl.try_acquire("k", 101.0)) ++p2;
    EXPECT_GT(p2, 0);
    EXPECT_LE(p2, 10);
}
