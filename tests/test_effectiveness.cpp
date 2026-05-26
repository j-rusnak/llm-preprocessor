// Effectiveness regression suite.
//
// Locks in *minimum* effectiveness thresholds for every subsystem so that
// a regression in compression ratio, retrieval quality, or routing latency
// fails CI. These thresholds are deliberately conservative; the
// `effectiveness_runner` benchmark reports the actual numbers.

#include "ab_harness.hpp"
#include "auth_middleware.hpp"
#include "bm25_index.hpp"
#include "context_packer.hpp"
#include "diff_patcher.hpp"
#include "embedding_cache.hpp"
#include "code_chunker.hpp"
#include "graph_aware_retriever.hpp"
#include "i_embedding_engine.hpp"
#include "intent_classifier.hpp"
#include "llm_tokenizer.hpp"
#include "model_router.hpp"
#include "prompt_cache.hpp"
#include "prompt_rewriter.hpp"
#include "rate_limiter.hpp"
#include "repo_index.hpp"
#include "retrieval_query.hpp"
#include "streaming_compactor.hpp"
#include "sync_endpoint.hpp"
#include "symbol_graph.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
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

fs::path repo_path(const std::string& relative_path) {
    if (fs::exists(relative_path)) return fs::path(relative_path);
    auto from_build_dir = fs::path("..") / relative_path;
    if (fs::exists(from_build_dir)) return from_build_dir;
    return fs::path(relative_path);
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

class RetrievalEmbedder : public preprocessor::IEmbeddingEngine {
public:
    explicit RetrievalEmbedder(std::size_t dim = 16) : dim_(dim) {}
    std::vector<float> generate_embedding(const std::string& text) override {
        std::vector<float> v(dim_, 0.0f);
        std::size_t h = std::hash<std::string>{}(text);
        for (std::size_t i = 0; i < dim_; ++i) {
            v[i] = static_cast<float>(((h >> (i % 32)) & 0xff) / 255.0);
        }
        return v;
    }
private:
    std::size_t dim_;
};

preprocessor::CodeChunk retrieval_chunk(std::uint64_t id,
                                        std::string file,
                                        std::string text) {
    preprocessor::CodeChunk c;
    c.id = id;
    c.file_path = std::move(file);
    c.text = std::move(text);
    c.start_line = 1;
    c.end_line = 1;
    return c;
}

std::unique_ptr<preprocessor::RepoIndex> retrieval_index() {
    auto emb = std::make_shared<RetrievalEmbedder>(16);
    auto chunker = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndexConfig cfg;
    cfg.embedding_dim = 16;
    cfg.watch_for_changes = false;
    return std::make_unique<preprocessor::RepoIndex>(emb, chunker, cfg);
}

preprocessor::RetrievedChunk retrieved_chunk(std::uint64_t id,
                                             std::string file,
                                             std::string text,
                                             float score = 1.0f) {
    preprocessor::RetrievedChunk r;
    r.chunk = retrieval_chunk(id, std::move(file), std::move(text));
    r.score = score;
    return r;
}

bool has_id(const std::vector<std::uint64_t>& ids, std::uint64_t id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

void hydrate_retrieval_chunks(
    preprocessor::RepoIndex& index,
    const std::vector<preprocessor::CodeChunk>& chunks) {
    std::vector<preprocessor::SyncVectorEntry> entries;
    for (const auto& c : chunks) {
        preprocessor::SyncVectorEntry entry;
        entry.chunk_id = c.id;
        entry.source_path = c.file_path;
        entry.text = c.text;
        entry.start_line = c.start_line;
        entry.end_line = c.end_line;
        entry.vec.assign(16, 0.0f);
        entry.vec[static_cast<std::size_t>(c.id % 16)] = 1.0f;
        entries.push_back(std::move(entry));
    }
    index.apply_synced_vectors(entries);
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

// ---------- ContextPacker ----------
TEST(Effectiveness_ContextPacking, DedupesAndPreservesDiverseHighSignalChunks) {
    preprocessor::ContextPackerConfig cfg;
    cfg.include_header = false;
    cfg.max_context_chars = 220;

    const auto packed = preprocessor::pack_context(
        {
            retrieved_chunk(1,
                            "src/proxy/auth.cpp",
                            "verify HMAC X-Preprocessor-Authorization bearer token",
                            0.99f),
            retrieved_chunk(2,
                            "src/proxy/auth.cpp",
                            std::string(500, 'A'),
                            0.98f),
            retrieved_chunk(3,
                            "cmake/package.cmake",
                            "install package config onnxruntime redistributable",
                            0.70f),
            retrieved_chunk(4,
                            "docs/auth.md",
                            "verify HMAC X-Preprocessor-Authorization bearer token",
                            0.60f),
        },
        cfg);

    EXPECT_TRUE(has_id(packed.included_chunk_ids, 1));
    EXPECT_TRUE(has_id(packed.included_chunk_ids, 3));
    EXPECT_TRUE(has_id(packed.omitted_chunk_ids, 2));
    EXPECT_TRUE(has_id(packed.deduped_chunk_ids, 4));
    EXPECT_TRUE(packed.truncated);
    EXPECT_LE(packed.chars_used, cfg.max_context_chars);
    EXPECT_NE(packed.text.find("src/proxy/auth.cpp"), std::string::npos);
    EXPECT_NE(packed.text.find("cmake/package.cmake"), std::string::npos);
}

TEST(Effectiveness_ContextPacking, CacheKeyStableWhenOmittedChunksDiffer) {
    preprocessor::ContextPackerConfig cfg;
    cfg.include_header = false;
    cfg.max_context_chars = 72;

    const auto first = preprocessor::pack_context(
        {
            retrieved_chunk(10, "src/a.cpp", "int stable = 1;", 1.0f),
            retrieved_chunk(20, "src/b.cpp", std::string(500, 'B'), 0.5f),
        },
        cfg);
    const auto second = preprocessor::pack_context(
        {
            retrieved_chunk(10, "src/a.cpp", "int stable = 1;", 1.0f),
            retrieved_chunk(30, "src/c.cpp", std::string(500, 'C'), 0.5f),
        },
        cfg);

    ASSERT_EQ(first.included_chunk_ids, second.included_chunk_ids);
    ASSERT_TRUE(first.truncated);
    ASSERT_TRUE(second.truncated);

    const auto first_key =
        preprocessor::PromptCache::make_key("model", "compiled", first.included_chunk_ids);
    const auto second_key =
        preprocessor::PromptCache::make_key("model", "compiled", second.included_chunk_ids);
    EXPECT_EQ(first_key, second_key);
}

TEST(Effectiveness_Retrieval, FixtureQueriesHitExpectedLanguageFileTop3) {
    const auto root = repo_path("tests/fixtures/retrieval");
    ASSERT_TRUE(fs::exists(root)) << root.string();

    auto index = retrieval_index();
    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;
    index->attach_symbol_graph(&graph, &extractor);
    index->index_path(root.string());

    ASSERT_GE(index->file_count(), 22u);

    struct QueryCase {
        std::string query;
        std::string expected_path_fragment;
    };
    const std::vector<QueryCase> cases = {
        {
            "proxy stats auth failures stream cancellation upstream timeout",
            "cpp/openai_proxy_slice.cpp"
        },
        {
            "debounced search AbortController stale fetch results",
            "typescript/searchPanel.ts"
        },
        {
            "jsonl ingestion retry exponential backoff batch records",
            "python/ingest_pipeline.py"
        },
        {
            "production deployment loopback auth unsafe remote proxy request size",
            "docs/production.md"
        },
        {
            "cmake package config install target onnx runtime redistributable vcpkg",
            "cmake/CMakeLists.txt"
        },
        {
            "security/auth_middleware_slice.cpp verify_local_proxy_request reject_replay_window allowed_skew_seconds",
            "security/auth_middleware_slice.cpp"
        },
        {
            "go http retry transport context deadline exponential backoff round trip",
            "go/http_retry_transport.go"
        },
        {
            "rust workspace cache lru snapshot eviction pathbuf",
            "rust/workspace_cache.rs"
        },
        {
            "java servlet auth filter hmac preprocessor authorization header",
            "java/AuthFilter.java"
        },
        {
            "yaml kubernetes deployment readiness probe auth token memory limit",
            "yaml/kubernetes-deployment.yaml"
        },
        {
            "sql prompt cache entries embedding vectors request audit schema",
            "sql/schema.sql"
        },
        {
            "cpp context budget guard elides duplicate chunks by score",
            "cpp/context_budget_guard.cpp"
        },
        {
            "typescript buildContextGraphRows RetrievalDiagnostic topKPreview graphLift nearMiss",
            "typescript/contextGraphPanel.ts"
        },
        {
            "python baseline comparison ndjson snapshot regression delta",
            "python/baseline_compare.py"
        },
        {
            "go embedding cache warmer prefetches repository retrieval vectors",
            "go/cache_warmer.go"
        },
        {
            "rust upstream stream cancellation aborts sink on disconnect",
            "rust/stream_cancel.rs"
        },
        {
            "java/ModelRoutingPolicy.java chooseTier CodeGenerate requestChars frontier fallback",
            "java/ModelRoutingPolicy.java"
        },
        {
            "yaml prometheus alert retrieval accuracy stream cancellation",
            "yaml/observability-rules.yaml"
        },
        {
            "sql dashboard history retention baseline snapshots",
            "sql/retention_policy.sql"
        },
        {
            "cmake package smoke imported target onnx runtime install",
            "cmake/PackageSmoke.cmake"
        },
        {
            "security tenant hmac nonce replay preprocessor authorization",
            "security/tenant_auth_policy.cpp"
        },
        {
            "markdown retrieval debugging near miss expected rank diagnostics",
            "docs/retrieval-debugging.md"
        },
    };

    for (const auto& c : cases) {
        const auto hits = index->search(c.query, 3);
        ASSERT_FALSE(hits.empty()) << c.query;
        bool found = false;
        std::string files;
        for (const auto& hit : hits) {
            std::string path = hit.chunk.file_path;
            std::replace(path.begin(), path.end(), '\\', '/');
            files += path + "\n";
            if (path.find(c.expected_path_fragment) != std::string::npos) {
                found = true;
            }
        }
        EXPECT_TRUE(found) << "query=" << c.query << "\nhits:\n" << files;
    }
}

TEST(Effectiveness_Retrieval, PathAndLanguageHintsImproveTop3) {
    preprocessor::BM25Index idx;
    idx.add(1, "TypeScript file handles fetch cancellation and stale results");
    idx.add(2,
            "ts scratch notes plus unrelated deployment archive package proxy "
            "vector cache graph symbol tokenizer compiler metrics sync");

    const auto raw_hits = idx.search("ts", 3);
    ASSERT_FALSE(raw_hits.empty());
    EXPECT_EQ(raw_hits[0].id, 2u);

    const auto parsed = preprocessor::parse_retrieval_query("ts");
    const auto normalized_hits =
        idx.search(preprocessor::build_lexical_query_text(parsed), 3);
    ASSERT_FALSE(normalized_hits.empty());
    EXPECT_EQ(normalized_hits[0].id, 1u);
}

// ---------- Retrieval / Graph Expansion ----------
TEST(Effectiveness_Retrieval, GraphExpansionImprovesTop3) {
    auto index = retrieval_index();
    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;

    auto controller = retrieval_chunk(
        100, "src/controller.cpp",
        "void handle_login(){ parse_body(); verify_signature(); }\n");
    auto route = retrieval_chunk(
        101, "src/routes.cpp",
        "void login_route(){ handle_login(); }\n");
    auto verifier = retrieval_chunk(
        102, "src/security.cpp",
        "void verify_signature(){ check_hmac(); }\n");

    hydrate_retrieval_chunks(*index, {controller, route, verifier});
    graph.update_chunk(controller, extractor.extract(controller));
    graph.update_chunk(route, extractor.extract(route));
    graph.update_chunk(verifier, extractor.extract(verifier));

    std::vector<preprocessor::RetrievedChunk> seeds{
        {controller, 1.0f},
        {route, 0.9f}
    };
    auto has_security = [](const std::vector<preprocessor::RetrievedChunk>& hits,
                           std::size_t top_n) {
        const std::size_t limit = (std::min)(top_n, hits.size());
        for (std::size_t i = 0; i < limit; ++i) {
            if (hits[i].chunk.file_path.find("security.cpp") != std::string::npos) {
                return true;
            }
        }
        return false;
    };
    ASSERT_FALSE(has_security(seeds, 3));

    preprocessor::GraphExpansionConfig cfg;
    cfg.max_expanded = 1;
    cfg.query_text = "signature verification";
    auto expanded = preprocessor::expand_with_graph(seeds, graph, *index, cfg);

    EXPECT_TRUE(has_security(expanded, 3));
}

TEST(Effectiveness_Retrieval, GraphExpansionDoesNotPolluteUnrelatedTopK) {
    auto index = retrieval_index();
    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;

    auto controller = retrieval_chunk(
        110, "src/controller.cpp",
        "void handle_profile(){ parse_profile(); }\n");
    auto route = retrieval_chunk(
        111, "src/routes.cpp",
        "void profile_route(){ handle_profile(); }\n");
    auto helper = retrieval_chunk(
        112, "src/profile.cpp",
        "void parse_profile(){}\n");
    auto unrelated = retrieval_chunk(
        113, "src/payments.cpp",
        "void charge_credit_card(){}\n");

    hydrate_retrieval_chunks(*index, {controller, route, helper, unrelated});
    graph.update_chunk(controller, extractor.extract(controller));
    graph.update_chunk(route, extractor.extract(route));
    graph.update_chunk(helper, extractor.extract(helper));
    graph.update_chunk(unrelated, extractor.extract(unrelated));

    std::vector<preprocessor::RetrievedChunk> seeds{
        {controller, 1.0f},
        {route, 0.9f}
    };
    preprocessor::GraphExpansionConfig cfg;
    cfg.max_expanded = 2;
    cfg.query_text = "billing payment";
    auto expanded = preprocessor::expand_with_graph(seeds, graph, *index, cfg);

    ASSERT_GE(expanded.size(), seeds.size());
    EXPECT_EQ(expanded[0].chunk.id, controller.id);
    EXPECT_EQ(expanded[1].chunk.id, route.id);
    for (const auto& hit : expanded) {
        EXPECT_EQ(hit.chunk.file_path.find("payments.cpp"), std::string::npos);
    }
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
