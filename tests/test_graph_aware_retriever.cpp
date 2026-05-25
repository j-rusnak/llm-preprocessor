#include "code_chunker.hpp"
#include "graph_aware_retriever.hpp"
#include "i_embedding_engine.hpp"
#include "repo_index.hpp"
#include "sync_endpoint.hpp"
#include "symbol_graph.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>

namespace fs = std::filesystem;

namespace {

class HashEmbedder : public preprocessor::IEmbeddingEngine {
public:
    explicit HashEmbedder(std::size_t dim = 16) : dim_(dim) {}
    std::vector<float> generate_embedding(const std::string& text) override {
        std::vector<float> v(dim_, 0.f);
        std::size_t h = std::hash<std::string>{}(text);
        for (std::size_t i = 0; i < dim_; ++i) {
            v[i] = static_cast<float>(((h >> (i % 32)) & 0xff) / 255.0);
        }
        return v;
    }
private:
    std::size_t dim_;
};

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("llm_pp_graph_" + std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
};

preprocessor::CodeChunk chunk(std::uint64_t id,
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

std::unique_ptr<preprocessor::RepoIndex> make_index() {
    auto emb = std::make_shared<HashEmbedder>(16);
    auto chunker = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndexConfig cfg;
    cfg.embedding_dim = 16;
    cfg.watch_for_changes = false;
    return std::make_unique<preprocessor::RepoIndex>(emb, chunker, cfg);
}

void hydrate(preprocessor::RepoIndex& index,
             const std::vector<preprocessor::CodeChunk>& chunks) {
    std::vector<preprocessor::SyncVectorEntry> entries;
    entries.reserve(chunks.size());
    for (const auto& c : chunks) {
        preprocessor::SyncVectorEntry entry;
        entry.chunk_id = c.id;
        entry.source_path = c.file_path;
        entry.text = c.text;
        entry.start_line = c.start_line;
        entry.end_line = c.end_line;
        entry.symbol = c.symbol;
        entry.vec.assign(16, 0.0f);
        entry.vec[static_cast<std::size_t>(c.id % 16)] = 1.0f;
        entries.push_back(std::move(entry));
    }
    index.apply_synced_vectors(entries);
}

} // namespace

TEST(GraphAwareRetriever, AppendsNeighborsAfterSeeds) {
    TempDir td;
    std::ofstream(td.path / "helper.cpp")
        << "void shared_helper(){ /* impl */ }\n";
    std::ofstream(td.path / "caller.cpp")
        << "void caller(){ shared_helper(); }\n";

    auto emb = std::make_shared<HashEmbedder>(16);
    auto chunker = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndexConfig cfg;
    cfg.embedding_dim = 16;
    cfg.watch_for_changes = false;
    preprocessor::RepoIndex index(emb, chunker, cfg);

    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;
    index.attach_symbol_graph(&graph, &extractor);
    index.index_path(td.path.string());

    auto seeds = index.search("caller", 1);
    ASSERT_FALSE(seeds.empty());

    auto expanded = preprocessor::expand_with_graph(seeds, graph, index);
    EXPECT_GE(expanded.size(), seeds.size());
    bool found_helper = false;
    for (const auto& r : expanded) {
        if (r.chunk.file_path.find("helper.cpp") != std::string::npos)
            found_helper = true;
    }
    EXPECT_TRUE(found_helper);
}

TEST(GraphAwareRetriever, NoExpansionWhenSeedsEmpty) {
    preprocessor::SymbolGraph graph;
    auto emb = std::make_shared<HashEmbedder>(16);
    auto chunker = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndexConfig cfg;
    cfg.embedding_dim = 16;
    cfg.watch_for_changes = false;
    preprocessor::RepoIndex index(emb, chunker, cfg);

    auto out = preprocessor::expand_with_graph({}, graph, index);
    EXPECT_TRUE(out.empty());
}

TEST(GraphAwareRetriever, RespectsMaxExpansionCap) {
    TempDir td;
    std::ofstream(td.path / "a.cpp") << "void helper_a(){}\n";
    std::ofstream(td.path / "b.cpp") << "void helper_b(){}\n";
    std::ofstream(td.path / "c.cpp") << "void helper_c(){}\n";
    std::ofstream(td.path / "caller.cpp")
        << "void caller(){ helper_a(); helper_b(); helper_c(); }\n";

    auto emb = std::make_shared<HashEmbedder>(16);
    auto chunker = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndexConfig cfg;
    cfg.embedding_dim = 16;
    cfg.watch_for_changes = false;
    preprocessor::RepoIndex index(emb, chunker, cfg);

    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;
    index.attach_symbol_graph(&graph, &extractor);
    index.index_path(td.path.string());

    auto seeds = index.search("caller", 1);
    ASSERT_FALSE(seeds.empty());

    preprocessor::GraphExpansionConfig gc;
    gc.max_expanded = 1;
    auto expanded = preprocessor::expand_with_graph(seeds, graph, index, gc);
    EXPECT_LE(expanded.size(), seeds.size() + 1);
}

TEST(GraphAwareRetriever, QueryRelevantSymbolRanksFirst) {
    auto index = make_index();
    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;

    auto seed = chunk(1, "src/handler.cpp",
        "void handle_request(){ parse_body(); audit_log(); verify_signature(); }\n");
    auto parse = chunk(2, "src/body.cpp", "void parse_body(){}\n");
    auto audit = chunk(3, "src/audit.cpp", "void audit_log(){}\n");
    auto verify = chunk(4, "src/security.cpp", "void verify_signature(){}\n");

    hydrate(*index, {parse, audit, verify});
    graph.update_chunk(seed, extractor.extract(seed));
    graph.update_chunk(parse, extractor.extract(parse));
    graph.update_chunk(audit, extractor.extract(audit));
    graph.update_chunk(verify, extractor.extract(verify));

    preprocessor::GraphExpansionConfig cfg;
    cfg.max_expanded = 3;
    cfg.query_text = "signature verification";
    auto expanded = preprocessor::expand_with_graph({{seed, 1.0f}},
                                                    graph, *index, cfg);

    ASSERT_GE(expanded.size(), 2u);
    EXPECT_NE(expanded[1].chunk.text.find("verify_signature"),
              std::string::npos);
}

TEST(GraphAwareRetriever, CamelCaseQueryRanksSnakeCaseSymbol) {
    auto index = make_index();
    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;

    auto seed = chunk(30, "src/handler.cpp",
        "void handle_request(){ parse_body(); audit_log(); verify_signature(); }\n");
    auto parse = chunk(31, "src/body.cpp", "void parse_body(){}\n");
    auto audit = chunk(32, "src/audit.cpp", "void audit_log(){}\n");
    auto verify = chunk(33, "src/security.cpp", "void verify_signature(){}\n");

    hydrate(*index, {parse, audit, verify});
    graph.update_chunk(seed, extractor.extract(seed));
    graph.update_chunk(parse, extractor.extract(parse));
    graph.update_chunk(audit, extractor.extract(audit));
    graph.update_chunk(verify, extractor.extract(verify));

    preprocessor::GraphExpansionConfig cfg;
    cfg.max_expanded = 3;
    cfg.query_text = "signatureVerify";
    auto expanded = preprocessor::expand_with_graph({{seed, 1.0f}},
                                                    graph, *index, cfg);

    ASSERT_GE(expanded.size(), 2u);
    EXPECT_NE(expanded[1].chunk.text.find("verify_signature"),
              std::string::npos);
}

TEST(GraphAwareRetriever, SuppressesDuplicateDefinitionsForSameSymbol) {
    auto index = make_index();
    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;

    auto seed = chunk(10, "src/use.cpp", "void use(){ shared_helper(); }\n");
    auto prod = chunk(11, "src/shared.cpp", "void shared_helper(){}\n");
    auto test = chunk(12, "tests/shared_test.cpp", "void shared_helper(){}\n");

    hydrate(*index, {prod, test});
    graph.update_chunk(seed, extractor.extract(seed));
    graph.update_chunk(prod, extractor.extract(prod));
    graph.update_chunk(test, extractor.extract(test));

    preprocessor::GraphExpansionConfig cfg;
    cfg.max_expanded = 4;
    cfg.query_text = "shared helper implementation";
    auto expanded = preprocessor::expand_with_graph({{seed, 1.0f}},
                                                    graph, *index, cfg);

    std::size_t shared_defs = 0;
    for (std::size_t i = 1; i < expanded.size(); ++i) {
        if (expanded[i].chunk.text.find("shared_helper") != std::string::npos) {
            ++shared_defs;
        }
    }
    EXPECT_EQ(shared_defs, 1u);
    ASSERT_GE(expanded.size(), 2u);
    EXPECT_NE(expanded[1].chunk.file_path.find("src/shared.cpp"),
              std::string::npos);
}

TEST(GraphAwareRetriever, TieBreaksExpansionDeterministically) {
    auto index = make_index();
    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;

    auto seed = chunk(20, "src/use.cpp", "void use(){ zed(); alpha(); }\n");
    auto zed = chunk(21, "src/z.cpp", "void zed(){}\n");
    auto alpha = chunk(22, "src/a.cpp", "void alpha(){}\n");

    hydrate(*index, {zed, alpha});
    graph.update_chunk(seed, extractor.extract(seed));
    graph.update_chunk(zed, extractor.extract(zed));
    graph.update_chunk(alpha, extractor.extract(alpha));

    preprocessor::GraphExpansionConfig cfg;
    cfg.max_expanded = 2;
    auto expanded = preprocessor::expand_with_graph({{seed, 1.0f}},
                                                    graph, *index, cfg);

    ASSERT_EQ(expanded.size(), 3u);
    EXPECT_NE(expanded[1].chunk.file_path.find("src/a.cpp"),
              std::string::npos);
    EXPECT_NE(expanded[2].chunk.file_path.find("src/z.cpp"),
              std::string::npos);
}
