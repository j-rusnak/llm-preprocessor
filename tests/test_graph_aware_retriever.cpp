#include "code_chunker.hpp"
#include "graph_aware_retriever.hpp"
#include "i_embedding_engine.hpp"
#include "repo_index.hpp"
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
