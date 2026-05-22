#include "code_chunker.hpp"
#include "project_card.hpp"
#include "repo_index.hpp"
#include "i_embedding_engine.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <gtest/gtest.h>

using preprocessor::BraceAwareChunker;
using preprocessor::IEmbeddingEngine;
using preprocessor::ProjectCard;
using preprocessor::ProjectCardBuilder;
using preprocessor::RepoIndex;
using preprocessor::RepoIndexConfig;

namespace {

// Deterministic hash-based fake embedder — same pattern as smoke_runner.
class FakeEmbedder : public IEmbeddingEngine {
public:
    explicit FakeEmbedder(std::size_t dim = 16) : dim_(dim) {}
    std::size_t embedding_dim() const { return dim_; }
    std::vector<float> generate_embedding(const std::string& text) override {
        std::vector<float> v(dim_, 0.f);
        std::hash<std::string> h;
        std::size_t s = h(text);
        for (std::size_t i = 0; i < dim_; ++i) {
            v[i] = static_cast<float>(((s >> (i % 32)) & 0xff) / 255.0);
        }
        return v;
    }
private:
    std::size_t dim_;
};

struct TempDir {
    std::filesystem::path path;
    TempDir() {
        path = std::filesystem::temp_directory_path() /
               ("pc_test_" + std::to_string(std::rand()));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

} // namespace

TEST(ProjectCard, EmptyIndexProducesEmptyCard) {
    auto emb = std::make_shared<FakeEmbedder>();
    auto chunker = std::make_shared<BraceAwareChunker>();
    RepoIndexConfig cfg;
    cfg.watch_for_changes = false;
    cfg.embedding_dim = emb->embedding_dim();
    RepoIndex idx(emb, chunker, cfg);

    ProjectCard card = ProjectCardBuilder::build(idx, "");
    EXPECT_EQ(card.total_files, 0u);
    EXPECT_EQ(card.total_chunks, 0u);
    EXPECT_TRUE(card.top_symbols.empty());
}

TEST(ProjectCard, AggregatesExtensionsSymbolsAndReadme) {
    TempDir td;
    // Two source files + a README at the root.
    std::ofstream(td.path / "alpha.cpp")
        << "int compute_thing() { return 1; }\nint helper() { return 2; }\n";
    std::ofstream(td.path / "beta.cpp")
        << "void do_stuff() {\n  // body\n}\n";
    std::ofstream(td.path / "README.md") << "Hello readme world!\n";

    auto emb = std::make_shared<FakeEmbedder>();
    auto chunker = std::make_shared<BraceAwareChunker>(/*max=*/400, /*min=*/1);
    RepoIndexConfig cfg;
    cfg.watch_for_changes = false;
    cfg.embedding_dim = emb->embedding_dim();
    RepoIndex idx(emb, chunker, cfg);
    idx.index_path(td.path.string());

    auto card = ProjectCardBuilder::build(idx, td.path.string(), 30, 64);
    // .md is also indexed by default; check the .cpp count specifically.
    ASSERT_EQ(card.files_by_extension.count(".cpp"), 1u);
    EXPECT_EQ(card.files_by_extension[".cpp"], 2u);
    EXPECT_GE(card.total_files, 2u);
    EXPECT_GT(card.total_chunks, 0u);
    EXPECT_NE(card.readme_excerpt.find("Hello readme"), std::string::npos);
}

TEST(ProjectCard, MarkdownContainsKeyFields) {
    ProjectCard card;
    card.root_path = "/tmp/x";
    card.total_files = 3;
    card.total_chunks = 7;
    card.files_by_extension[".cpp"] = 2;
    card.top_symbols.push_back("foo");
    card.readme_excerpt = "the readme";
    auto md = card.to_markdown();
    EXPECT_NE(md.find("/tmp/x"), std::string::npos);
    EXPECT_NE(md.find(".cpp"), std::string::npos);
    EXPECT_NE(md.find("foo"), std::string::npos);
    EXPECT_NE(md.find("the readme"), std::string::npos);
}
