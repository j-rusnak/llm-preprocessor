#include "code_chunker.hpp"
#include "i_embedding_engine.hpp"
#include "repo_index.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace {

/// Deterministic mock embedding engine: hashes each input string to a few
/// dimensions so identical texts yield identical vectors. No model needed.
class MockEmbedder : public preprocessor::IEmbeddingEngine {
public:
    explicit MockEmbedder(std::size_t dim) : dim_(dim) {}

    std::vector<float> generate_embedding(const std::string& text) override {
        std::vector<float> v(dim_, 0.0f);
        std::size_t h = std::hash<std::string>{}(text);
        for (std::size_t i = 0; i < dim_; ++i) {
            v[i] = static_cast<float>(((h >> (i % 32)) & 0xFF) / 255.0);
        }
        // L2 normalize (cosine metric assumes normalized vectors).
        float n = 0.0f;
        for (float x : v) n += x * x;
        n = n > 0.0f ? std::sqrt(n) : 1.0f;
        for (auto& x : v) x /= n;
        return v;
    }
private:
    std::size_t dim_;
};

class TempDir {
public:
    TempDir() {
        path_ = fs::temp_directory_path() / ("repo_idx_" + std::to_string(::time(nullptr)) +
            "_" + std::to_string(counter_.fetch_add(1)));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    fs::path path() const { return path_; }
    void write(const std::string& rel, const std::string& contents) {
        fs::path p = path_ / rel;
        fs::create_directories(p.parent_path());
        std::ofstream(p) << contents;
    }
private:
    fs::path path_;
    static std::atomic<std::uint64_t> counter_;
};
std::atomic<std::uint64_t> TempDir::counter_{0};

} // namespace

TEST(RepoIndex, IndexAndSearch) {
    TempDir d;
    d.write("foo.cpp", R"(int compute_hash(int x) {
    return x * 31;
}
)");
    d.write("bar.cpp", R"(void render_screen() {
    draw_frame();
}
)");

    auto embedder = std::make_shared<MockEmbedder>(16);
    auto chunker  = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndexConfig cfg;
    cfg.embedding_dim = 16;
    cfg.watch_for_changes = false;
    preprocessor::RepoIndex idx(embedder, chunker, cfg);

    idx.index_path(d.path().string());
    EXPECT_GE(idx.file_count(), 1u);
    EXPECT_GE(idx.chunk_count(), 1u);

    auto hits = idx.search("compute_hash", 3);
    ASSERT_FALSE(hits.empty());
    // BM25 ensures the relevant chunk wins on identifier match.
    EXPECT_NE(hits[0].chunk.file_path.find("foo.cpp"), std::string::npos);
}

TEST(RepoIndex, ReindexFileReplacesChunks) {
    TempDir d;
    d.write("a.cpp", R"(void original_symbol() { return; }
)");
    auto embedder = std::make_shared<MockEmbedder>(16);
    auto chunker  = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndexConfig cfg;
    cfg.embedding_dim = 16;
    cfg.watch_for_changes = false;
    preprocessor::RepoIndex idx(embedder, chunker, cfg);
    idx.index_path(d.path().string());

    auto before = idx.search("original_symbol", 3);
    ASSERT_FALSE(before.empty());

    d.write("a.cpp", R"(void replacement_symbol() { return; }
)");
    idx.reindex_file((d.path() / "a.cpp").string());

    auto orig = idx.search("original_symbol", 3);
    bool any_old = false;
    for (auto& h : orig) {
        if (h.chunk.text.find("original_symbol") != std::string::npos) any_old = true;
    }
    EXPECT_FALSE(any_old);

    auto repl = idx.search("replacement_symbol", 3);
    ASSERT_FALSE(repl.empty());
    EXPECT_NE(repl[0].chunk.text.find("replacement_symbol"), std::string::npos);
}

TEST(RepoIndex, ForgetFileRemoves) {
    TempDir d;
    d.write("z.cpp", R"(void unique_thing() {}
)");
    auto embedder = std::make_shared<MockEmbedder>(16);
    auto chunker  = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndexConfig cfg;
    cfg.embedding_dim = 16;
    cfg.watch_for_changes = false;
    preprocessor::RepoIndex idx(embedder, chunker, cfg);
    idx.index_path(d.path().string());
    EXPECT_GE(idx.chunk_count(), 1u);

    idx.forget_file((d.path() / "z.cpp").string());
    EXPECT_EQ(idx.file_count(), 0u);
    EXPECT_EQ(idx.chunk_count(), 0u);
}

TEST(RepoIndex, ForgetFileKeepsDuplicateContentFromOtherFiles) {
    TempDir d;
    const std::string same = R"(void shared_symbol() {
    return;
}
)";
    d.write("a.cpp", same);
    d.write("b.cpp", same);

    auto embedder = std::make_shared<MockEmbedder>(16);
    auto chunker  = std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndexConfig cfg;
    cfg.embedding_dim = 16;
    cfg.watch_for_changes = false;
    preprocessor::RepoIndex idx(embedder, chunker, cfg);
    idx.index_path(d.path().string());

    EXPECT_EQ(idx.file_count(), 2u);
    ASSERT_EQ(idx.chunk_count(), 1u);

    idx.forget_file((d.path() / "a.cpp").string());

    EXPECT_EQ(idx.file_count(), 1u);
    EXPECT_EQ(idx.chunk_count(), 1u);

    auto hits = idx.search("shared_symbol", 3);
    ASSERT_FALSE(hits.empty());
    EXPECT_NE(hits[0].chunk.file_path.find("b.cpp"), std::string::npos);
}
