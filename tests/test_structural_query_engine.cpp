#include "code_chunker.hpp"
#include "i_embedding_engine.hpp"
#include "repo_index.hpp"
#include "structural_query_engine.hpp"
#include "symbol_graph.hpp"

#include <chrono>
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
               ("llm_pp_struct_" + std::to_string(
                    std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
};

struct Fixture {
    TempDir td;
    std::shared_ptr<preprocessor::IEmbeddingEngine> emb =
        std::make_shared<HashEmbedder>(16);
    std::shared_ptr<preprocessor::IChunker> chunker =
        std::make_shared<preprocessor::BraceAwareChunker>(400, 1);
    preprocessor::RepoIndex index;
    preprocessor::SymbolGraph graph;
    preprocessor::RegexSymbolExtractor extractor;

    Fixture()
        : index(emb, chunker, [] {
              preprocessor::RepoIndexConfig c;
              c.embedding_dim = 16;
              c.watch_for_changes = false;
              return c;
          }()) {
        index.attach_symbol_graph(&graph, &extractor);
        std::ofstream(td.path / "math.cpp")
            << "int add(int a,int b){ return a+b; }\n"
               "int caller(){ return add(1,2); }\n";
        std::ofstream(td.path / "util.cpp")
            << "void util_fn(){}\n";
        index.index_path(td.path.string());
    }
};

} // namespace

TEST(StructuralQueryEngine, AnswersDefinitionLookup) {
    Fixture f;
    preprocessor::StructuralQueryEngine eng(f.graph, f.index);
    auto ans = eng.try_answer("where is `add`?");
    ASSERT_TRUE(ans.has_value());
    EXPECT_NE(ans->find("math.cpp"), std::string::npos);
}

TEST(StructuralQueryEngine, AnswersCallersQuery) {
    Fixture f;
    preprocessor::StructuralQueryEngine eng(f.graph, f.index);
    auto ans = eng.try_answer("what calls `add`");
    ASSERT_TRUE(ans.has_value());
    EXPECT_NE(ans->find("math.cpp"), std::string::npos);
}

TEST(StructuralQueryEngine, AnswersFunctionsInFile) {
    Fixture f;
    preprocessor::StructuralQueryEngine eng(f.graph, f.index);
    auto ans = eng.try_answer("functions in math.cpp");
    ASSERT_TRUE(ans.has_value());
    EXPECT_NE(ans->find("add"), std::string::npos);
}

TEST(StructuralQueryEngine, AnswersRepoStats) {
    Fixture f;
    preprocessor::StructuralQueryEngine eng(f.graph, f.index);
    auto ans = eng.try_answer("how many chunks are in the repo?");
    ASSERT_TRUE(ans.has_value());
    EXPECT_NE(ans->find("chunk"), std::string::npos);
}

TEST(StructuralQueryEngine, ReturnsNulloptForFreeform) {
    Fixture f;
    preprocessor::StructuralQueryEngine eng(f.graph, f.index);
    auto ans = eng.try_answer("please write a haiku about pointers");
    EXPECT_FALSE(ans.has_value());
}
