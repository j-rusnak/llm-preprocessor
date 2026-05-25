#include "code_chunker.hpp"
#include "i_embedding_engine.hpp"
#include "mcp_server.hpp"
#include "project_card.hpp"
#include "repo_index.hpp"
#include "structural_query_engine.hpp"
#include "symbol_graph.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

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
               ("llm_pp_mcp_" + std::to_string(
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
    std::unique_ptr<preprocessor::StructuralQueryEngine> engine;

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
            << "void util_fn(){ return; }\n";
        index.index_path(td.path.string());
        engine = std::make_unique<preprocessor::StructuralQueryEngine>(graph, index);
    }
};

json call(preprocessor::McpServer& s, const std::string& method,
          json params = json::object(), int id = 1) {
    json req = {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"method", method},
        {"params", std::move(params)}
    };
    std::string reply = s.handle_message(req.dump());
    return json::parse(reply);
}

} // namespace

TEST(McpServer, InitializeAdvertisesServerInfo) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    auto rep = call(server, "initialize");
    ASSERT_TRUE(rep.contains("result"));
    EXPECT_EQ(rep["result"]["serverInfo"]["name"], "llm-preprocessor");
    EXPECT_TRUE(rep["result"]["capabilities"].contains("tools"));
    EXPECT_TRUE(rep["result"]["capabilities"].contains("resources"));
}

TEST(McpServer, ToolsListExposesAllTools) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    auto rep = call(server, "tools/list");
    ASSERT_TRUE(rep["result"].contains("tools"));
    auto tools = rep["result"]["tools"];
    std::vector<std::string> names;
    for (const auto& t : tools) names.push_back(t["name"].get<std::string>());
    EXPECT_NE(std::find(names.begin(), names.end(), "search_repo"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "structural_query"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "get_chunk"), names.end());
}

TEST(McpServer, SearchRepoReturnsChunks) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    json params = {{"name", "search_repo"},
                   {"arguments", {{"query", "add"}, {"k", 3}}}};
    auto rep = call(server, "tools/call", params);
    ASSERT_TRUE(rep["result"].contains("content"));
    EXPECT_FALSE(rep["result"]["isError"].get<bool>());
    EXPECT_GE(rep["result"]["content"].size(), 1u);
    EXPECT_EQ(rep["result"]["content"][0]["type"], "text");
}

TEST(McpServer, StructuralQueryUsesEngine) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    server.set_symbol_graph(&f.graph);
    server.set_structural_engine(f.engine.get());
    json params = {{"name", "structural_query"},
                   {"arguments", {{"query", "where is `add`?"}}}};
    auto rep = call(server, "tools/call", params);
    ASSERT_TRUE(rep["result"].contains("content"));
    const std::string text = rep["result"]["content"][0]["text"].get<std::string>();
    EXPECT_NE(text.find("math.cpp"), std::string::npos);
}

TEST(McpServer, StructuralQueryWithoutEngineReturnsFallback) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    json params = {{"name", "structural_query"},
                   {"arguments", {{"query", "anything"}}}};
    auto rep = call(server, "tools/call", params);
    const std::string text = rep["result"]["content"][0]["text"].get<std::string>();
    EXPECT_NE(text.find("not configured"), std::string::npos);
}

TEST(McpServer, GetChunkLooksUpById) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    // Find some real chunk id via search.
    auto hits = f.index.search("add", 1);
    ASSERT_FALSE(hits.empty());
    std::uint64_t id = hits.front().chunk.id;

    json params = {{"name", "get_chunk"}, {"arguments", {{"id", id}}}};
    auto rep = call(server, "tools/call", params);
    EXPECT_FALSE(rep["result"]["isError"].get<bool>());
    const std::string text = rep["result"]["content"][0]["text"].get<std::string>();
    EXPECT_NE(text.find("chunk_id="), std::string::npos);
}

TEST(McpServer, GetChunkUnknownIdIsError) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    json params = {{"name", "get_chunk"}, {"arguments", {{"id", 999999999}}}};
    auto rep = call(server, "tools/call", params);
    EXPECT_TRUE(rep["result"]["isError"].get<bool>());
}

TEST(McpServer, ResourcesListIncludesCardAndStats) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    auto rep = call(server, "resources/list");
    auto res = rep["result"]["resources"];
    std::vector<std::string> uris;
    for (const auto& r : res) uris.push_back(r["uri"].get<std::string>());
    EXPECT_NE(std::find(uris.begin(), uris.end(), "repo://card"), uris.end());
    EXPECT_NE(std::find(uris.begin(), uris.end(), "repo://stats"), uris.end());
}

TEST(McpServer, ResourcesReadStatsReportsCounters) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    server.set_symbol_graph(&f.graph);
    json params = {{"uri", "repo://stats"}};
    auto rep = call(server, "resources/read", params);
    ASSERT_TRUE(rep["result"].contains("contents"));
    auto inner = json::parse(rep["result"]["contents"][0]["text"].get<std::string>());
    EXPECT_EQ(inner["file_count"], f.index.file_count());
    EXPECT_TRUE(inner["symbol_graph_attached"].get<bool>());
    EXPECT_TRUE(inner.contains("definition_count"));
}

TEST(McpServer, ResourcesReadCardServesMarkdown) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    auto card = preprocessor::ProjectCardBuilder::build(f.index, f.td.path.string());
    server.set_project_card(&card);
    auto rep = call(server, "resources/read", json{{"uri", "repo://card"}});
    EXPECT_EQ(rep["result"]["contents"][0]["mimeType"], "text/markdown");
    EXPECT_FALSE(rep["result"]["contents"][0]["text"].get<std::string>().empty());
}

TEST(McpServer, UnknownMethodReturnsMethodNotFound) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    auto rep = call(server, "nonexistent/method");
    ASSERT_TRUE(rep.contains("error"));
    EXPECT_EQ(rep["error"]["code"], -32601);
}

TEST(McpServer, NotificationsProduceNoResponse) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    json note = {{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}};
    std::string reply = server.handle_message(note.dump());
    EXPECT_TRUE(reply.empty());
}

TEST(McpServer, MalformedJsonReturnsParseError) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    std::string reply = server.handle_message("{not json");
    auto rep = json::parse(reply);
    ASSERT_TRUE(rep.contains("error"));
    EXPECT_EQ(rep["error"]["code"], -32700);
}

TEST(McpServer, ServeStdioRoundTripsMultipleMessages) {
    Fixture f;
    preprocessor::McpServer server(f.index);
    std::ostringstream out;
    std::istringstream in(
        R"({"jsonrpc":"2.0","id":1,"method":"initialize"})" "\n"
        R"({"jsonrpc":"2.0","id":2,"method":"tools/list"})" "\n");
    server.serve_stdio(in, out);
    std::string s = out.str();
    // Expect two JSON lines.
    EXPECT_NE(s.find("\"id\":1"), std::string::npos);
    EXPECT_NE(s.find("\"id\":2"), std::string::npos);
    EXPECT_NE(s.find("tools"), std::string::npos);
}
