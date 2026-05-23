#include "mcp_server.hpp"

#include "code_chunker.hpp"
#include "project_card.hpp"
#include "repo_index.hpp"
#include "structural_query_engine.hpp"
#include "symbol_graph.hpp"

#include <nlohmann/json.hpp>

#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

using json = nlohmann::json;

namespace preprocessor {

namespace {

// --- JSON-RPC envelope helpers -------------------------------------------

json make_response(const json& id, json result) {
    json r;
    r["jsonrpc"] = "2.0";
    r["id"] = id;
    r["result"] = std::move(result);
    return r;
}

json make_error(const json& id, int code, const std::string& message) {
    json r;
    r["jsonrpc"] = "2.0";
    r["id"] = id;
    r["error"] = {{"code", code}, {"message", message}};
    return r;
}

std::string clip(const std::string& s, std::size_t max_chars) {
    if (s.size() <= max_chars) {
        return s;
    }
    std::string out = s.substr(0, max_chars);
    out += "\n... [truncated]";
    return out;
}

json text_content(const std::string& text) {
    return json{{"type", "text"}, {"text", text}};
}

std::string chunk_to_text(const CodeChunk& c, std::size_t max_chars) {
    std::ostringstream os;
    os << "// " << c.file_path << ":" << c.start_line << "-" << c.end_line
       << "  (chunk_id=" << c.id << ")\n"
       << clip(c.text, max_chars);
    return os.str();
}

} // namespace

// --- Impl ----------------------------------------------------------------

struct McpServer::Impl {
    RepoIndex& index;
    McpServerConfig cfg;
    const SymbolGraph* graph = nullptr;
    const StructuralQueryEngine* engine = nullptr;
    const ProjectCard* card = nullptr;

    Impl(RepoIndex& idx, McpServerConfig c)
        : index(idx), cfg(std::move(c)) {}

    json tools_descriptor() const {
        json tools = json::array();
        tools.push_back({
            {"name", "search_repo"},
            {"description",
             "Hybrid (BM25 + vector) retrieval over the indexed repository. "
             "Returns the top-k chunks most relevant to the query."},
            {"inputSchema", {
                {"type", "object"},
                {"properties", {
                    {"query", {{"type", "string"}, {"description", "Free-text query."}}},
                    {"k", {{"type", "integer"},
                           {"description", "Max number of chunks to return."},
                           {"minimum", 1}, {"maximum", 32}}}
                }},
                {"required", json::array({"query"})}
            }}
        });
        tools.push_back({
            {"name", "structural_query"},
            {"description",
             "Answer simple structural questions (definitions, callers, "
             "functions-in-file, repo stats) from the local symbol graph "
             "without calling any LLM."},
            {"inputSchema", {
                {"type", "object"},
                {"properties", {
                    {"query", {{"type", "string"}}}
                }},
                {"required", json::array({"query"})}
            }}
        });
        tools.push_back({
            {"name", "get_chunk"},
            {"description", "Fetch one chunk by its numeric id."},
            {"inputSchema", {
                {"type", "object"},
                {"properties", {
                    {"id", {{"type", "integer"}, {"minimum", 0}}}
                }},
                {"required", json::array({"id"})}
            }}
        });
        return tools;
    }

    json resources_descriptor() const {
        json resources = json::array();
        resources.push_back({
            {"uri", "repo://card"},
            {"name", "Project card"},
            {"description", "Compact project summary (extensions, top symbols, README excerpt)."},
            {"mimeType", "text/markdown"}
        });
        resources.push_back({
            {"uri", "repo://stats"},
            {"name", "Repo stats"},
            {"description", "Indexing + symbol graph counters."},
            {"mimeType", "application/json"}
        });
        return resources;
    }

    json call_tool(const std::string& name, const json& args) {
        if (name == "search_repo") {
            std::string query = args.value("query", "");
            if (query.empty()) {
                throw std::invalid_argument("search_repo: 'query' is required");
            }
            std::size_t k = cfg.default_search_k;
            if (args.contains("k") && args["k"].is_number_integer()) {
                long requested = args["k"].get<long>();
                if (requested > 0) {
                    k = static_cast<std::size_t>(requested);
                }
            }
            auto hits = index.search(query, k);
            json content = json::array();
            if (hits.empty()) {
                content.push_back(text_content("No matching chunks."));
            } else {
                for (const auto& h : hits) {
                    content.push_back(text_content(chunk_to_text(h.chunk, cfg.max_chunk_chars)));
                }
            }
            return json{{"content", content}, {"isError", false}};
        }
        if (name == "structural_query") {
            std::string query = args.value("query", "");
            if (query.empty()) {
                throw std::invalid_argument("structural_query: 'query' is required");
            }
            std::string answer = "Structural fast path is not configured.";
            if (engine) {
                auto maybe = engine->try_answer(query);
                answer = maybe.value_or("No structural answer available for this query.");
            }
            json content = json::array({text_content(answer)});
            return json{{"content", content}, {"isError", false}};
        }
        if (name == "get_chunk") {
            if (!args.contains("id") || !args["id"].is_number_integer()) {
                throw std::invalid_argument("get_chunk: 'id' must be an integer");
            }
            std::uint64_t id = args["id"].get<std::uint64_t>();
            CodeChunk c;
            if (!index.try_get_chunk(id, c)) {
                json content = json::array({text_content("Unknown chunk id: " + std::to_string(id))});
                return json{{"content", content}, {"isError", true}};
            }
            json content = json::array({text_content(chunk_to_text(c, cfg.max_chunk_chars))});
            return json{{"content", content}, {"isError", false}};
        }
        throw std::invalid_argument("Unknown tool: " + name);
    }

    json read_resource(const std::string& uri) {
        json contents = json::array();
        if (uri == "repo://card") {
            std::string md = card ? card->to_markdown()
                                  : std::string("# Project card\n\n(not configured)\n");
            contents.push_back({{"uri", uri}, {"mimeType", "text/markdown"}, {"text", md}});
            return json{{"contents", contents}};
        }
        if (uri == "repo://stats") {
            json stats = {
                {"chunk_count", index.chunk_count()},
                {"file_count", index.file_count()},
                {"symbol_graph_attached", graph != nullptr},
                {"structural_engine_attached", engine != nullptr}
            };
            if (graph) {
                stats["definition_count"] = graph->definition_count();
                stats["reference_count"] = graph->reference_count();
            }
            contents.push_back({{"uri", uri},
                                {"mimeType", "application/json"},
                                {"text", stats.dump()}});
            return json{{"contents", contents}};
        }
        throw std::invalid_argument("Unknown resource: " + uri);
    }

    json dispatch(const json& msg) {
        const std::string method = msg.value("method", "");
        const json id = msg.contains("id") ? msg["id"] : json(nullptr);
        const bool is_notification = !msg.contains("id");

        try {
            if (method == "initialize") {
                json result = {
                    {"protocolVersion", cfg.protocol_version},
                    {"capabilities", {
                        {"tools", json::object()},
                        {"resources", json::object()}
                    }},
                    {"serverInfo", {
                        {"name", cfg.server_name},
                        {"version", cfg.server_version}
                    }}
                };
                return make_response(id, std::move(result));
            }
            if (method == "ping") {
                return make_response(id, json::object());
            }
            if (method == "tools/list") {
                return make_response(id, json{{"tools", tools_descriptor()}});
            }
            if (method == "tools/call") {
                const json& params = msg.value("params", json::object());
                const std::string name = params.value("name", "");
                const json args = params.value("arguments", json::object());
                return make_response(id, call_tool(name, args));
            }
            if (method == "resources/list") {
                return make_response(id, json{{"resources", resources_descriptor()}});
            }
            if (method == "resources/read") {
                const json& params = msg.value("params", json::object());
                const std::string uri = params.value("uri", "");
                return make_response(id, read_resource(uri));
            }
            if (is_notification) {
                // Unknown notifications are silently dropped per JSON-RPC 2.0.
                return json();
            }
            return make_error(id, -32601, "Method not found: " + method);
        } catch (const std::invalid_argument& e) {
            if (is_notification) return json();
            return make_error(id, -32602, e.what());
        } catch (const std::exception& e) {
            if (is_notification) return json();
            return make_error(id, -32603, std::string("Internal error: ") + e.what());
        }
    }
};

// --- Public API ----------------------------------------------------------

McpServer::McpServer(RepoIndex& index, McpServerConfig cfg)
    : impl_(std::make_unique<Impl>(index, std::move(cfg))) {}

McpServer::~McpServer() = default;

void McpServer::set_symbol_graph(const SymbolGraph* graph) noexcept {
    impl_->graph = graph;
}
void McpServer::set_structural_engine(const StructuralQueryEngine* engine) noexcept {
    impl_->engine = engine;
}
void McpServer::set_project_card(const ProjectCard* card) noexcept {
    impl_->card = card;
}

std::string McpServer::handle_message(const std::string& line) {
    json msg;
    try {
        msg = json::parse(line);
    } catch (const std::exception& e) {
        return make_error(nullptr, -32700,
                          std::string("Parse error: ") + e.what()).dump();
    }
    json reply = impl_->dispatch(msg);
    if (reply.is_null() || reply.empty()) {
        return std::string();
    }
    return reply.dump();
}

void McpServer::serve_stdio(std::istream& in, std::ostream& out) {
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::string reply = handle_message(line);
        if (!reply.empty()) {
            out << reply << '\n';
            out.flush();
        }
    }
}

} // namespace preprocessor
