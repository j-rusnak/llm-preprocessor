#pragma once

#include <iosfwd>
#include <memory>
#include <string>

namespace preprocessor {

class RepoIndex;
class SymbolGraph;
class StructuralQueryEngine;
struct ProjectCard;

/// Knobs for `McpServer`. Defaults are safe for local stdio use.
struct McpServerConfig {
    std::string server_name = "llm-preprocessor";
    std::string server_version = "0.4.0";
    std::string protocol_version = "2024-11-05";
    /// Cap on chunks returned by the `search_repo` tool per call.
    std::size_t default_search_k = 6;
    /// Cap on characters per chunk surfaced by `search_repo` / `get_chunk`.
    std::size_t max_chunk_chars = 4000;
};

/// Minimal Model Context Protocol server: JSON-RPC 2.0 over newline-
/// delimited JSON on stdio. Exposes the RAG stack to MCP-aware clients
/// (Claude Desktop, VS Code MCP, Continue, etc.) as a set of tools and
/// read-only resources.
///
/// Surfaces (no LLM calls — these are local-only operations):
///   Tools:
///     - search_repo        : hybrid retrieval over the indexed repo
///     - structural_query   : zero-LLM structural fast-path
///     - get_chunk          : fetch a single chunk by id
///   Resources:
///     - repo://card        : project card markdown
///     - repo://stats       : indexing + graph counters as JSON
///
/// Phase 4: this is the canonical way for editors/agents to consume the
/// preprocessor without going through the OpenAI-compatible HTTP proxy.
/// The same binary serves both modes; pick at startup via `--mcp` / `--serve`.
class McpServer {
public:
    McpServer(RepoIndex& index, McpServerConfig cfg = {});
    ~McpServer();

    McpServer(const McpServer&) = delete;
    McpServer& operator=(const McpServer&) = delete;

    /// Optional Phase 3 wiring. Both pointers are borrowed; caller owns.
    void set_symbol_graph(const SymbolGraph* graph) noexcept;
    void set_structural_engine(const StructuralQueryEngine* engine) noexcept;
    void set_project_card(const ProjectCard* card) noexcept;

    /// Handle one JSON-RPC message. Returns the JSON response as a string,
    /// or an empty string if the message was a notification (no `id`).
    /// Never throws; on parse/dispatch errors it returns a JSON-RPC error
    /// envelope with the appropriate code.
    std::string handle_message(const std::string& line);

    /// Drive the stdio loop. Reads newline-delimited JSON from `in`, writes
    /// each non-empty response (followed by '\n') to `out`. Blocks until
    /// EOF on `in`. `out` is flushed after every write.
    void serve_stdio(std::istream& in, std::ostream& out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace preprocessor
