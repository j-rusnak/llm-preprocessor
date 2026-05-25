#pragma once

#include "code_chunker.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace preprocessor {

/// Kind of a symbol definition or reference site.
enum class SymbolKind {
    Unknown,
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Variable,
    Macro
};

std::string to_string(SymbolKind kind);

/// One symbol definition discovered in a chunk.
struct SymbolDef {
    std::string name;
    SymbolKind kind = SymbolKind::Unknown;
    std::string file_path;
    std::size_t line = 0;        // 1-based
    std::uint64_t chunk_id = 0;
};

/// One identifier reference discovered in a chunk (a "call site"-ish edge).
struct SymbolRef {
    std::string name;
    std::string file_path;
    std::size_t line = 0;
    std::uint64_t chunk_id = 0;
};

/// Result of running an extractor over a single chunk.
struct ExtractedSymbols {
    std::vector<SymbolDef> defs;
    std::vector<SymbolRef> refs;
};

/// Abstract symbol extractor. Regex-based and tree-sitter-based
/// implementations both satisfy this interface; downstream code (graph
/// builder, retrievers) depends solely on it so a real AST backend can drop
/// in later without churn.
class ISymbolExtractor {
public:
    virtual ~ISymbolExtractor() = default;
    virtual ExtractedSymbols extract(const CodeChunk& chunk) const = 0;
    virtual std::string name() const = 0;
};

/// Lightweight regex/text-scanner extractor that covers the most common
/// curly-brace languages (C, C++, Java, JS/TS, Rust, Go) plus Python.
///
/// It is intentionally coarse: false positives on `references` are
/// acceptable because the graph is used to *expand* a retrieval result set,
/// not to compile code. False negatives are also acceptable — the hybrid
/// retriever already finds the relevant chunks; the graph is purely
/// augmentation.
class RegexSymbolExtractor : public ISymbolExtractor {
public:
    ExtractedSymbols extract(const CodeChunk& chunk) const override;
    std::string name() const override { return "regex"; }
};

/// In-memory symbol graph backing graph-aware retrieval and the structural
/// fast-path. Stores:
///   - definitions indexed by simple (unqualified) name
///   - identifier references back-indexed by name
///   - a per-chunk adjacency list (chunk -> chunks containing definitions
///     of names referenced by this chunk)
///
/// Thread-safety: all mutators and queries are serialised by an internal
/// mutex. Queries return copies so callers can drop the lock immediately.
class SymbolGraph {
public:
    SymbolGraph();
    ~SymbolGraph();

    SymbolGraph(const SymbolGraph&) = delete;
    SymbolGraph& operator=(const SymbolGraph&) = delete;

    /// Replace any previous contributions from `chunk` with the freshly
    /// extracted symbols.
    void update_chunk(const CodeChunk& chunk, const ExtractedSymbols& extracted);

    /// Drop every fact contributed by `chunk_id`.
    void remove_chunk(std::uint64_t chunk_id);

    /// Drop every fact contributed by any chunk derived from `file_path`.
    void remove_file(const std::string& file_path);

    /// Definitions of `name` (case-sensitive). Empty if unknown.
    std::vector<SymbolDef> find_definitions(const std::string& name) const;

    /// References (call sites) of `name`. Empty if unknown.
    std::vector<SymbolRef> find_references(const std::string& name) const;

    /// Chunks that contain a *definition* of any name referenced inside the
    /// chunks listed in `seed_ids`. Useful for "include the implementations
    /// of the helpers the retrieved code calls" expansion.
    /// `max_results` caps the total fan-out.
    std::vector<std::uint64_t>
    neighbors_of(const std::vector<std::uint64_t>& seed_ids,
                 std::size_t max_results = 8) const;

    /// All definitions defined inside `file_path`.
    std::vector<SymbolDef> defs_in_file(const std::string& file_path) const;

    /// Distinct file paths that reference `name`.
    std::vector<std::string> files_referencing(const std::string& name) const;

    std::size_t definition_count() const;
    std::size_t reference_count() const;

private:
    mutable std::mutex mu_;

    // name -> defs
    std::unordered_map<std::string, std::vector<SymbolDef>> defs_by_name_;
    // name -> refs
    std::unordered_map<std::string, std::vector<SymbolRef>> refs_by_name_;

    // file -> chunk_ids contributing facts
    std::unordered_map<std::string, std::unordered_set<std::uint64_t>>
        chunks_by_file_;
    // chunk_id -> defs it provided
    std::unordered_map<std::uint64_t, std::vector<SymbolDef>> defs_by_chunk_;
    // chunk_id -> refs it provided
    std::unordered_map<std::uint64_t, std::vector<SymbolRef>> refs_by_chunk_;

    std::size_t def_total_ = 0;
    std::size_t ref_total_ = 0;
};

} // namespace preprocessor
