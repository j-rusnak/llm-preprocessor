#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace preprocessor {

/// A single chunk of source code carved out of a file. `id` is a content
/// hash (xxhash64) of `text` and uniquely identifies the chunk regardless of
/// where it appears.
struct CodeChunk {
    std::uint64_t id;
    std::string file_path;
    std::string text;
    std::size_t start_line; // 1-based, inclusive
    std::size_t end_line;   // 1-based, inclusive
    std::string symbol;     // best-guess enclosing symbol name (may be empty)
};

/// Abstract chunker that turns a source file into retrievable chunks.
///
/// Phase 0 ships only the interface and a lightweight `LineWindowChunker`
/// fallback. Phase 1 will add an AST-aware tree-sitter implementation that
/// chunks per function/class. Downstream code (indexer, retriever) depends
/// solely on this interface so the upgrade is drop-in.
class IChunker {
public:
    virtual ~IChunker() = default;

    /// Chunk a single file. `file_path` is informational; `source` carries
    /// the actual bytes.
    virtual std::vector<CodeChunk> chunk(const std::string& file_path,
                                         const std::string& source) const = 0;

    virtual std::string name() const = 0;
};

/// Fixed-window chunker: splits source into overlapping `window_lines`-line
/// windows with `overlap_lines` of overlap between adjacent windows. Useful
/// as a baseline before the AST chunker lands, and as a fallback for files
/// in languages tree-sitter cannot parse.
class LineWindowChunker : public IChunker {
public:
    LineWindowChunker(std::size_t window_lines = 60,
                      std::size_t overlap_lines = 10);

    std::vector<CodeChunk> chunk(const std::string& file_path,
                                 const std::string& source) const override;

    std::string name() const override { return "line-window"; }

private:
    std::size_t window_lines_;
    std::size_t overlap_lines_;
};

} // namespace preprocessor
