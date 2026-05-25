#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace preprocessor {

/// One hunk inside a unified-diff file section.
struct DiffHunk {
    std::size_t old_start = 0;  // 1-based line in the original file
    std::size_t old_count = 0;
    std::size_t new_start = 0;
    std::size_t new_count = 0;
    /// Hunk body lines INCLUDING the leading marker (' ', '-', '+', '\\').
    std::vector<std::string> body;
};

/// One file section inside a unified diff.
struct DiffFile {
    std::string old_path;          // e.g. "a/src/foo.cpp" (or empty for /dev/null on create)
    std::string new_path;          // e.g. "b/src/foo.cpp" (or empty for /dev/null on delete)
    std::vector<DiffHunk> hunks;
};

/// Per-file outcome of `DiffPatcher::apply`.
struct DiffApplyOutcome {
    std::string path;              // resolved path (no a/ b/ prefix)
    bool applied = false;
    std::string error;             // empty when applied
    std::string original_content;  // pre-patch (when known)
    std::string patched_content;   // post-patch (when applied)
};

/// Aggregate outcome of `DiffPatcher::apply`.
struct DiffApplyResult {
    bool ok = false;
    std::vector<DiffApplyOutcome> files;
};

/// Permissive unified-diff parser and applier. Targets payloads produced by
/// upstream LLMs for the `CodeEdit` bucket so we can avoid sending entire
/// files back through the wire on large edits.
///
/// Phase 6: parses standard `diff --git` / `--- ` / `+++ ` headers and
/// `@@` hunks, validates context against current file content, and applies
/// in memory. The class is intentionally side-effect free; callers decide
/// whether to flush patched contents to disk.
class DiffPatcher {
public:
    DiffPatcher() = default;

    /// Parse a unified diff. Returns the file sections found. Malformed
    /// hunks are skipped silently (permissive mode); use `parse_strict` to
    /// throw on the first error.
    std::vector<DiffFile> parse(std::string_view diff) const;

    /// Same as `parse`, but throws `std::runtime_error` on the first
    /// structural error.
    std::vector<DiffFile> parse_strict(std::string_view diff) const;

    /// Apply a parsed diff against an in-memory file map. The map is keyed
    /// by path (no `a/` / `b/` prefix). Files referenced by the diff but
    /// missing from the map are treated as empty (new-file mode).
    DiffApplyResult apply(const std::vector<DiffFile>& files,
                          const std::unordered_map<std::string, std::string>& contents) const;

    /// Convenience: parse + apply in one call.
    DiffApplyResult apply(std::string_view diff,
                          const std::unordered_map<std::string, std::string>& contents) const;

    /// Strip a leading `a/` or `b/` prefix. Public for testing.
    static std::string strip_prefix(std::string_view path);
};

}  // namespace preprocessor
