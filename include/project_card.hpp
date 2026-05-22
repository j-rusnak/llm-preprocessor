#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace preprocessor {

class RepoIndex;

/// Compact, structured summary of a repository. Built once on startup (and
/// cheap to recompute) so the LLM can be grounded with project-wide context
/// without paying per-token re-reads of the directory tree.
///
/// Phase 2: serves as the `{{project_card}}` slot in prompt templates.
struct ProjectCard {
    std::string root_path;
    std::size_t total_files = 0;
    std::size_t total_chunks = 0;
    /// File-count grouped by extension (e.g. ".cpp" -> 42). Sorted lexically.
    std::map<std::string, std::size_t> files_by_extension;
    /// Best-effort list of distinctive enclosing symbols (functions/classes).
    std::vector<std::string> top_symbols;
    /// First ~`readme_chars` bytes of README.md, if present at the root.
    std::string readme_excerpt;

    /// Compact markdown rendering for direct injection into a system prompt.
    std::string to_markdown() const;
    /// Structured JSON rendering for templates and diagnostics.
    nlohmann::json to_json() const;
};

/// Builds a `ProjectCard` from a populated `RepoIndex` (chunk metadata) plus
/// optional filesystem reads for the README excerpt.
class ProjectCardBuilder {
public:
    /// `max_symbols` caps how many distinct enclosing-symbol names are listed.
    /// `readme_chars` caps the README excerpt size. Pass 0 to skip the README.
    static ProjectCard build(const RepoIndex& index,
                             const std::string& repo_root,
                             std::size_t max_symbols = 30,
                             std::size_t readme_chars = 1000);
};

} // namespace preprocessor
