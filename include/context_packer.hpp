#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace preprocessor {

struct RetrievedChunk;

struct ContextPackerConfig {
    /// Cap applied to chunk entries only. 0 disables the cap.
    std::size_t max_context_chars = 8000;
    /// Include the legacy "Retrieved code context" heading.
    bool include_header = true;
};

struct PackedContext {
    std::string text;
    std::vector<std::uint64_t> included_chunk_ids;
    std::vector<std::uint64_t> omitted_chunk_ids;
    std::size_t chars_used = 0;
    bool truncated = false;
};

PackedContext pack_context(const std::vector<RetrievedChunk>& chunks,
                           ContextPackerConfig config = {});

} // namespace preprocessor
