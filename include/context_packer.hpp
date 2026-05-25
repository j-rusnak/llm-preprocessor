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
    /// Drop repeated chunk ids and exact repeated chunk bodies before packing.
    bool dedupe_chunks = true;
    /// Make sure a tight budget sees at least one high-ranked chunk per file
    /// before spending the remaining budget on additional chunks from a file.
    bool prefer_file_diversity = true;
    std::size_t min_chunks_per_distinct_file = 1;
};

struct PackedContext {
    std::string text;
    std::vector<std::uint64_t> included_chunk_ids;
    std::vector<std::uint64_t> omitted_chunk_ids;
    std::vector<std::uint64_t> deduped_chunk_ids;
    std::size_t chars_used = 0;
    bool truncated = false;
};

PackedContext pack_context(const std::vector<RetrievedChunk>& chunks,
                           ContextPackerConfig config = {});

} // namespace preprocessor
