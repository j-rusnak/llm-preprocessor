#include "context_packer.hpp"

#include "repo_index.hpp"

#include <sstream>

namespace preprocessor {

namespace {

std::string format_entry(const RetrievedChunk& rc) {
    std::ostringstream entry;
    entry << "\n--- " << rc.chunk.file_path
          << " [L" << rc.chunk.start_line << "-L" << rc.chunk.end_line << "]";
    if (!rc.chunk.symbol.empty()) entry << " " << rc.chunk.symbol;
    entry << " ---\n" << rc.chunk.text << "\n";
    return entry.str();
}

void append_omitted(PackedContext& packed,
                    const std::vector<RetrievedChunk>& chunks,
                    std::size_t from) {
    for (std::size_t i = from; i < chunks.size(); ++i) {
        packed.omitted_chunk_ids.push_back(chunks[i].chunk.id);
    }
    packed.truncated = from < chunks.size();
}

} // namespace

PackedContext pack_context(const std::vector<RetrievedChunk>& chunks,
                           ContextPackerConfig config) {
    PackedContext packed;
    if (chunks.empty()) return packed;

    if (config.include_header) {
        packed.text = "Retrieved code context (most relevant first):\n";
    }

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const std::string entry = format_entry(chunks[i]);
        if (config.max_context_chars > 0 &&
            packed.chars_used + entry.size() > config.max_context_chars) {
            append_omitted(packed, chunks, i);
            break;
        }

        packed.text += entry;
        packed.chars_used += entry.size();
        packed.included_chunk_ids.push_back(chunks[i].chunk.id);
    }

    return packed;
}

} // namespace preprocessor
