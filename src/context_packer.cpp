#include "context_packer.hpp"

#include "repo_index.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

std::vector<RetrievedChunk> dedupe_chunks(const std::vector<RetrievedChunk>& chunks,
                                          PackedContext& packed,
                                          bool enabled) {
    if (!enabled) return chunks;

    std::vector<RetrievedChunk> deduped;
    deduped.reserve(chunks.size());
    std::unordered_set<std::uint64_t> seen_ids;
    std::unordered_set<std::string> seen_bodies;
    std::unordered_set<std::uint64_t> reported_ids;

    for (const RetrievedChunk& rc : chunks) {
        const bool repeated_id = !seen_ids.insert(rc.chunk.id).second;
        const bool repeated_body = !seen_bodies.insert(rc.chunk.text).second;
        if (repeated_id || repeated_body) {
            if (reported_ids.insert(rc.chunk.id).second) {
                packed.deduped_chunk_ids.push_back(rc.chunk.id);
            }
            continue;
        }
        deduped.push_back(rc);
    }

    return deduped;
}

std::vector<RetrievedChunk> rank_chunks(std::vector<RetrievedChunk> chunks) {
    std::stable_sort(chunks.begin(), chunks.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.score > rhs.score;
    });
    return chunks;
}

std::vector<RetrievedChunk> diversify_files(const std::vector<RetrievedChunk>& chunks,
                                            const ContextPackerConfig& config) {
    if (!config.prefer_file_diversity ||
        config.min_chunks_per_distinct_file == 0 ||
        chunks.size() < 2) {
        return chunks;
    }

    std::vector<RetrievedChunk> ordered;
    ordered.reserve(chunks.size());
    std::vector<bool> selected(chunks.size(), false);
    std::unordered_map<std::string, std::size_t> selected_per_file;

    for (std::size_t pass = 0; pass < config.min_chunks_per_distinct_file; ++pass) {
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            if (selected[i]) continue;
            const std::string& file = chunks[i].chunk.file_path;
            if (selected_per_file[file] > pass) continue;
            selected[i] = true;
            ++selected_per_file[file];
            ordered.push_back(chunks[i]);
        }
    }

    for (std::size_t i = 0; i < chunks.size(); ++i) {
        if (!selected[i]) ordered.push_back(chunks[i]);
    }

    return ordered;
}

} // namespace

PackedContext pack_context(const std::vector<RetrievedChunk>& chunks,
                           ContextPackerConfig config) {
    PackedContext packed;
    if (chunks.empty()) return packed;

    auto candidates = dedupe_chunks(chunks, packed, config.dedupe_chunks);
    candidates = rank_chunks(std::move(candidates));
    candidates = diversify_files(candidates, config);

    if (config.include_header) {
        packed.text = "Retrieved code context (most relevant first):\n";
    }

    for (const RetrievedChunk& candidate : candidates) {
        const std::string entry = format_entry(candidate);
        if (config.max_context_chars > 0 &&
            packed.chars_used + entry.size() > config.max_context_chars) {
            packed.omitted_chunk_ids.push_back(candidate.chunk.id);
            packed.truncated = true;
            continue;
        }

        packed.text += entry;
        packed.chars_used += entry.size();
        packed.included_chunk_ids.push_back(candidate.chunk.id);
    }

    return packed;
}

} // namespace preprocessor
