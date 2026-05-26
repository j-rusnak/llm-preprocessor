// Context packing guard used by the retrieval effectiveness corpus.
#include <algorithm>
#include <string>
#include <vector>

struct CandidateChunk {
    std::string file_path;
    std::string content_hash;
    double score = 0.0;
    bool duplicate = false;
};

std::vector<CandidateChunk> choose_context_budget(
    std::vector<CandidateChunk> candidates,
    const std::size_t max_chunks) {
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const CandidateChunk& left, const CandidateChunk& right) {
                         return left.score > right.score;
                     });

    std::vector<CandidateChunk> packed;
    for (const auto& chunk : candidates) {
        if (packed.size() >= max_chunks) break;
        if (chunk.duplicate) continue;
        packed.push_back(chunk);
    }
    return packed;
}
