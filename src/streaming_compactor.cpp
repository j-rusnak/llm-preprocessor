#include "streaming_compactor.hpp"

#include <algorithm>

namespace preprocessor {

namespace {

std::string first_excerpt(const std::string& s, std::size_t cap) {
    std::string out;
    out.reserve(std::min(cap, s.size()));
    for (char c : s) {
        if (out.size() >= cap) break;
        if (c == '\n' || c == '\r') {
            if (!out.empty()) break;
            continue;
        }
        out.push_back(c);
    }
    while (!out.empty() && (out.back() == ' ' || out.back() == '\t')) out.pop_back();
    if (out.size() < s.size()) out.append("…");
    return out;
}

std::size_t total_chars(const std::vector<ChatTurn>& v) {
    std::size_t n = 0;
    for (const auto& t : v) n += t.role.size() + t.content.size() + 4;
    return n;
}

}  // namespace

CompactionResult StreamingCompactor::compact(const std::vector<ChatTurn>& history) const {
    CompactionResult r;
    if (history.empty()) return r;

    const std::size_t keep = std::min(cfg_.keep_recent, history.size());
    std::size_t split = history.size() - keep;

    // Grow the kept window backwards until the budget is reached or we hit
    // the start of history.
    auto kept = std::vector<ChatTurn>(history.begin() + split, history.end());
    while (split > 0 && total_chars(kept) < cfg_.max_total_chars) {
        const auto& candidate = history[split - 1];
        if (total_chars(kept) + candidate.role.size() + candidate.content.size() + 4 >
            cfg_.max_total_chars) {
            break;
        }
        kept.insert(kept.begin(), candidate);
        --split;
    }
    r.kept = std::move(kept);

    if (split == 0) return r;

    std::string summary = "Summary of earlier turns:\n";
    for (std::size_t i = 0; i < split; ++i) {
        summary += "- ";
        summary += history[i].role;
        summary += ": ";
        summary += first_excerpt(history[i].content, cfg_.summary_chars_per_turn);
        summary += '\n';
    }
    r.rolled = split;
    r.rolled_summary = std::move(summary);
    return r;
}

}  // namespace preprocessor
