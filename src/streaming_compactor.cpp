#include "streaming_compactor.hpp"

#include <algorithm>

namespace preprocessor {

namespace {

const std::string& summary_header() {
    static const std::string header = "Summary of earlier turns:\n";
    return header;
}

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
    if (out.size() < s.size()) out.append("...");
    return out;
}

std::size_t total_chars(const std::vector<ChatTurn>& v) {
    std::size_t n = 0;
    for (const auto& t : v) n += t.role.size() + t.content.size() + 4;
    return n;
}

std::string clip_to_budget(const std::string& s, std::size_t budget,
                           std::size_t preserve_prefix = 0) {
    if (budget == 0) return {};
    if (s.size() <= budget) return s;
    preserve_prefix = std::min(preserve_prefix, budget);
    static const std::string marker = "\n... [summary truncated]\n";
    if (budget <= preserve_prefix + marker.size()) return s.substr(0, budget);
    if (budget <= marker.size()) return s.substr(0, budget);
    return s.substr(0, budget - marker.size()) + marker;
}

}  // namespace

CompactionResult StreamingCompactor::compact(const std::vector<ChatTurn>& history) const {
    CompactionResult r;
    if (history.empty()) return r;

    const std::size_t keep = std::min(cfg_.keep_recent, history.size());
    std::size_t split = history.size() - keep;

    // Grow the kept window backwards until the budget is reached or we hit
    // the start of history. When older turns remain, reserve enough space for
    // the summary header so the rolled summary stays recognizable after
    // clipping.
    auto kept = std::vector<ChatTurn>(history.begin() + split, history.end());
    std::size_t kept_chars = total_chars(kept);
    while (split > 0 && kept_chars < cfg_.max_total_chars) {
        const auto& candidate = history[split - 1];
        const std::size_t candidate_chars =
            candidate.role.size() + candidate.content.size() + 4;
        const std::size_t projected_kept = kept_chars + candidate_chars;
        const bool older_turns_remain = split > 1;
        const std::size_t summary_reserve =
            older_turns_remain ? summary_header().size() : 0;
        if (projected_kept + summary_reserve > cfg_.max_total_chars) {
            break;
        }
        kept.insert(kept.begin(), candidate);
        kept_chars = projected_kept;
        --split;
    }
    r.kept = std::move(kept);

    if (split == 0) return r;

    std::string summary = summary_header();
    for (std::size_t i = 0; i < split; ++i) {
        summary += "- ";
        summary += history[i].role;
        summary += ": ";
        summary += first_excerpt(history[i].content, cfg_.summary_chars_per_turn);
        summary += '\n';
    }
    r.rolled = split;
    const std::size_t summary_budget =
        kept_chars < cfg_.max_total_chars ? cfg_.max_total_chars - kept_chars : 0;
    r.rolled_summary = clip_to_budget(summary, summary_budget, summary_header().size());
    return r;
}

}  // namespace preprocessor
