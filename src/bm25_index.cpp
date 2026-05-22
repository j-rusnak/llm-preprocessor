#include "bm25_index.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace preprocessor {

namespace {

bool is_id_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Identifier-aware tokenisation:
//   raw token = maximal run of [A-Za-z0-9_]
//   subtokens = split on _ and camelCase boundaries
// Both raw + subtokens are emitted so "getUserName" matches "getusername"
// (legacy) AND "user" AND "name".
std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> out;
    std::string raw;
    raw.reserve(32);
    auto flush = [&]() {
        if (raw.empty()) return;
        std::string lo = lower(raw);
        out.push_back(lo);

        // Sub-split on underscore + camelCase.
        std::string cur;
        cur.reserve(raw.size());
        auto push = [&]() {
            if (!cur.empty()) {
                std::string clo = lower(cur);
                if (clo != lo) out.push_back(clo);
                cur.clear();
            }
        };
        for (std::size_t i = 0; i < raw.size(); ++i) {
            char c = raw[i];
            if (c == '_') { push(); continue; }
            bool boundary = !cur.empty()
                && std::isupper(static_cast<unsigned char>(c))
                && std::islower(static_cast<unsigned char>(cur.back()));
            if (boundary) push();
            cur += c;
        }
        push();
        raw.clear();
    };

    for (char c : text) {
        if (is_id_char(c)) raw += c;
        else flush();
    }
    flush();
    return out;
}

} // namespace

BM25Index::BM25Index(float k1, float b) : k1_(k1), b_(b) {
    if (k1 < 0.0f) throw std::invalid_argument("BM25 k1 must be >= 0");
    if (b < 0.0f || b > 1.0f) throw std::invalid_argument("BM25 b must be in [0, 1]");
}

void BM25Index::add(std::uint64_t id, const std::string& text) {
    if (doc_lengths_.count(id)) remove(id);
    auto tokens = tokenize(text);
    if (tokens.empty()) {
        doc_lengths_[id] = 0;
        return;
    }

    // Aggregate term frequencies for this doc.
    std::unordered_map<std::string, std::uint32_t> tf;
    tf.reserve(tokens.size());
    for (auto& t : tokens) ++tf[t];

    for (auto& [term, count] : tf) {
        postings_[term].push_back({id, count});
    }
    doc_lengths_[id] = static_cast<std::uint32_t>(tokens.size());
    total_tokens_ += tokens.size();
}

void BM25Index::remove(std::uint64_t id) {
    auto it = doc_lengths_.find(id);
    if (it == doc_lengths_.end()) return;
    total_tokens_ -= it->second;
    doc_lengths_.erase(it);

    // Lazy-purge from postings on next search would be cheaper for large
    // indices; for Phase 1 we do an eager sweep — RepoIndex churn is low.
    for (auto pit = postings_.begin(); pit != postings_.end(); ) {
        auto& vec = pit->second;
        vec.erase(std::remove_if(vec.begin(), vec.end(),
                                 [id](const Posting& p) { return p.id == id; }),
                  vec.end());
        if (vec.empty()) pit = postings_.erase(pit);
        else ++pit;
    }
}

std::vector<BM25Hit> BM25Index::search(const std::string& query, std::size_t k) const {
    if (k == 0 || doc_lengths_.empty()) return {};

    auto qtokens = tokenize(query);
    if (qtokens.empty()) return {};

    const float N = static_cast<float>(doc_lengths_.size());
    const float avgdl = N > 0.0f ? static_cast<float>(total_tokens_) / N : 1.0f;

    std::unordered_map<std::uint64_t, float> scores;
    scores.reserve(qtokens.size() * 4);

    std::unordered_set<std::string> seen_q;
    seen_q.reserve(qtokens.size());

    for (const auto& term : qtokens) {
        if (!seen_q.insert(term).second) continue; // dedupe query terms
        auto it = postings_.find(term);
        if (it == postings_.end()) continue;
        const auto& plist = it->second;
        const float df = static_cast<float>(plist.size());
        // Okapi BM25 IDF, +1 inside the log to keep it >= 0.
        const float idf = std::log(1.0f + (N - df + 0.5f) / (df + 0.5f));

        for (const auto& p : plist) {
            const float dl = static_cast<float>(doc_lengths_.at(p.id));
            const float denom = p.tf + k1_ * (1.0f - b_ + b_ * dl / avgdl);
            scores[p.id] += idf * (p.tf * (k1_ + 1.0f)) / (denom > 0.0f ? denom : 1.0f);
        }
    }
    if (scores.empty()) return {};

    std::vector<BM25Hit> hits;
    hits.reserve(scores.size());
    for (auto& [id, s] : scores) hits.push_back({id, s});

    if (hits.size() > k) {
        std::partial_sort(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(k),
                          hits.end(),
                          [](const BM25Hit& a, const BM25Hit& b) { return a.score > b.score; });
        hits.resize(k);
    } else {
        std::sort(hits.begin(), hits.end(),
                  [](const BM25Hit& a, const BM25Hit& b) { return a.score > b.score; });
    }
    return hits;
}

} // namespace preprocessor
