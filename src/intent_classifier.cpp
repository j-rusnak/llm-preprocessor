#include "intent_classifier.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <string>

namespace preprocessor {

namespace {

std::string to_lower(const std::string& s) {
    std::string out;
    out.resize(s.size());
    std::transform(s.begin(), s.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

bool contains_word(const std::string& haystack_lower, const char* needle) {
    // Whole-word match: needle must be bounded by non-alphanumeric chars.
    const std::string n(needle);
    std::string::size_type pos = 0;
    while ((pos = haystack_lower.find(n, pos)) != std::string::npos) {
        const bool left_ok = (pos == 0) ||
            !std::isalnum(static_cast<unsigned char>(haystack_lower[pos - 1]));
        const std::size_t end = pos + n.size();
        const bool right_ok = (end >= haystack_lower.size()) ||
            !std::isalnum(static_cast<unsigned char>(haystack_lower[end]));
        if (left_ok && right_ok) return true;
        pos = end;
    }
    return false;
}

template <std::size_t N>
bool any_word(const std::string& s, const std::array<const char*, N>& kws) {
    for (const char* k : kws) {
        if (contains_word(s, k)) return true;
    }
    return false;
}

} // namespace

std::string to_string(PromptBucket bucket) {
    switch (bucket) {
        case PromptBucket::CodeEdit:     return "code_edit";
        case PromptBucket::CodeExplain:  return "code_explain";
        case PromptBucket::CodeGenerate: return "code_generate";
        case PromptBucket::MetaQuery:    return "meta_query";
        case PromptBucket::Freeform:     return "freeform";
    }
    return "freeform";
}

PromptBucket bucket_from_string(const std::string& name) {
    if (name == "code_edit")     return PromptBucket::CodeEdit;
    if (name == "code_explain")  return PromptBucket::CodeExplain;
    if (name == "code_generate") return PromptBucket::CodeGenerate;
    if (name == "meta_query")    return PromptBucket::MetaQuery;
    return PromptBucket::Freeform;
}

PromptBucket HeuristicIntentClassifier::classify(const std::string& user_message) const {
    if (user_message.empty()) return PromptBucket::Freeform;
    const std::string s = to_lower(user_message);

    // Meta queries about the project itself - check first because they often
    // contain edit/explain keywords too.
    static constexpr std::array<const char*, 7> meta_kw{
        "which files", "what files", "how many files", "list files",
        "project structure", "directory", "repo"
    };
    for (const char* k : meta_kw) {
        if (s.find(k) != std::string::npos) return PromptBucket::MetaQuery;
    }

    // Edit: imperative mutation verbs.
    static constexpr std::array<const char*, 11> edit_kw{
        "edit", "modify", "rename", "refactor", "fix", "change",
        "update", "remove", "delete", "patch", "rewrite"
    };
    if (any_word(s, edit_kw)) return PromptBucket::CodeEdit;

    // Generate: net-new code requests.
    static constexpr std::array<const char*, 8> gen_kw{
        "write", "create", "implement", "generate", "build", "add",
        "scaffold", "stub"
    };
    if (any_word(s, gen_kw)) return PromptBucket::CodeGenerate;

    // Explain: questions about existing code.
    static constexpr std::array<const char*, 9> exp_kw{
        "explain", "why", "how does", "what does", "describe",
        "walk me through", "summarize", "summarise", "trace"
    };
    for (const char* k : exp_kw) {
        if (s.find(k) != std::string::npos) return PromptBucket::CodeExplain;
    }

    return PromptBucket::Freeform;
}

} // namespace preprocessor
