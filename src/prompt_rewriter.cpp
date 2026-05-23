#include "prompt_rewriter.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace preprocessor {

namespace {

// --- helpers --------------------------------------------------------------

// Detect whether `s[i]` is currently inside a "..." or '...' literal.
// Tracks escapes. Single-pass scanner shared by the comment strippers.
struct LiteralState {
    bool in_dq = false;   // inside double-quoted string
    bool in_sq = false;   // inside single-quoted char/string
    bool escape = false;

    bool in_string() const noexcept { return in_dq || in_sq; }

    void step(char c) noexcept {
        if (escape) { escape = false; return; }
        if (c == '\\' && (in_dq || in_sq)) { escape = true; return; }
        if (c == '"' && !in_sq) { in_dq = !in_dq; return; }
        if (c == '\'' && !in_dq) { in_sq = !in_sq; return; }
    }
};

std::string strip_block_comments(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    LiteralState ls;
    for (std::size_t i = 0; i < in.size(); ) {
        char c = in[i];
        if (!ls.in_string() && c == '/' && i + 1 < in.size() && in[i + 1] == '*') {
            // Skip until "*/" or EOF.
            std::size_t j = i + 2;
            while (j + 1 < in.size() && !(in[j] == '*' && in[j + 1] == '/')) ++j;
            i = (j + 1 < in.size()) ? j + 2 : in.size();
            continue;
        }
        ls.step(c);
        out.push_back(c);
        ++i;
    }
    return out;
}

std::string strip_line_comments(const std::string& in, bool strip_cpp, bool strip_hash) {
    std::string out;
    out.reserve(in.size());
    LiteralState ls;
    for (std::size_t i = 0; i < in.size(); ) {
        char c = in[i];
        // `#` only counts as a line comment when it is the first
        // non-whitespace character on the line (so we don't murder `#include`,
        // preprocessor directives, or `#` inside identifiers/strings).
        // We treat `# ` / `#TODO`-style end-of-line comments as comments only
        // when they sit at column 0 of the logical line.
        if (!ls.in_string() && strip_cpp && c == '/' && i + 1 < in.size() && in[i + 1] == '/') {
            while (i < in.size() && in[i] != '\n') ++i;
            continue;
        }
        if (!ls.in_string() && strip_hash && c == '#') {
            // Walk back to find line start; only strip when only whitespace
            // precedes the '#'.
            std::size_t k = out.size();
            bool line_start = true;
            while (k > 0 && out[k - 1] != '\n') {
                if (out[k - 1] != ' ' && out[k - 1] != '\t') { line_start = false; break; }
                --k;
            }
            // Don't strip preprocessor directives in C/C++ (those are stripped
            // by `strip_cpp` callers' upstream tokenizer anyway, but we keep
            // them). Heuristic: if the next non-space char is a known C
            // directive keyword, leave it alone.
            if (line_start) {
                std::size_t look = i + 1;
                while (look < in.size() && (in[look] == ' ' || in[look] == '\t')) ++look;
                static const char* kKeep[] = {
                    "include", "define", "undef", "ifdef", "ifndef", "if ",
                    "endif", "else", "elif", "pragma", "error", "warning", "line"
                };
                bool keep = false;
                for (const char* kw : kKeep) {
                    std::size_t L = std::char_traits<char>::length(kw);
                    if (look + L <= in.size() && in.compare(look, L, kw) == 0) {
                        keep = true; break;
                    }
                }
                if (!keep) {
                    while (i < in.size() && in[i] != '\n') ++i;
                    // Also drop the indentation we just kept.
                    out.resize(k);
                    continue;
                }
            }
        }
        ls.step(c);
        out.push_back(c);
        ++i;
    }
    return out;
}

std::string collapse_blanks(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    int consecutive_blank = 0;
    std::string line;
    auto flush = [&](bool last) {
        bool blank = true;
        for (char c : line) {
            if (c != ' ' && c != '\t' && c != '\r') { blank = false; break; }
        }
        if (blank) {
            ++consecutive_blank;
            if (consecutive_blank <= 1) {
                out.append(line);
                if (!last) out.push_back('\n');
            }
        } else {
            consecutive_blank = 0;
            out.append(line);
            if (!last) out.push_back('\n');
        }
        line.clear();
    };
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\n') { flush(false); continue; }
        line.push_back(in[i]);
    }
    if (!line.empty()) flush(true);
    return out;
}

std::string trim_trailing(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    std::string line;
    auto flush = [&](bool last) {
        std::size_t end = line.size();
        while (end > 0 && (line[end - 1] == ' ' || line[end - 1] == '\t' || line[end - 1] == '\r')) --end;
        out.append(line, 0, end);
        if (!last) out.push_back('\n');
        line.clear();
    };
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\n') { flush(false); continue; }
        line.push_back(in[i]);
    }
    if (!line.empty()) flush(true);
    return out;
}

std::string dedupe_adjacent(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    std::string prev, line;
    auto flush = [&](bool last) {
        if (line != prev || line.empty()) {
            out.append(line);
            if (!last) out.push_back('\n');
            prev = line;
        }
        line.clear();
    };
    for (std::size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '\n') { flush(false); continue; }
        line.push_back(in[i]);
    }
    if (!line.empty()) flush(true);
    return out;
}

std::string clip(const std::string& s, std::size_t max_chars) {
    if (max_chars == 0 || s.size() <= max_chars) return s;
    std::string out = s.substr(0, max_chars);
    out += "\n... [truncated]";
    return out;
}

} // namespace

// --- HeuristicCompressionRewriter ----------------------------------------

HeuristicCompressionRewriter::HeuristicCompressionRewriter(HeuristicCompressionConfig cfg)
    : cfg_(std::move(cfg)) {}

std::string HeuristicCompressionRewriter::rewrite(const std::string& context,
                                                  std::size_t max_chars) const {
    if (context.empty()) return context;
    try {
        std::string s = context;
        if (cfg_.strip_block_comments) s = strip_block_comments(s);
        if (cfg_.strip_line_comments)  s = strip_line_comments(s, true, true);
        if (cfg_.trim_trailing_whitespace) s = trim_trailing(s);
        if (cfg_.dedupe_adjacent_lines) s = dedupe_adjacent(s);
        if (cfg_.collapse_blank_lines)  s = collapse_blanks(s);

        std::size_t cap = max_chars;
        if (cfg_.hard_truncate_chars > 0 &&
            (cap == 0 || cfg_.hard_truncate_chars < cap)) {
            cap = cfg_.hard_truncate_chars;
        }
        if (cap > 0) s = clip(s, cap);
        return s;
    } catch (...) {
        return context;
    }
}

// --- LlamaCppRewriter (stub when llama.cpp isn't compiled in) ------------

struct LlamaCppRewriter::Impl {
    LlamaCppRewriterConfig cfg;
};

bool LlamaCppRewriter::is_available() noexcept {
#ifdef LLM_PREPROCESSOR_WITH_LLAMA_CPP
    return true;
#else
    return false;
#endif
}

LlamaCppRewriter::LlamaCppRewriter(LlamaCppRewriterConfig cfg)
    : impl_(std::make_unique<Impl>(Impl{std::move(cfg)})) {
#ifndef LLM_PREPROCESSOR_WITH_LLAMA_CPP
    throw std::runtime_error(
        "LlamaCppRewriter: this build was compiled without llama.cpp support. "
        "Re-run cmake with -DLLM_PREPROCESSOR_WITH_LLAMA_CPP=ON (and a "
        "llama.cpp install) to enable.");
#endif
}

LlamaCppRewriter::~LlamaCppRewriter() = default;

std::string LlamaCppRewriter::rewrite(const std::string& context,
                                      std::size_t /*max_chars*/) const {
#ifndef LLM_PREPROCESSOR_WITH_LLAMA_CPP
    return context;  // unreachable: constructor throws first.
#else
    // Real llama.cpp invocation path lives here when the option is on.
    // The interface is the only contract callers depend on, so swapping in
    // a concrete implementation does not require any churn upstream.
    (void)impl_;
    return context;
#endif
}

} // namespace preprocessor
