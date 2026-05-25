#include "retrieval_query.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace preprocessor {

namespace {

bool is_query_char(unsigned char c) {
    return std::isalnum(c) || c == '_' || c == '-' || c == '.' ||
           c == '/' || c == '\\' || c == '+';
}

bool is_separator(char c) {
    return c == '_' || c == '-' || c == '.' || c == '/' || c == '\\';
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return s;
}

std::string normalize_spaces(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool in_space = false;
    for (unsigned char c : text) {
        if (std::isspace(c)) {
            if (!out.empty()) in_space = true;
            continue;
        }
        if (in_space) {
            out.push_back(' ');
            in_space = false;
        }
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

void add_unique(std::vector<std::string>& values, std::string value) {
    if (value.empty()) return;
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(std::move(value));
    }
}

bool is_short_stop_word(const std::string& term) {
    static const char* const stop_words[] = {
        "an", "as", "at", "be", "by", "do", "if", "in", "is",
        "it", "me", "my", "no", "of", "on", "or", "so", "to",
        "up", "us", "we"
    };
    for (const char* stop_word : stop_words) {
        if (term == stop_word) return true;
    }
    return false;
}

void add_search_term(std::vector<std::string>& terms, std::string term) {
    if (term.size() < 2 || is_short_stop_word(term)) return;
    add_unique(terms, std::move(term));
}

std::string alnum_compact(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (unsigned char c : raw) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    return out;
}

std::string normalized_identifier(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (unsigned char c : raw) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        } else if (c == '_' || c == '-') {
            out.push_back('_');
        }
    }
    while (!out.empty() && out.front() == '_') out.erase(out.begin());
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out;
}

bool has_camel_boundary(const std::string& raw) {
    for (std::size_t i = 1; i < raw.size(); ++i) {
        const unsigned char prev = static_cast<unsigned char>(raw[i - 1]);
        const unsigned char cur = static_cast<unsigned char>(raw[i]);
        const unsigned char next =
            i + 1 < raw.size() ? static_cast<unsigned char>(raw[i + 1]) : 0;
        if (std::isupper(cur) &&
            (std::islower(prev) || std::isdigit(prev) ||
             (std::isupper(prev) && next != 0 && std::islower(next)))) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> split_identifier_parts(const std::string& raw) {
    std::vector<std::string> parts;
    std::string current;

    auto flush = [&]() {
        if (current.size() >= 2) {
            parts.push_back(std::move(current));
        }
        current.clear();
    };

    for (std::size_t i = 0; i < raw.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(raw[i]);
        if (!std::isalnum(c)) {
            flush();
            continue;
        }

        if (!current.empty() && std::isupper(c)) {
            const unsigned char prev = static_cast<unsigned char>(raw[i - 1]);
            const unsigned char next =
                i + 1 < raw.size() ? static_cast<unsigned char>(raw[i + 1]) : 0;
            if (std::islower(prev) || std::isdigit(prev) ||
                (std::isupper(prev) && next != 0 && std::islower(next))) {
                flush();
            }
        }

        current.push_back(static_cast<char>(std::tolower(c)));
    }
    flush();
    return parts;
}

std::string normalize_path(std::string raw) {
    std::replace(raw.begin(), raw.end(), '\\', '/');
    raw = lower(std::move(raw));
    while (!raw.empty() &&
           (raw.front() == '"' || raw.front() == '\'' || raw.front() == '(' ||
            raw.front() == '[' || raw.front() == '{')) {
        raw.erase(raw.begin());
    }
    while (!raw.empty() &&
           (raw.back() == '"' || raw.back() == '\'' || raw.back() == ')' ||
            raw.back() == ']' || raw.back() == '}' || raw.back() == ',' ||
            raw.back() == ';' || raw.back() == ':')) {
        raw.pop_back();
    }
    return raw;
}

bool has_known_extension(const std::string& path) {
    static const char* const extensions[] = {
        ".c",   ".cc",   ".cpp", ".cxx", ".h",   ".hh",  ".hpp",
        ".hxx", ".js",   ".jsx", ".ts",  ".tsx", ".py",  ".rs",
        ".go",  ".java", ".kt",  ".cs",  ".md",  ".mdx", ".json",
        ".jsonl", ".yaml", ".yml", ".toml", ".cmake", ".txt"
    };
    for (const char* ext : extensions) {
        const std::string suffix(ext);
        if (path.size() >= suffix.size() &&
            path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return true;
        }
    }
    return path == "cmakelists.txt";
}

bool looks_like_path(const std::string& raw) {
    const auto path = normalize_path(raw);
    return path.find('/') != std::string::npos || has_known_extension(path);
}

void add_language_from_token(std::vector<std::string>& languages,
                             const std::string& token) {
    if (token == "c++" || token == "cpp" || token == "cplusplus") {
        add_unique(languages, "cpp");
    } else if (token == "ts" || token == "typescript") {
        add_unique(languages, "typescript");
    } else if (token == "js" || token == "javascript") {
        add_unique(languages, "javascript");
    } else if (token == "py" || token == "python") {
        add_unique(languages, "python");
    } else if (token == "md" || token == "markdown") {
        add_unique(languages, "markdown");
    } else if (token == "json" || token == "jsonl") {
        add_unique(languages, "json");
    } else if (token == "yaml" || token == "yml") {
        add_unique(languages, "yaml");
    } else if (token == "toml" || token == "cmake" || token == "rust" ||
               token == "go" || token == "java" || token == "kotlin" ||
               token == "csharp") {
        add_unique(languages, token);
    }
}

void add_language_from_path(std::vector<std::string>& languages,
                            const std::string& path) {
    if (path == "cmakelists.txt" ||
        (path.size() >= 6 && path.compare(path.size() - 6, 6, ".cmake") == 0)) {
        add_unique(languages, "cmake");
    } else if (path.size() >= 4 &&
               (path.compare(path.size() - 4, 4, ".cpp") == 0 ||
                path.compare(path.size() - 4, 4, ".cxx") == 0 ||
                path.compare(path.size() - 4, 4, ".hpp") == 0 ||
                path.compare(path.size() - 4, 4, ".hxx") == 0)) {
        add_unique(languages, "cpp");
    } else if (path.size() >= 3 &&
               (path.compare(path.size() - 3, 3, ".cc") == 0 ||
                path.compare(path.size() - 3, 3, ".hh") == 0 ||
                path.compare(path.size() - 2, 2, ".h") == 0)) {
        add_unique(languages, "cpp");
    } else if (path.size() >= 3 &&
               path.compare(path.size() - 3, 3, ".ts") == 0) {
        add_unique(languages, "typescript");
    } else if (path.size() >= 4 &&
               path.compare(path.size() - 4, 4, ".tsx") == 0) {
        add_unique(languages, "typescript");
    } else if (path.size() >= 3 &&
               path.compare(path.size() - 3, 3, ".js") == 0) {
        add_unique(languages, "javascript");
    } else if (path.size() >= 4 &&
               path.compare(path.size() - 4, 4, ".jsx") == 0) {
        add_unique(languages, "javascript");
    } else if (path.size() >= 3 &&
               path.compare(path.size() - 3, 3, ".py") == 0) {
        add_unique(languages, "python");
    } else if ((path.size() >= 3 &&
                path.compare(path.size() - 3, 3, ".md") == 0) ||
               (path.size() >= 4 &&
                path.compare(path.size() - 4, 4, ".mdx") == 0)) {
        add_unique(languages, "markdown");
    } else if ((path.size() >= 5 &&
                path.compare(path.size() - 5, 5, ".json") == 0) ||
               (path.size() >= 6 &&
                path.compare(path.size() - 6, 6, ".jsonl") == 0)) {
        add_unique(languages, "json");
    } else if ((path.size() >= 5 &&
                path.compare(path.size() - 5, 5, ".yaml") == 0) ||
               (path.size() >= 4 &&
                path.compare(path.size() - 4, 4, ".yml") == 0)) {
        add_unique(languages, "yaml");
    } else if (path.size() >= 5 &&
               path.compare(path.size() - 5, 5, ".toml") == 0) {
        add_unique(languages, "toml");
    }
}

std::vector<std::string> raw_tokens(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;
    for (unsigned char c : text) {
        if (!is_query_char(c)) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
            continue;
        }
        current.push_back(static_cast<char>(c));
    }
    if (!current.empty()) tokens.push_back(std::move(current));
    return tokens;
}

bool has_identifier_separator(const std::string& raw) {
    return std::find_if(raw.begin(), raw.end(), [](char c) {
        return c == '_' || c == '-';
    }) != raw.end();
}

} // namespace

RetrievalQuery parse_retrieval_query(const std::string& text) {
    RetrievalQuery query;
    query.original_text = text;
    query.normalized_text = normalize_spaces(text);

    if (lower(text).find("c++") != std::string::npos) {
        add_unique(query.language_hints, "cpp");
    }

    for (const auto& raw : raw_tokens(text)) {
        const bool is_path = looks_like_path(raw);
        const std::string compact = alnum_compact(raw);
        const std::string normalized = normalized_identifier(raw);
        const auto parts = split_identifier_parts(raw);

        if (is_path) {
            const auto path = normalize_path(raw);
            add_unique(query.path_hints, path);
            add_language_from_path(query.language_hints, path);
        }

        for (const auto& part : parts) {
            add_search_term(query.terms, part);
            add_language_from_token(query.language_hints, part);
        }

        if (!is_path && compact.size() >= 2 && parts.size() > 1) {
            add_search_term(query.terms, compact);
        }

        add_language_from_token(query.language_hints, lower(raw));
        add_language_from_token(query.language_hints, compact);

        if (!is_path && !normalized.empty() &&
            (has_identifier_separator(raw) || has_camel_boundary(raw))) {
            add_unique(query.identifier_terms, normalized);
        }
    }

    return query;
}

} // namespace preprocessor
