#include "structural_query_engine.hpp"

#include "repo_index.hpp"
#include "symbol_graph.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <unordered_set>

namespace preprocessor {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

// Extract the most likely identifier the user is asking about. Looks for
// `backticked` first, then a CamelCase / snake_case token in the message.
std::string extract_identifier(const std::string& msg) {
    std::smatch m;
    static const std::regex kBacktick(R"(`([A-Za-z_][A-Za-z0-9_]*)`)");
    if (std::regex_search(msg, m, kBacktick)) return m.str(1);
    static const std::regex kCamel(R"(\b([A-Z][A-Za-z0-9_]{2,}|[a-z]+_[A-Za-z0-9_]+)\b)");
    if (std::regex_search(msg, m, kCamel)) return m.str(1);
    // Fallback: last quoted token "..."
    static const std::regex kDq(R"DQ("([A-Za-z_][A-Za-z0-9_]*)")DQ");
    if (std::regex_search(msg, m, kDq)) return m.str(1);
    return {};
}

// Format up to `max` results to keep responses short.
template <typename T, typename Fn>
std::string format_list(const std::vector<T>& items, std::size_t max, Fn fmt) {
    std::ostringstream out;
    std::size_t shown = std::min(items.size(), max);
    for (std::size_t i = 0; i < shown; ++i) {
        out << "- " << fmt(items[i]) << "\n";
    }
    if (items.size() > max) {
        out << "...(" << (items.size() - max) << " more)\n";
    }
    return out.str();
}

} // namespace

StructuralQueryEngine::StructuralQueryEngine(const SymbolGraph& graph,
                                             const RepoIndex& index)
    : graph_(graph), index_(index) {}

std::optional<std::string>
StructuralQueryEngine::try_answer(const std::string& user_message) const {
    const std::string m = lower(user_message);
    const std::string ident = extract_identifier(user_message);

    auto starts_with = [&](const char* needle) {
        return m.find(needle) != std::string::npos;
    };

    // ---- definition lookup --------------------------------------------------
    if (!ident.empty() &&
        (starts_with("where is") || starts_with("where's") ||
         starts_with("define") || starts_with("definition of") ||
         starts_with("which file defines"))) {
        auto defs = graph_.find_definitions(ident);
        if (defs.empty()) {
            return std::string{"No definition for `"} + ident +
                   "` was found in the indexed repository.";
        }
        std::ostringstream out;
        out << "`" << ident << "` is defined in " << defs.size()
            << " place(s):\n";
        out << format_list(defs, 8, [](const SymbolDef& d) {
            std::ostringstream e;
            e << d.file_path << ":" << d.line << "  (" << to_string(d.kind) << ")";
            return e.str();
        });
        return out.str();
    }

    // ---- callers / references -----------------------------------------------
    if (!ident.empty() &&
        (starts_with("what calls") || starts_with("who calls") ||
         starts_with("callers of") || starts_with("which files use") ||
         starts_with("what references") || starts_with("references to") ||
         starts_with("files that use"))) {
        auto refs = graph_.find_references(ident);
        if (refs.empty()) {
            return std::string{"No references to `"} + ident +
                   "` were found.";
        }
        std::unordered_set<std::string> files;
        for (const auto& r : refs) files.insert(r.file_path);
        std::ostringstream out;
        out << "`" << ident << "` is referenced " << refs.size()
            << " time(s) across " << files.size() << " file(s):\n";
        out << format_list(refs, 10, [](const SymbolRef& r) {
            std::ostringstream e;
            e << r.file_path << ":" << r.line;
            return e.str();
        });
        return out.str();
    }

    // ---- list functions in a file ------------------------------------------
    {
        static const std::regex kInFile(
            R"(([A-Za-z0-9_./\\-]+\.[A-Za-z]+))");
        std::smatch sm;
        if (std::regex_search(user_message, sm, kInFile) &&
            (starts_with("functions in") || starts_with("symbols in") ||
             starts_with("what's in") || starts_with("list functions") ||
             starts_with("contents of"))) {
            std::string file = sm.str(1);
            auto defs = graph_.defs_in_file(file);
            if (defs.empty()) {
                // try suffix-match across indexed files
                auto snap = index_.snapshot_chunks();
                for (const auto& c : snap) {
                    if (c.file_path.size() >= file.size() &&
                        c.file_path.compare(c.file_path.size() - file.size(),
                                            file.size(), file) == 0) {
                        defs = graph_.defs_in_file(c.file_path);
                        if (!defs.empty()) { file = c.file_path; break; }
                    }
                }
            }
            if (defs.empty()) {
                return std::string{"No indexed symbols were found in `"} +
                       file + "`.";
            }
            std::ostringstream out;
            out << "Symbols defined in `" << file << "` ("
                << defs.size() << "):\n";
            std::sort(defs.begin(), defs.end(),
                      [](const SymbolDef& a, const SymbolDef& b) {
                          return a.line < b.line;
                      });
            out << format_list(defs, 20, [](const SymbolDef& d) {
                std::ostringstream e;
                e << d.name << "  (" << to_string(d.kind) << ", L"
                  << d.line << ")";
                return e.str();
            });
            return out.str();
        }
    }

    // ---- repo-level totals --------------------------------------------------
    if (starts_with("how many files") ||
        starts_with("how many chunks") ||
        starts_with("repo stats") ||
        starts_with("repository stats")) {
        std::ostringstream out;
        out << "Repository index:\n"
            << "- files: " << index_.file_count() << "\n"
            << "- chunks: " << index_.chunk_count() << "\n"
            << "- symbol definitions: " << graph_.definition_count() << "\n"
            << "- symbol references: " << graph_.reference_count() << "\n";
        return out.str();
    }

    return std::nullopt;
}

} // namespace preprocessor
