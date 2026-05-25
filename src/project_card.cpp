#include "project_card.hpp"

#include "code_chunker.hpp"
#include "repo_index.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace preprocessor {

namespace {

std::string extension_of(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return {};
    // Skip extensions that look like directory components.
    if (path.find_first_of("/\\", pos) != std::string::npos) return {};
    return path.substr(pos);
}

std::string read_readme(const std::string& repo_root, std::size_t max_chars) {
    if (max_chars == 0 || repo_root.empty()) return {};
    namespace fs = std::filesystem;
    fs::path root(repo_root);
    static const char* candidates[] = {"README.md", "README.MD", "README.rst", "README.txt", "README"};
    for (const char* name : candidates) {
        fs::path p = root / name;
        std::error_code ec;
        if (!fs::exists(p, ec) || !fs::is_regular_file(p, ec)) continue;
        std::ifstream f(p, std::ios::binary);
        if (!f) continue;
        std::string buf(max_chars, '\0');
        f.read(buf.data(), static_cast<std::streamsize>(max_chars));
        buf.resize(static_cast<std::size_t>(f.gcount()));
        return buf;
    }
    return {};
}

} // namespace

std::string ProjectCard::to_markdown() const {
    std::ostringstream out;
    out << "# Project Card\n";
    if (!root_path.empty()) {
        out << "Root: `" << root_path << "`\n";
    }
    out << "Files indexed: " << total_files
        << " ; chunks: " << total_chunks << "\n";

    if (!files_by_extension.empty()) {
        out << "\n## Languages / file types\n";
        for (const auto& kv : files_by_extension) {
            out << "- `" << (kv.first.empty() ? "(none)" : kv.first)
                << "`: " << kv.second << "\n";
        }
    }
    if (!top_symbols.empty()) {
        out << "\n## Notable symbols\n";
        for (const auto& s : top_symbols) {
            out << "- `" << s << "`\n";
        }
    }
    if (!readme_excerpt.empty()) {
        out << "\n## README excerpt\n";
        out << readme_excerpt;
        if (readme_excerpt.back() != '\n') out << "\n";
    }
    return out.str();
}

nlohmann::json ProjectCard::to_json() const {
    nlohmann::json j;
    j["root_path"] = root_path;
    j["total_files"] = total_files;
    j["total_chunks"] = total_chunks;
    j["files_by_extension"] = files_by_extension;
    j["top_symbols"] = top_symbols;
    j["readme_excerpt"] = readme_excerpt;
    return j;
}

ProjectCard ProjectCardBuilder::build(const RepoIndex& index,
                                      const std::string& repo_root,
                                      std::size_t max_symbols,
                                      std::size_t readme_chars) {
    ProjectCard card;
    card.root_path = repo_root;
    const auto chunks = index.snapshot_chunks();
    card.total_chunks = chunks.size();

    std::unordered_set<std::string> seen_files;
    std::unordered_set<std::string> seen_symbols;
    std::vector<std::string> symbol_order;

    for (const auto& c : chunks) {
        if (seen_files.insert(c.file_path).second) {
            ++card.files_by_extension[extension_of(c.file_path)];
        }
        if (!c.symbol.empty() && seen_symbols.size() < max_symbols &&
            seen_symbols.insert(c.symbol).second) {
            symbol_order.push_back(c.symbol);
        }
    }
    card.total_files = seen_files.size();
    card.top_symbols = std::move(symbol_order);
    card.readme_excerpt = read_readme(repo_root, readme_chars);
    return card;
}

} // namespace preprocessor
