#include "diff_patcher.hpp"

#include <sstream>
#include <stdexcept>

namespace preprocessor {

namespace {

std::vector<std::string> split_lines(std::string_view s) {
    std::vector<std::string> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\n') {
            out.emplace_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    if (start < s.size()) {
        out.emplace_back(s.substr(start));
    } else if (!s.empty() && s.back() == '\n') {
        // Preserve trailing empty line so round-trips stay stable.
        out.emplace_back("");
    }
    return out;
}

std::string join_lines(const std::vector<std::string>& lines, bool trailing_newline) {
    std::string out;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        out += lines[i];
        if (i + 1 < lines.size()) out += '\n';
    }
    if (trailing_newline) out += '\n';
    return out;
}

bool parse_hunk_header(const std::string& line, DiffHunk& h) {
    // Expected: @@ -old_start[,old_count] +new_start[,new_count] @@ [section]
    if (line.size() < 8 || line[0] != '@' || line[1] != '@') return false;
    std::size_t i = 2;
    while (i < line.size() && line[i] == ' ') ++i;
    if (i >= line.size() || line[i] != '-') return false;
    ++i;
    auto read_pair = [&](std::size_t& start, std::size_t& count) {
        std::size_t n = 0;
        bool any = false;
        while (i < line.size() && line[i] >= '0' && line[i] <= '9') {
            n = n * 10 + (line[i] - '0'); ++i; any = true;
        }
        if (!any) return false;
        start = n;
        count = 1;
        if (i < line.size() && line[i] == ',') {
            ++i;
            n = 0; any = false;
            while (i < line.size() && line[i] >= '0' && line[i] <= '9') {
                n = n * 10 + (line[i] - '0'); ++i; any = true;
            }
            if (!any) return false;
            count = n;
        }
        return true;
    };
    if (!read_pair(h.old_start, h.old_count)) return false;
    while (i < line.size() && line[i] == ' ') ++i;
    if (i >= line.size() || line[i] != '+') return false;
    ++i;
    if (!read_pair(h.new_start, h.new_count)) return false;
    return true;
}

}  // namespace

std::string DiffPatcher::strip_prefix(std::string_view path) {
    if (path.size() >= 2 && (path[0] == 'a' || path[0] == 'b') && path[1] == '/') {
        return std::string(path.substr(2));
    }
    return std::string(path);
}

std::vector<DiffFile> DiffPatcher::parse(std::string_view diff) const {
    std::vector<DiffFile> out;
    auto lines = split_lines(diff);
    // Drop a trailing empty sentinel line introduced by split_lines when the
    // diff ends with '\n'; otherwise it would be interpreted as an empty
    // context line and mismatch the source.
    while (!lines.empty() && lines.back().empty()) {
        lines.pop_back();
    }
    DiffFile current;
    DiffHunk hunk;
    bool in_hunk = false;
    bool have_file = false;

    auto flush_hunk = [&]() {
        if (in_hunk) {
            current.hunks.push_back(std::move(hunk));
            hunk = DiffHunk{};
            in_hunk = false;
        }
    };
    auto flush_file = [&]() {
        flush_hunk();
        if (have_file && (!current.old_path.empty() || !current.new_path.empty())) {
            out.push_back(std::move(current));
        }
        current = DiffFile{};
        have_file = false;
    };

    for (std::size_t li = 0; li < lines.size(); ++li) {
        const auto& l = lines[li];
        if (l.rfind("diff --git ", 0) == 0) {
            flush_file();
            have_file = true;
            continue;
        }
        if (l.rfind("--- ", 0) == 0) {
            flush_hunk();
            current.old_path = strip_prefix(std::string_view(l).substr(4));
            if (current.old_path == "/dev/null") current.old_path.clear();
            have_file = true;
            continue;
        }
        if (l.rfind("+++ ", 0) == 0) {
            current.new_path = strip_prefix(std::string_view(l).substr(4));
            if (current.new_path == "/dev/null") current.new_path.clear();
            have_file = true;
            continue;
        }
        if (l.rfind("@@", 0) == 0) {
            flush_hunk();
            if (parse_hunk_header(l, hunk)) {
                in_hunk = true;
            }
            continue;
        }
        if (in_hunk) {
            if (l.empty()) {
                hunk.body.push_back(" ");
                continue;
            }
            char marker = l[0];
            if (marker == ' ' || marker == '+' || marker == '-' || marker == '\\') {
                hunk.body.push_back(l);
            } else {
                // Non-hunk content terminates the hunk.
                flush_hunk();
            }
        }
    }
    flush_file();
    return out;
}

std::vector<DiffFile> DiffPatcher::parse_strict(std::string_view diff) const {
    auto files = parse(diff);
    if (files.empty()) {
        throw std::runtime_error("DiffPatcher::parse_strict: no file sections found");
    }
    for (const auto& f : files) {
        if (f.hunks.empty()) {
            throw std::runtime_error("DiffPatcher::parse_strict: file '" +
                                     (f.new_path.empty() ? f.old_path : f.new_path) +
                                     "' has no hunks");
        }
    }
    return files;
}

DiffApplyResult DiffPatcher::apply(
    const std::vector<DiffFile>& files,
    const std::unordered_map<std::string, std::string>& contents) const {
    DiffApplyResult result;
    result.ok = true;

    for (const auto& f : files) {
        DiffApplyOutcome out;
        out.path = f.new_path.empty() ? f.old_path : f.new_path;

        auto it = contents.find(out.path);
        std::string original;
        bool had_trailing_newline = false;
        if (it != contents.end()) {
            original = it->second;
            had_trailing_newline = !original.empty() && original.back() == '\n';
        }
        out.original_content = original;
        auto src_lines = split_lines(original);
        if (had_trailing_newline && !src_lines.empty() && src_lines.back().empty()) {
            src_lines.pop_back();
        }

        std::vector<std::string> dst_lines;
        std::size_t src_idx = 0;  // 0-based cursor into src_lines
        bool failed = false;

        for (const auto& h : f.hunks) {
            std::size_t hunk_src = (h.old_start == 0) ? 0 : (h.old_start - 1);
            if (hunk_src < src_idx) {
                out.error = "hunk start moves backwards";
                failed = true; break;
            }
            // Copy unchanged lines up to this hunk's start.
            while (src_idx < hunk_src && src_idx < src_lines.size()) {
                dst_lines.push_back(src_lines[src_idx++]);
            }
            // Walk the hunk body.
            for (const auto& bl : h.body) {
                if (bl.empty()) continue;
                char marker = bl[0];
                std::string payload = bl.substr(1);
                if (marker == '\\') continue;  // "\ No newline at end of file"
                if (marker == ' ') {
                    if (src_idx >= src_lines.size() || src_lines[src_idx] != payload) {
                        out.error = "context mismatch at hunk line " +
                                    std::to_string(src_idx + 1);
                        failed = true; break;
                    }
                    dst_lines.push_back(payload);
                    ++src_idx;
                } else if (marker == '-') {
                    if (src_idx >= src_lines.size() || src_lines[src_idx] != payload) {
                        out.error = "delete mismatch at hunk line " +
                                    std::to_string(src_idx + 1);
                        failed = true; break;
                    }
                    ++src_idx;
                } else if (marker == '+') {
                    dst_lines.push_back(payload);
                }
            }
            if (failed) break;
        }
        if (!failed) {
            while (src_idx < src_lines.size()) {
                dst_lines.push_back(src_lines[src_idx++]);
            }
            out.patched_content = join_lines(dst_lines, had_trailing_newline);
            out.applied = true;
        } else {
            result.ok = false;
        }
        result.files.push_back(std::move(out));
    }
    return result;
}

DiffApplyResult DiffPatcher::apply(
    std::string_view diff,
    const std::unordered_map<std::string, std::string>& contents) const {
    return apply(parse(diff), contents);
}

}  // namespace preprocessor
