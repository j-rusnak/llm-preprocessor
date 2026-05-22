#include "code_chunker.hpp"

#include <cctype>
#include <sstream>
#include <stdexcept>

#include <xxhash.h>

namespace preprocessor {

LineWindowChunker::LineWindowChunker(std::size_t window_lines,
                                     std::size_t overlap_lines)
    : window_lines_(window_lines), overlap_lines_(overlap_lines) {
    if (window_lines_ == 0) {
        throw std::invalid_argument("window_lines must be > 0");
    }
    if (overlap_lines_ >= window_lines_) {
        throw std::invalid_argument("overlap_lines must be < window_lines");
    }
}

std::vector<CodeChunk> LineWindowChunker::chunk(const std::string& file_path,
                                                const std::string& source) const {
    std::vector<std::string> lines;
    {
        std::istringstream iss(source);
        std::string line;
        while (std::getline(iss, line)) {
            lines.push_back(std::move(line));
        }
    }

    std::vector<CodeChunk> out;
    if (lines.empty()) {
        return out;
    }

    const std::size_t step = window_lines_ - overlap_lines_;
    for (std::size_t start = 0; start < lines.size(); start += step) {
        const std::size_t end = std::min(start + window_lines_, lines.size());

        std::string text;
        text.reserve(64 * (end - start));
        for (std::size_t i = start; i < end; ++i) {
            text += lines[i];
            text += '\n';
        }

        CodeChunk chunk;
        chunk.id = XXH64(text.data(), text.size(), /*seed*/ 0);
        chunk.file_path = file_path;
        chunk.text = std::move(text);
        chunk.start_line = start + 1;
        chunk.end_line = end;
        chunk.symbol = {};
        out.push_back(std::move(chunk));

        if (end == lines.size()) break;
    }
    return out;
}

namespace {

// Best-guess symbol name: last identifier in `signature` before the `{`.
std::string trailing_identifier(const std::string& signature) {
    if (signature.empty()) return {};
    auto is_id = [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; };

    std::ptrdiff_t end = static_cast<std::ptrdiff_t>(signature.size());
    while (end > 0 && !is_id(signature[end - 1])) --end;
    std::ptrdiff_t begin = end;
    while (begin > 0 && is_id(signature[begin - 1])) --begin;
    if (begin == end) return {};
    return signature.substr(static_cast<std::size_t>(begin),
                            static_cast<std::size_t>(end - begin));
}

// Emit one chunk covering [start_line, end_line] (1-based, inclusive).
void emit_chunk(std::vector<CodeChunk>& out,
                const std::string& file_path,
                const std::vector<std::string>& lines,
                std::size_t start_line,
                std::size_t end_line,
                const std::string& symbol) {
    if (start_line == 0 || end_line < start_line || end_line > lines.size()) return;
    std::string text;
    text.reserve(64 * (end_line - start_line + 1));
    for (std::size_t i = start_line - 1; i < end_line; ++i) {
        text += lines[i];
        text += '\n';
    }
    CodeChunk chunk;
    chunk.id = XXH64(text.data(), text.size(), /*seed*/ 0);
    chunk.file_path = file_path;
    chunk.text = std::move(text);
    chunk.start_line = start_line;
    chunk.end_line = end_line;
    chunk.symbol = symbol;
    out.push_back(std::move(chunk));
}

// Split an oversized range into ~max_chunk_lines windows (no overlap) so a
// 10k-line file at depth 0 cannot produce one giant chunk.
void emit_windowed(std::vector<CodeChunk>& out,
                   const std::string& file_path,
                   const std::vector<std::string>& lines,
                   std::size_t start_line,
                   std::size_t end_line,
                   std::size_t max_chunk_lines,
                   const std::string& symbol) {
    for (std::size_t s = start_line; s <= end_line; s += max_chunk_lines) {
        std::size_t e = std::min(s + max_chunk_lines - 1, end_line);
        emit_chunk(out, file_path, lines, s, e, symbol);
    }
}

} // namespace

BraceAwareChunker::BraceAwareChunker(std::size_t max_chunk_lines,
                                     std::size_t min_chunk_lines)
    : max_chunk_lines_(max_chunk_lines), min_chunk_lines_(min_chunk_lines) {
    if (max_chunk_lines_ == 0) {
        throw std::invalid_argument("max_chunk_lines must be > 0");
    }
    if (min_chunk_lines_ == 0) {
        throw std::invalid_argument("min_chunk_lines must be > 0");
    }
}

std::vector<CodeChunk> BraceAwareChunker::chunk(const std::string& file_path,
                                                const std::string& source) const {
    std::vector<std::string> lines;
    {
        std::istringstream iss(source);
        std::string line;
        while (std::getline(iss, line)) {
            lines.push_back(std::move(line));
        }
    }
    std::vector<CodeChunk> out;
    if (lines.empty()) return out;

    // Single pass over the source bytes tracking brace depth modulo
    // comments and string literals. We record top-level brace OPEN line and
    // CLOSE line and use that to slice the line list.
    int depth = 0;
    enum class Mode { Code, LineComment, BlockComment, StringDQ, StringSQ };
    Mode mode = Mode::Code;

    // Lines covered by the most recent top-level block: [top_open_line, top_close_line].
    // 0 means "no pending block".
    std::size_t pending_open_line = 0;
    std::size_t last_emit_end_line = 0; // 1-based, last line already emitted

    std::size_t cur_line = 1; // 1-based
    auto bump_line = [&]() { ++cur_line; };

    for (std::size_t i = 0; i < source.size(); ++i) {
        char c = source[i];
        char next = (i + 1 < source.size()) ? source[i + 1] : '\0';

        switch (mode) {
        case Mode::LineComment:
            if (c == '\n') { mode = Mode::Code; bump_line(); }
            break;
        case Mode::BlockComment:
            if (c == '*' && next == '/') { mode = Mode::Code; ++i; }
            else if (c == '\n') bump_line();
            break;
        case Mode::StringDQ:
            if (c == '\\' && next != '\0') { ++i; }
            else if (c == '"') mode = Mode::Code;
            else if (c == '\n') bump_line(); // unterminated string -> keep going
            break;
        case Mode::StringSQ:
            if (c == '\\' && next != '\0') { ++i; }
            else if (c == '\'') mode = Mode::Code;
            else if (c == '\n') bump_line();
            break;
        case Mode::Code:
            if (c == '/' && next == '/') { mode = Mode::LineComment; ++i; }
            else if (c == '/' && next == '*') { mode = Mode::BlockComment; ++i; }
            else if (c == '"') mode = Mode::StringDQ;
            else if (c == '\'') mode = Mode::StringSQ;
            else if (c == '{') {
                if (depth == 0) pending_open_line = cur_line;
                ++depth;
            } else if (c == '}') {
                if (depth > 0) --depth;
                if (depth == 0 && pending_open_line != 0) {
                    // Find the start of the signature: walk back from
                    // pending_open_line until we hit either a blank line or
                    // the previously-emitted boundary.
                    std::size_t sig_start = pending_open_line;
                    while (sig_start > last_emit_end_line + 1) {
                        const std::string& prev = lines[sig_start - 2];
                        bool blank = true;
                        for (char pc : prev) {
                            if (!std::isspace(static_cast<unsigned char>(pc))) { blank = false; break; }
                        }
                        if (blank) break;
                        --sig_start;
                    }

                    // Emit any preamble between previous emit and signature.
                    if (sig_start > last_emit_end_line + 1) {
                        std::size_t preamble_start = last_emit_end_line + 1;
                        std::size_t preamble_end = sig_start - 1;
                        if (preamble_end - preamble_start + 1 >= min_chunk_lines_) {
                            emit_windowed(out, file_path, lines, preamble_start,
                                          preamble_end, max_chunk_lines_, "<preamble>");
                        }
                    }

                    // Emit the block itself, windowed if oversized.
                    std::size_t block_end = cur_line;
                    std::string sig_line = lines[pending_open_line - 1];
                    // strip trailing `{...` for symbol guess
                    auto brace = sig_line.find('{');
                    if (brace != std::string::npos) sig_line.erase(brace);
                    // For `int add(int a, int b)` style, the function name is
                    // the identifier just before the first `(`. Strip the
                    // parameter list before extracting the trailing identifier.
                    auto paren = sig_line.find('(');
                    if (paren != std::string::npos) sig_line.erase(paren);
                    std::string sym = trailing_identifier(sig_line);

                    if (block_end - sig_start + 1 > max_chunk_lines_) {
                        emit_windowed(out, file_path, lines, sig_start, block_end,
                                      max_chunk_lines_, sym);
                    } else {
                        emit_chunk(out, file_path, lines, sig_start, block_end, sym);
                    }
                    last_emit_end_line = block_end;
                    pending_open_line = 0;
                }
            } else if (c == '\n') {
                bump_line();
            }
            break;
        }
    }

    // Trailing preamble (e.g. file with no top-level blocks, or trailing
    // top-level decls after the last block).
    if (last_emit_end_line < lines.size()) {
        std::size_t s = last_emit_end_line + 1;
        std::size_t e = lines.size();
        if (e - s + 1 >= min_chunk_lines_) {
            emit_windowed(out, file_path, lines, s, e, max_chunk_lines_, "<preamble>");
        } else if (out.empty()) {
            // Tiny file -- still emit so it's searchable.
            emit_chunk(out, file_path, lines, s, e, "<preamble>");
        }
    }
    return out;
}

} // namespace preprocessor
