#include "code_chunker.hpp"

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

} // namespace preprocessor
