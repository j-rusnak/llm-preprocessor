#include <gtest/gtest.h>
#include "code_chunker.hpp"

#include <string>
#include <unordered_set>

namespace {
std::string make_source(std::size_t n_lines) {
    std::string s;
    for (std::size_t i = 0; i < n_lines; ++i) {
        s += "line " + std::to_string(i) + "\n";
    }
    return s;
}
} // namespace

TEST(LineWindowChunkerTest, EmptySourceReturnsNoChunks) {
    preprocessor::LineWindowChunker chunker(10, 2);
    EXPECT_TRUE(chunker.chunk("a.cpp", "").empty());
}

TEST(LineWindowChunkerTest, SmallSourceFitsOneChunk) {
    preprocessor::LineWindowChunker chunker(60, 10);
    auto chunks = chunker.chunk("a.cpp", make_source(5));
    ASSERT_EQ(chunks.size(), 1u);
    EXPECT_EQ(chunks[0].start_line, 1u);
    EXPECT_EQ(chunks[0].end_line, 5u);
    EXPECT_EQ(chunks[0].file_path, "a.cpp");
    EXPECT_NE(chunks[0].id, 0u);
}

TEST(LineWindowChunkerTest, OverlappingWindows) {
    preprocessor::LineWindowChunker chunker(10, 2);
    auto chunks = chunker.chunk("big.cpp", make_source(25));
    ASSERT_GE(chunks.size(), 3u);
    // Each window steps by window - overlap = 8 lines.
    EXPECT_EQ(chunks[0].start_line, 1u);
    EXPECT_EQ(chunks[1].start_line, 9u);
    EXPECT_EQ(chunks[2].start_line, 17u);
}

TEST(LineWindowChunkerTest, IdsAreContentAddressed) {
    preprocessor::LineWindowChunker chunker(10, 2);
    auto a = chunker.chunk("file_a.cpp", make_source(20));
    auto b = chunker.chunk("file_b.cpp", make_source(20));
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].id, b[i].id);  // same content -> same id
    }
}

TEST(LineWindowChunkerTest, RejectsBadConfig) {
    EXPECT_THROW(preprocessor::LineWindowChunker(0, 0), std::invalid_argument);
    EXPECT_THROW(preprocessor::LineWindowChunker(5, 5), std::invalid_argument);
    EXPECT_THROW(preprocessor::LineWindowChunker(5, 10), std::invalid_argument);
}
