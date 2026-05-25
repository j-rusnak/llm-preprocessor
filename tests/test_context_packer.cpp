#include "context_packer.hpp"
#include "repo_index.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

preprocessor::RetrievedChunk chunk(std::uint64_t id,
                                   std::string file,
                                   std::string symbol,
                                   std::string text) {
    preprocessor::RetrievedChunk r;
    r.score = 1.0f;
    r.chunk.id = id;
    r.chunk.file_path = std::move(file);
    r.chunk.symbol = std::move(symbol);
    r.chunk.text = std::move(text);
    r.chunk.start_line = 3;
    r.chunk.end_line = 9;
    return r;
}

} // namespace

TEST(ContextPacker, PreservesLegacyContextFormattingWithHeader) {
    preprocessor::ContextPackerConfig cfg;
    cfg.include_header = true;

    const auto packed = preprocessor::pack_context(
        {chunk(7, "src/a.cpp", "run", "void run(){}")}, cfg);

    EXPECT_EQ(packed.text,
              "Retrieved code context (most relevant first):\n"
              "\n--- src/a.cpp [L3-L9] run ---\n"
              "void run(){}\n");
    ASSERT_EQ(packed.included_chunk_ids.size(), 1u);
    EXPECT_EQ(packed.included_chunk_ids[0], 7u);
    EXPECT_TRUE(packed.omitted_chunk_ids.empty());
    EXPECT_FALSE(packed.truncated);
}

TEST(ContextPacker, CanPackWithoutHeaderForTemplateSlots) {
    preprocessor::ContextPackerConfig cfg;
    cfg.include_header = false;

    const auto packed = preprocessor::pack_context(
        {chunk(1, "lib/mod.ts", "buildPrompt", "export const x = 1;")}, cfg);

    EXPECT_EQ(packed.text,
              "\n--- lib/mod.ts [L3-L9] buildPrompt ---\n"
              "export const x = 1;\n");
}

TEST(ContextPacker, RecordsOmittedChunksWhenBudgetIsExhausted) {
    preprocessor::ContextPackerConfig cfg;
    cfg.include_header = true;
    cfg.max_context_chars = 64;

    const auto packed = preprocessor::pack_context(
        {chunk(1, "a.cpp", "a", "int a;"),
         chunk(2, "b.cpp", "b", std::string(200, 'B'))},
        cfg);

    ASSERT_EQ(packed.included_chunk_ids.size(), 1u);
    EXPECT_EQ(packed.included_chunk_ids[0], 1u);
    ASSERT_EQ(packed.omitted_chunk_ids.size(), 1u);
    EXPECT_EQ(packed.omitted_chunk_ids[0], 2u);
    EXPECT_TRUE(packed.truncated);
    EXPECT_NE(packed.text.find("a.cpp"), std::string::npos);
    EXPECT_EQ(packed.text.find("b.cpp"), std::string::npos);
}

TEST(ContextPacker, ZeroBudgetKeepsAllChunks) {
    preprocessor::ContextPackerConfig cfg;
    cfg.max_context_chars = 0;

    const auto packed = preprocessor::pack_context(
        {chunk(1, "a.cpp", "a", std::string(200, 'A')),
         chunk(2, "b.cpp", "b", std::string(200, 'B'))},
        cfg);

    EXPECT_EQ(packed.included_chunk_ids.size(), 2u);
    EXPECT_TRUE(packed.omitted_chunk_ids.empty());
    EXPECT_FALSE(packed.truncated);
}
