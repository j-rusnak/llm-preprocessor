#include "prompt_rewriter.hpp"

#include <gtest/gtest.h>
#include <string>

using preprocessor::HeuristicCompressionConfig;
using preprocessor::HeuristicCompressionRewriter;
using preprocessor::LlamaCppRewriter;
using preprocessor::LlamaCppRewriterConfig;

namespace {

std::string lines(std::initializer_list<const char*> ls) {
    std::string out;
    for (auto l : ls) { out.append(l); out.push_back('\n'); }
    return out;
}

} // namespace

TEST(HeuristicCompressionRewriter, EmptyInputReturnsEmpty) {
    HeuristicCompressionRewriter r;
    EXPECT_EQ(r.rewrite("", 0), "");
}

TEST(HeuristicCompressionRewriter, StripsCppLineComments) {
    HeuristicCompressionRewriter r;
    std::string in = lines({
        "int add(int a, int b) { // sum",
        "    return a + b; // returns sum",
        "}"
    });
    auto out = r.rewrite(in, 0);
    EXPECT_EQ(out.find("// sum"), std::string::npos);
    EXPECT_EQ(out.find("// returns sum"), std::string::npos);
    EXPECT_NE(out.find("return a + b;"), std::string::npos);
}

TEST(HeuristicCompressionRewriter, StripsCStyleBlockComments) {
    HeuristicCompressionRewriter r;
    std::string in = "int x = 1; /* block\ncomment */ int y = 2;\n";
    auto out = r.rewrite(in, 0);
    EXPECT_EQ(out.find("block"), std::string::npos);
    EXPECT_NE(out.find("int x"), std::string::npos);
    EXPECT_NE(out.find("int y"), std::string::npos);
}

TEST(HeuristicCompressionRewriter, PreservesCommentsInsideStrings) {
    HeuristicCompressionRewriter r;
    std::string in = "const char* s = \"keep // me\";\n";
    auto out = r.rewrite(in, 0);
    EXPECT_NE(out.find("keep // me"), std::string::npos);
}

TEST(HeuristicCompressionRewriter, StripsPythonHashCommentsAtLineStart) {
    HeuristicCompressionRewriter r;
    std::string in = lines({
        "# top-level note",
        "def add(a, b):",
        "    # inline note",
        "    return a + b",
    });
    auto out = r.rewrite(in, 0);
    EXPECT_EQ(out.find("top-level note"), std::string::npos);
    EXPECT_EQ(out.find("inline note"), std::string::npos);
    EXPECT_NE(out.find("def add"), std::string::npos);
    EXPECT_NE(out.find("return a + b"), std::string::npos);
}

TEST(HeuristicCompressionRewriter, KeepsCPreprocessorDirectives) {
    HeuristicCompressionRewriter r;
    std::string in = lines({
        "#include <vector>",
        "#define FOO 1",
        "int x;",
    });
    auto out = r.rewrite(in, 0);
    EXPECT_NE(out.find("#include <vector>"), std::string::npos);
    EXPECT_NE(out.find("#define FOO 1"), std::string::npos);
}

TEST(HeuristicCompressionRewriter, CollapsesBlankLines) {
    HeuristicCompressionRewriter r;
    std::string in = "a\n\n\n\n\nb\n";
    auto out = r.rewrite(in, 0);
    // Exactly one blank between a and b.
    EXPECT_NE(out.find("a\n\nb"), std::string::npos);
    EXPECT_EQ(out.find("\n\n\n"), std::string::npos);
}

TEST(HeuristicCompressionRewriter, DedupesAdjacentLines) {
    HeuristicCompressionRewriter r;
    std::string in = lines({"keep", "dup", "dup", "dup", "tail"});
    auto out = r.rewrite(in, 0);
    // Only one "dup" line should remain.
    auto first = out.find("dup");
    ASSERT_NE(first, std::string::npos);
    EXPECT_EQ(out.find("dup", first + 1), std::string::npos);
}

TEST(HeuristicCompressionRewriter, HardTruncateAppliesMaxCharsHint) {
    HeuristicCompressionRewriter r;
    std::string in(500, 'x');
    auto out = r.rewrite(in, 50);
    EXPECT_LE(out.size(), 50 + 30u);  // allow room for "[truncated]" marker
    EXPECT_NE(out.find("truncated"), std::string::npos);
}

TEST(HeuristicCompressionRewriter, ReducesRealisticContextBlock) {
    HeuristicCompressionRewriter r;
    std::string in = lines({
        "// File: src/math.cpp:1-6  (chunk_id=1)",
        "/* Module-level header */",
        "#include <cstdint>",
        "",
        "",
        "// Returns the sum",
        "int add(int a, int b) {",
        "    // trivial",
        "    return a + b;   ",
        "}",
        "",
        "",
        "",
    });
    auto out = r.rewrite(in, 0);
    EXPECT_LT(out.size(), in.size());
    EXPECT_NE(out.find("#include <cstdint>"), std::string::npos);
    EXPECT_NE(out.find("int add"), std::string::npos);
    EXPECT_EQ(out.find("trivial"), std::string::npos);
}

TEST(HeuristicCompressionRewriter, NameIsHeuristic) {
    HeuristicCompressionRewriter r;
    EXPECT_EQ(r.name(), "heuristic");
}

TEST(LlamaCppRewriter, AvailabilityMatchesBuildFlag) {
#ifdef LLM_PREPROCESSOR_WITH_LLAMA_CPP
    EXPECT_TRUE(LlamaCppRewriter::is_available());
#else
    EXPECT_FALSE(LlamaCppRewriter::is_available());
#endif
}

TEST(LlamaCppRewriter, ConstructorThrowsWhenDisabled) {
#ifndef LLM_PREPROCESSOR_WITH_LLAMA_CPP
    LlamaCppRewriterConfig cfg;
    cfg.model_path = "doesnt-matter.gguf";
    EXPECT_THROW(LlamaCppRewriter{cfg}, std::runtime_error);
#else
    GTEST_SKIP() << "llama.cpp support compiled in; constructor would succeed.";
#endif
}
