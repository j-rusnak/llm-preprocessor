#include "diff_patcher.hpp"

#include <gtest/gtest.h>

using preprocessor::DiffPatcher;

namespace {
const char* kSimpleDiff =
    "diff --git a/foo.txt b/foo.txt\n"
    "--- a/foo.txt\n"
    "+++ b/foo.txt\n"
    "@@ -1,3 +1,3 @@\n"
    " alpha\n"
    "-beta\n"
    "+BETA\n"
    " gamma\n";
}

TEST(DiffPatcher, ParsesSimpleHunk) {
    DiffPatcher p;
    auto files = p.parse(kSimpleDiff);
    ASSERT_EQ(files.size(), 1u);
    EXPECT_EQ(files[0].old_path, "foo.txt");
    EXPECT_EQ(files[0].new_path, "foo.txt");
    ASSERT_EQ(files[0].hunks.size(), 1u);
    EXPECT_EQ(files[0].hunks[0].old_start, 1u);
    EXPECT_EQ(files[0].hunks[0].old_count, 3u);
}

TEST(DiffPatcher, AppliesToMatchingContent) {
    DiffPatcher p;
    std::unordered_map<std::string, std::string> contents{
        {"foo.txt", "alpha\nbeta\ngamma\n"}
    };
    auto r = p.apply(kSimpleDiff, contents);
    ASSERT_TRUE(r.ok);
    ASSERT_EQ(r.files.size(), 1u);
    EXPECT_TRUE(r.files[0].applied);
    EXPECT_EQ(r.files[0].patched_content, "alpha\nBETA\ngamma\n");
}

TEST(DiffPatcher, FailsOnContextMismatch) {
    DiffPatcher p;
    std::unordered_map<std::string, std::string> contents{
        {"foo.txt", "alpha\nXXX\ngamma\n"}
    };
    auto r = p.apply(kSimpleDiff, contents);
    EXPECT_FALSE(r.ok);
    ASSERT_EQ(r.files.size(), 1u);
    EXPECT_FALSE(r.files[0].applied);
    EXPECT_FALSE(r.files[0].error.empty());
}

TEST(DiffPatcher, StripPrefixHandlesAB) {
    EXPECT_EQ(DiffPatcher::strip_prefix("a/src/x.cpp"), "src/x.cpp");
    EXPECT_EQ(DiffPatcher::strip_prefix("b/src/x.cpp"), "src/x.cpp");
    EXPECT_EQ(DiffPatcher::strip_prefix("src/x.cpp"), "src/x.cpp");
}

TEST(DiffPatcher, ParseStrictThrowsOnEmpty) {
    DiffPatcher p;
    EXPECT_THROW(p.parse_strict(""), std::runtime_error);
}
