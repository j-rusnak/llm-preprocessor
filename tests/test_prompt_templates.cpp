#include "prompt_templates.hpp"

#include <atomic>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using preprocessor::PromptBucket;
using preprocessor::PromptRenderInput;
using preprocessor::PromptTemplates;

namespace {
struct TempFile {
    std::string path;
    // Intentionally no cleanup destructor: copy/move would unlink the file
    // before the test reads it. OS temp dir handles eventual cleanup.
};

std::atomic<int> g_tmp_counter{0};

TempFile write_temp(const std::string& contents) {
    TempFile t;
    auto p = std::filesystem::temp_directory_path() /
             ("llm_pp_tmpl_" + std::to_string(g_tmp_counter.fetch_add(1)) +
              "_" + std::to_string(std::time(nullptr)) + ".json");
    t.path = p.string();
    std::ofstream f(t.path, std::ios::binary);
    f << contents;
    f.close();
    return t;
}
} // namespace

TEST(PromptTemplates, DefaultsCoverEveryBucket) {
    PromptTemplates t;
    for (auto b : {PromptBucket::CodeEdit, PromptBucket::CodeExplain,
                   PromptBucket::CodeGenerate, PromptBucket::MetaQuery,
                   PromptBucket::Freeform}) {
        EXPECT_FALSE(t.source_for(b).empty());
    }
}

TEST(PromptTemplates, RendersUserAndContext) {
    PromptTemplates t;
    PromptRenderInput in;
    in.context = "snippet body";
    in.project_card = "card body";
    in.user_message = "hi";
    auto out = t.render(PromptBucket::Freeform, in);
    EXPECT_NE(out.find("snippet body"), std::string::npos);
    EXPECT_NE(out.find("card body"), std::string::npos);
}

TEST(PromptTemplates, OmitsEmptyProjectCardSection) {
    PromptTemplates t;
    PromptRenderInput in;
    in.context = "ctx";
    auto out = t.render(PromptBucket::CodeEdit, in);
    EXPECT_NE(out.find("ctx"), std::string::npos);
    // No leftover empty card marker.
    EXPECT_EQ(out.find("card body"), std::string::npos);
}

TEST(PromptTemplates, SetTemplateOverride) {
    PromptTemplates t;
    t.set_template(PromptBucket::Freeform, "MSG={{ user_message }}");
    PromptRenderInput in;
    in.user_message = "hello";
    EXPECT_EQ(t.render(PromptBucket::Freeform, in), "MSG=hello");
}

TEST(PromptTemplates, LoadFromFileOverridesOnlyListedBuckets) {
    auto tmp = write_temp(R"({"code_edit": "EDIT={{ user_message }}"})");
    auto t = PromptTemplates::load_from_file(tmp.path);
    PromptRenderInput in;
    in.user_message = "go";
    EXPECT_EQ(t.render(PromptBucket::CodeEdit, in), "EDIT=go");
    // Freeform keeps its default and still renders.
    EXPECT_FALSE(t.render(PromptBucket::Freeform, in).empty());
}

TEST(PromptTemplates, LoadFromFileRejectsUnknownBucket) {
    auto tmp = write_temp(R"({"unknown_bucket": "x"})");
    EXPECT_THROW(PromptTemplates::load_from_file(tmp.path), std::runtime_error);
}

TEST(PromptTemplates, LoadFromFileRejectsBadJson) {
    auto tmp = write_temp("{ not json");
    EXPECT_THROW(PromptTemplates::load_from_file(tmp.path), std::runtime_error);
}
