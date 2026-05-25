#include "intent_classifier.hpp"
#include "prompt_optimizer.hpp"
#include "prompt_templates.hpp"
#include "repo_index.hpp"

#include <memory>
#include <gtest/gtest.h>

using preprocessor::CodeChunk;
using preprocessor::HeuristicIntentClassifier;
using preprocessor::ProjectCard;
using preprocessor::PromptBucket;
using preprocessor::PromptOptimizer;
using preprocessor::PromptOptimizerConfig;
using preprocessor::PromptTemplates;
using preprocessor::RetrievedChunk;

namespace {

RetrievedChunk mkchunk(std::uint64_t id, std::string text,
                       std::string file = "f.cpp", std::string sym = "fn") {
    RetrievedChunk r;
    r.score = 1.0f;
    r.chunk.id = id;
    r.chunk.text = std::move(text);
    r.chunk.file_path = std::move(file);
    r.chunk.symbol = std::move(sym);
    r.chunk.start_line = 1;
    r.chunk.end_line = 10;
    return r;
}

PromptOptimizer make_optimiser(bool enabled = true, bool include_card = true) {
    auto cls = std::make_shared<HeuristicIntentClassifier>();
    PromptOptimizerConfig cfg;
    cfg.enabled = enabled;
    cfg.include_project_card = include_card;
    return PromptOptimizer(PromptTemplates{}, std::move(cls), cfg);
}

} // namespace

TEST(PromptOptimizer, RejectsNullClassifier) {
    EXPECT_THROW(PromptOptimizer(PromptTemplates{}, nullptr, {}), std::invalid_argument);
}

TEST(PromptOptimizer, DisabledFallsBackToPlainBlock) {
    auto opt = make_optimiser(/*enabled=*/false);
    auto r = opt.optimise("rewrite this function",
                          {mkchunk(1, "void f(){}", "a.cpp", "f")});
    EXPECT_EQ(r.bucket, PromptBucket::CodeEdit);
    EXPECT_FALSE(r.used_template);
    EXPECT_NE(r.system_message.find("Retrieved code context"), std::string::npos);
    EXPECT_NE(r.system_message.find("a.cpp"), std::string::npos);
}

TEST(PromptOptimizer, EnabledRendersTemplateForBucket) {
    auto opt = make_optimiser(/*enabled=*/true, /*include_card=*/false);
    auto r = opt.optimise("Explain how this works",
                          {mkchunk(1, "int g(){return 0;}")});
    EXPECT_EQ(r.bucket, PromptBucket::CodeExplain);
    EXPECT_TRUE(r.used_template);
    // CodeExplain default header phrase.
    EXPECT_NE(r.system_message.find("code reviewer"), std::string::npos);
    EXPECT_NE(r.system_message.find("int g()"), std::string::npos);
}

TEST(PromptOptimizer, EmptyChunksAndNoCardYieldsEmptyMessage) {
    auto opt = make_optimiser(/*enabled=*/true, /*include_card=*/false);
    auto r = opt.optimise("hello", {});
    EXPECT_EQ(r.bucket, PromptBucket::Freeform);
    EXPECT_FALSE(r.used_template);
    EXPECT_TRUE(r.system_message.empty());
}

TEST(PromptOptimizer, ProjectCardInjectedWhenAvailable) {
    auto opt = make_optimiser(/*enabled=*/true, /*include_card=*/true);
    ProjectCard card;
    card.root_path = "/repo";
    card.total_files = 1;
    card.total_chunks = 2;
    opt.set_project_card(card);

    auto r = opt.optimise("freeform question", {});
    EXPECT_TRUE(r.used_template);
    EXPECT_NE(r.system_message.find("/repo"), std::string::npos);
}

TEST(PromptOptimizer, ToggleAtRuntime) {
    auto opt = make_optimiser(/*enabled=*/true, /*include_card=*/false);
    EXPECT_TRUE(opt.enabled());
    opt.set_enabled(false);
    EXPECT_FALSE(opt.enabled());
    auto r = opt.optimise("write a new fn", {mkchunk(1, "// hi")});
    EXPECT_FALSE(r.used_template);
}

TEST(PromptOptimizer, ContextCharBudgetTruncates) {
    auto cls = std::make_shared<HeuristicIntentClassifier>();
    PromptOptimizerConfig cfg;
    cfg.enabled = true;
    cfg.include_project_card = false;
    cfg.max_context_chars = 60;
    PromptOptimizer opt(PromptTemplates{}, cls, cfg);

    std::string huge(500, 'X');
    auto r = opt.optimise("explain this",
                          {mkchunk(1, huge, "a.cpp", "f"),
                           mkchunk(2, huge, "b.cpp", "g")});
    // Budget should drop at least the second chunk.
    EXPECT_EQ(r.system_message.find("b.cpp"), std::string::npos);
}

TEST(PromptOptimizer, ResultIncludesPackedContextMetadata) {
    auto cls = std::make_shared<HeuristicIntentClassifier>();
    PromptOptimizerConfig cfg;
    cfg.enabled = true;
    cfg.include_project_card = false;
    cfg.max_context_chars = 45;
    PromptOptimizer opt(PromptTemplates{}, cls, cfg);

    auto r = opt.optimise("explain this",
                          {mkchunk(10, "int a;", "a.cpp", "a"),
                           mkchunk(20, std::string(200, 'B'), "b.cpp", "b")});

    ASSERT_EQ(r.packed_context.included_chunk_ids.size(), 1u);
    EXPECT_EQ(r.packed_context.included_chunk_ids[0], 10u);
    ASSERT_EQ(r.packed_context.omitted_chunk_ids.size(), 1u);
    EXPECT_EQ(r.packed_context.omitted_chunk_ids[0], 20u);
    EXPECT_TRUE(r.packed_context.truncated);
    EXPECT_NE(r.packed_context.text.find("a.cpp"), std::string::npos);
    EXPECT_EQ(r.packed_context.text.find("b.cpp"), std::string::npos);
}
