#include <gtest/gtest.h>
#include "prompt_compiler.hpp"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <utility>

TEST(PromptCompilerTest, BuildsPayloadWithAllSections) {
    preprocessor::PromptCompiler compiler("You are a test bot.");

    std::vector<std::pair<std::string, std::string>> history = {
        {"user", "hello"},
        {"assistant", "hi"}
    };

    std::string payload = compiler.build_payload("what is 2+2?", "math context", history);
    auto j = nlohmann::json::parse(payload);

    ASSERT_TRUE(j.is_array());
    ASSERT_EQ(j.size(), 4u); // system + 2 history + 1 user

    EXPECT_EQ(j[0]["role"], "system");
    EXPECT_EQ(j[0]["content"], "You are a test bot.");

    EXPECT_EQ(j[1]["role"], "user");
    EXPECT_EQ(j[1]["content"], "hello");

    EXPECT_EQ(j[2]["role"], "assistant");
    EXPECT_EQ(j[2]["content"], "hi");

    EXPECT_EQ(j[3]["role"], "user");
    // Should contain the context wrapper.
    std::string user_content = j[3]["content"];
    EXPECT_NE(user_content.find("<context>"), std::string::npos);
    EXPECT_NE(user_content.find("math context"), std::string::npos);
    EXPECT_NE(user_content.find("what is 2+2?"), std::string::npos);
}

TEST(PromptCompilerTest, NoContextSkipsWrapper) {
    preprocessor::PromptCompiler compiler("system msg");

    std::vector<std::pair<std::string, std::string>> history;
    std::string payload = compiler.build_payload("raw question", "", history);
    auto j = nlohmann::json::parse(payload);

    ASSERT_EQ(j.size(), 2u); // system + user
    EXPECT_EQ(j[1]["content"], "raw question");
}

TEST(PromptCompilerTest, EmptyHistoryProducesSystemAndUser) {
    preprocessor::PromptCompiler compiler("sys");

    std::vector<std::pair<std::string, std::string>> history;
    std::string payload = compiler.build_payload("hi", "", history);
    auto j = nlohmann::json::parse(payload);

    ASSERT_EQ(j.size(), 2u);
    EXPECT_EQ(j[0]["role"], "system");
    EXPECT_EQ(j[1]["role"], "user");
}

TEST(PromptCompilerTest, OutputIsValidJson) {
    preprocessor::PromptCompiler compiler("test");

    std::vector<std::pair<std::string, std::string>> history = {
        {"user", "Special chars: \"quotes\" and <tags>"}
    };

    std::string payload = compiler.build_payload("input with \"quotes\"", "", history);
    // Should not throw — valid JSON.
    EXPECT_NO_THROW((void)nlohmann::json::parse(payload));
}
