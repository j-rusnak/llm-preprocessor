#include "symbol_graph.hpp"

#include <gtest/gtest.h>

using preprocessor::CodeChunk;
using preprocessor::ExtractedSymbols;
using preprocessor::RegexSymbolExtractor;
using preprocessor::SymbolGraph;
using preprocessor::SymbolKind;

namespace {

CodeChunk make_chunk(std::uint64_t id, const std::string& file,
                     std::size_t start, const std::string& text) {
    CodeChunk c;
    c.id = id;
    c.file_path = file;
    c.start_line = start;
    c.end_line = start + 5;
    c.text = text;
    return c;
}

} // namespace

TEST(RegexSymbolExtractor, ExtractsCppFunctionAndCall) {
    RegexSymbolExtractor ex;
    auto defc = make_chunk(1, "src/math.cpp", 10,
        "int add(int a,int b){ return a+b; }\n");
    auto callc = make_chunk(2, "src/main.cpp", 1,
        "int main(){ return add(1,2); }\n");
    auto dout = ex.extract(defc);
    auto cout_ = ex.extract(callc);
    bool has_add = false;
    for (const auto& d : dout.defs) if (d.name == "add") has_add = true;
    EXPECT_TRUE(has_add);
    bool refs_add = false;
    for (const auto& r : cout_.refs) if (r.name == "add") refs_add = true;
    EXPECT_TRUE(refs_add);
}

TEST(RegexSymbolExtractor, IgnoresIdentifiersInCommentsAndStrings) {
    RegexSymbolExtractor ex;
    auto c = make_chunk(2, "x.cpp", 1,
        "// fake_def() should not match\n"
        "const char* s = \"another_fake(x)\";\n"
        "void real_def() { do_thing(); }\n");
    auto out = ex.extract(c);
    for (const auto& d : out.defs) {
        EXPECT_NE(d.name, "fake_def");
    }
    for (const auto& r : out.refs) {
        EXPECT_NE(r.name, "another_fake");
    }
}

TEST(RegexSymbolExtractor, ExtractsClassStructEnum) {
    RegexSymbolExtractor ex;
    auto c = make_chunk(3, "x.cpp", 1,
        "class MyClass {};\nstruct MyStruct {};\nenum class MyEnum { A };\n");
    auto out = ex.extract(c);
    bool cls=false, st=false, en=false;
    for (const auto& d : out.defs) {
        if (d.name == "MyClass" && d.kind == SymbolKind::Class) cls = true;
        if (d.name == "MyStruct" && d.kind == SymbolKind::Struct) st = true;
        if (d.name == "MyEnum" && d.kind == SymbolKind::Enum) en = true;
    }
    EXPECT_TRUE(cls);
    EXPECT_TRUE(st);
    EXPECT_TRUE(en);
}

TEST(RegexSymbolExtractor, ExtractsTypeScriptExportsAndArrowFunctions) {
    RegexSymbolExtractor ex;
    auto c = make_chunk(4, "agent.ts", 1,
        "export function parseRoute(input: string) { return input.trim(); }\n"
        "export class AgentRunner {}\n"
        "const buildPrompt = (request: Request) => parseRoute(request.path);\n"
        "export const routeTable = { parseRoute };\n");
    auto out = ex.extract(c);
    bool parse_route = false;
    bool runner = false;
    bool build_prompt = false;
    bool route_table = false;
    for (const auto& d : out.defs) {
        if (d.name == "parseRoute" && d.kind == SymbolKind::Function) {
            parse_route = true;
        }
        if (d.name == "AgentRunner" && d.kind == SymbolKind::Class) {
            runner = true;
        }
        if (d.name == "buildPrompt" && d.kind == SymbolKind::Function) {
            build_prompt = true;
        }
        if (d.name == "routeTable" && d.kind == SymbolKind::Variable) {
            route_table = true;
        }
    }
    EXPECT_TRUE(parse_route);
    EXPECT_TRUE(runner);
    EXPECT_TRUE(build_prompt);
    EXPECT_TRUE(route_table);
}

TEST(RegexSymbolExtractor, IndexesQualifiedCppMethodBySimpleName) {
    RegexSymbolExtractor ex;
    auto c = make_chunk(5, "server.cpp", 40,
        "void HttpServer::Listen() { accept_loop(); }\n");
    auto out = ex.extract(c);
    bool qualified = false;
    bool simple = false;
    for (const auto& d : out.defs) {
        if (d.name == "HttpServer::Listen" && d.kind == SymbolKind::Method) {
            qualified = true;
        }
        if (d.name == "Listen" && d.kind == SymbolKind::Method) {
            simple = true;
        }
    }
    EXPECT_TRUE(qualified);
    EXPECT_TRUE(simple);
}

TEST(SymbolGraph, UpdateAndQueryRoundTrip) {
    SymbolGraph g;
    RegexSymbolExtractor ex;
    auto def_c = make_chunk(10, "a.cpp", 1, "void target(){}\n");
    auto call_c = make_chunk(11, "b.cpp", 1, "void caller(){ target(); }\n");
    g.update_chunk(def_c, ex.extract(def_c));
    g.update_chunk(call_c, ex.extract(call_c));
    EXPECT_GE(g.definition_count(), 2u);
    EXPECT_GE(g.reference_count(), 1u);
    auto defs = g.find_definitions("target");
    ASSERT_EQ(defs.size(), 1u);
    EXPECT_EQ(defs[0].file_path, "a.cpp");
    auto refs = g.find_references("target");
    EXPECT_FALSE(refs.empty());
}

TEST(SymbolGraph, RemoveFileDropsAllContributions) {
    SymbolGraph g;
    RegexSymbolExtractor ex;
    auto c1 = make_chunk(20, "a.cpp", 1, "void foo(){} void caller(){ foo(); }\n");
    auto c2 = make_chunk(21, "b.cpp", 1, "void bar(){}\n");
    g.update_chunk(c1, ex.extract(c1));
    g.update_chunk(c2, ex.extract(c2));
    EXPECT_FALSE(g.find_definitions("foo").empty());
    g.remove_file("a.cpp");
    EXPECT_TRUE(g.find_definitions("foo").empty());
    EXPECT_TRUE(g.find_references("foo").empty());
    EXPECT_FALSE(g.find_definitions("bar").empty());
}

TEST(SymbolGraph, NeighborsOfFollowsReferenceEdges) {
    SymbolGraph g;
    RegexSymbolExtractor ex;
    // helper defined in one chunk, called from another.
    auto helper = make_chunk(100, "helper.cpp", 1, "void helper_fn(){}\n");
    auto caller = make_chunk(101, "caller.cpp", 1, "void run(){ helper_fn(); }\n");
    g.update_chunk(helper, ex.extract(helper));
    g.update_chunk(caller, ex.extract(caller));
    auto n = g.neighbors_of({101}, 8);
    ASSERT_EQ(n.size(), 1u);
    EXPECT_EQ(n[0], 100u);
}

TEST(SymbolGraph, UpdateChunkReplacesPriorFacts) {
    SymbolGraph g;
    RegexSymbolExtractor ex;
    auto v1 = make_chunk(200, "x.cpp", 1, "void old_fn(){}\n");
    g.update_chunk(v1, ex.extract(v1));
    EXPECT_FALSE(g.find_definitions("old_fn").empty());
    auto v2 = make_chunk(200, "x.cpp", 1, "void new_fn(){}\n");
    g.update_chunk(v2, ex.extract(v2));
    EXPECT_TRUE(g.find_definitions("old_fn").empty());
    EXPECT_FALSE(g.find_definitions("new_fn").empty());
}

TEST(SymbolGraph, DefsInFileLists) {
    SymbolGraph g;
    RegexSymbolExtractor ex;
    auto c = make_chunk(300, "lib.cpp", 1, "void a(){} void b(){} void c_fn(){}\n");
    g.update_chunk(c, ex.extract(c));
    auto defs = g.defs_in_file("lib.cpp");
    EXPECT_GE(defs.size(), 3u);
}
