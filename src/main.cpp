#include "chat_history_store.hpp"
#include "code_chunker.hpp"
#include "config_loader.hpp"
#include "context_gatherer.hpp"
#include "embedding_engine.hpp"
#include "intent_classifier.hpp"
#include "intent_router.hpp"
#include "llm_tokenizer.hpp"
#include "mcp_server.hpp"
#include "openai_proxy.hpp"
#include "project_card.hpp"
#include "prompt_cache.hpp"
#include "prompt_compiler.hpp"
#include "prompt_optimizer.hpp"
#include "prompt_rewriter.hpp"
#include "prompt_templates.hpp"
#include "proxy_metrics.hpp"
#include "repo_index.hpp"
#include "structural_query_engine.hpp"
#include "symbol_graph.hpp"
#include "text_sanitizer.hpp"
#include "tokenizer.hpp"

#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include <curl/curl.h>

#ifndef PREPROCESSOR_VERSION
#define PREPROCESSOR_VERSION "unknown"
#endif
static constexpr const char* VERSION = PREPROCESSOR_VERSION;

static void print_help() {
    std::cout << "Usage: preprocessor_app [OPTIONS] [config_path]\n\n"
              << "Options:\n"
              << "  --help      Show this help message and exit\n"
              << "  --version   Show version information and exit\n"
              << "  --serve     Run the OpenAI-compatible HTTP proxy\n"
              << "  --mcp       Run as an MCP server over stdio (JSON-RPC)\n"
              << "  --health    Self-check (config + model files); exits 0 if healthy\n\n"
              << "Arguments:\n"
              << "  config_path  Path to JSON config file (default: config.json)\n";
}

static int run_serve(const preprocessor::Config& config) {
    if (!std::filesystem::exists(config.model_path) ||
        !std::filesystem::exists(config.vocab_path)) {
        std::cerr << "[FATAL] --serve requires model + vocab files. Missing:\n"
                  << "  model_path: " << config.model_path << "\n"
                  << "  vocab_path: " << config.vocab_path << "\n";
        return 2;
    }

    auto tokenizer = std::make_shared<preprocessor::Tokenizer>(config.vocab_path);
    auto embedder  = std::make_shared<preprocessor::EmbeddingEngine>(config.model_path, tokenizer);
    auto chunker   = std::make_shared<preprocessor::BraceAwareChunker>();

    preprocessor::RepoIndexConfig idx_cfg;
    idx_cfg.embedding_dim = config.embedding_dim;
    preprocessor::RepoIndex index(embedder, chunker, idx_cfg);

    // Phase 3: optional symbol graph. Attach BEFORE index_path so the graph
    // is populated as part of the initial walk.
    std::unique_ptr<preprocessor::SymbolGraph> symbol_graph;
    std::unique_ptr<preprocessor::RegexSymbolExtractor> symbol_extractor;
    if (config.symbol_graph_enabled) {
        symbol_graph = std::make_unique<preprocessor::SymbolGraph>();
        symbol_extractor = std::make_unique<preprocessor::RegexSymbolExtractor>();
        index.attach_symbol_graph(symbol_graph.get(), symbol_extractor.get());
    }

    if (!config.repo_root.empty()) {
        std::cout << "[INFO] Indexing repo: " << config.repo_root << "\n";
        index.index_path(config.repo_root);
        std::cout << "[INFO] Indexed " << index.file_count() << " files / "
                  << index.chunk_count() << " chunks\n";
        if (symbol_graph) {
            std::cout << "[INFO] Symbol graph: "
                      << symbol_graph->definition_count() << " defs / "
                      << symbol_graph->reference_count() << " refs\n";
        }
    }

    preprocessor::PromptCache cache(config.cache_db_path);
    preprocessor::ProxyMetrics metrics;
    preprocessor::HeuristicLLMTokenizer llm_tokenizer;

    preprocessor::OpenAIProxyConfig pcfg;
    pcfg.upstream_url = config.upstream_url;
    pcfg.upstream_api_key = config.upstream_api_key;
    pcfg.retrieval_k = config.retrieval_k;
    pcfg.max_context_chars = config.max_context_chars;

    preprocessor::OpenAIProxy proxy(index, cache, metrics, llm_tokenizer, pcfg);

    // Phase 2: optional prompt optimiser.
    std::unique_ptr<preprocessor::PromptOptimizer> optimiser;
    if (config.prompt_optimizer_enabled) {
        preprocessor::PromptTemplates templates =
            config.prompt_templates_path.empty()
                ? preprocessor::PromptTemplates{}
                : preprocessor::PromptTemplates::load_from_file(config.prompt_templates_path);
        auto classifier = std::make_shared<preprocessor::HeuristicIntentClassifier>();
        preprocessor::PromptOptimizerConfig ocfg;
        ocfg.enabled = true;
        ocfg.max_context_chars = config.max_context_chars;
        ocfg.include_project_card = config.include_project_card;
        optimiser = std::make_unique<preprocessor::PromptOptimizer>(
            std::move(templates), std::move(classifier), ocfg);
        if (config.include_project_card && !config.repo_root.empty()) {
            optimiser->set_project_card(
                preprocessor::ProjectCardBuilder::build(index, config.repo_root));
        }
        proxy.set_prompt_optimizer(optimiser.get());
        std::cout << "[INFO] Prompt optimiser enabled (templates: "
                  << (config.prompt_templates_path.empty() ? "built-in"
                                                           : config.prompt_templates_path)
                  << ")\n";
    }

    // Phase 3: wire graph expansion + structural fast path.
    std::unique_ptr<preprocessor::StructuralQueryEngine> structural;
    if (symbol_graph && config.graph_expansion_enabled) {
        proxy.set_symbol_graph(symbol_graph.get());
        std::cout << "[INFO] Graph-aware retrieval expansion enabled\n";
    }
    if (symbol_graph && config.structural_fast_path_enabled) {
        structural = std::make_unique<preprocessor::StructuralQueryEngine>(
            *symbol_graph, index);
        proxy.set_structural_query_engine(structural.get());
        std::cout << "[INFO] Structural query fast path enabled\n";
    }

    // Phase 5: optional prompt rewriter / context compressor.
    std::unique_ptr<preprocessor::IPromptRewriter> rewriter;
    if (config.prompt_rewriter_enabled) {
        try {
            if (config.prompt_rewriter_kind == "llama-cpp") {
                preprocessor::LlamaCppRewriterConfig lcfg;
                lcfg.model_path = config.llama_model_path;
                rewriter = std::make_unique<preprocessor::LlamaCppRewriter>(lcfg);
            } else {
                preprocessor::HeuristicCompressionConfig hcfg;
                hcfg.hard_truncate_chars = config.prompt_rewriter_max_chars;
                rewriter = std::make_unique<preprocessor::HeuristicCompressionRewriter>(hcfg);
            }
            proxy.set_prompt_rewriter(rewriter.get());
            std::cout << "[INFO] Prompt rewriter enabled ("
                      << rewriter->name() << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "[WARN] Prompt rewriter disabled: " << e.what() << "\n";
        }
    }

    std::cout << "[INFO] Proxy listening on http://" << config.proxy_host << ":"
              << config.proxy_port << "  (upstream: " << config.upstream_url << ")\n";
    proxy.listen(config.proxy_host, config.proxy_port);
    return 0;
}

static int run_mcp(const preprocessor::Config& config) {
    if (!std::filesystem::exists(config.model_path) ||
        !std::filesystem::exists(config.vocab_path)) {
        std::cerr << "[FATAL] --mcp requires model + vocab files. Missing:\n"
                  << "  model_path: " << config.model_path << "\n"
                  << "  vocab_path: " << config.vocab_path << "\n";
        return 2;
    }

    auto tokenizer = std::make_shared<preprocessor::Tokenizer>(config.vocab_path);
    auto embedder  = std::make_shared<preprocessor::EmbeddingEngine>(config.model_path, tokenizer);
    auto chunker   = std::make_shared<preprocessor::BraceAwareChunker>();

    preprocessor::RepoIndexConfig idx_cfg;
    idx_cfg.embedding_dim = config.embedding_dim;
    preprocessor::RepoIndex index(embedder, chunker, idx_cfg);

    std::unique_ptr<preprocessor::SymbolGraph> symbol_graph;
    std::unique_ptr<preprocessor::RegexSymbolExtractor> symbol_extractor;
    if (config.symbol_graph_enabled) {
        symbol_graph = std::make_unique<preprocessor::SymbolGraph>();
        symbol_extractor = std::make_unique<preprocessor::RegexSymbolExtractor>();
        index.attach_symbol_graph(symbol_graph.get(), symbol_extractor.get());
    }

    if (!config.repo_root.empty()) {
        std::cerr << "[INFO] Indexing repo: " << config.repo_root << "\n";
        index.index_path(config.repo_root);
        std::cerr << "[INFO] Indexed " << index.file_count() << " files / "
                  << index.chunk_count() << " chunks\n";
    }

    preprocessor::McpServerConfig mcfg;
    mcfg.default_search_k = config.retrieval_k;
    preprocessor::McpServer server(index, mcfg);

    std::unique_ptr<preprocessor::StructuralQueryEngine> structural;
    if (symbol_graph) {
        server.set_symbol_graph(symbol_graph.get());
        if (config.structural_fast_path_enabled) {
            structural = std::make_unique<preprocessor::StructuralQueryEngine>(
                *symbol_graph, index);
            server.set_structural_engine(structural.get());
        }
    }

    preprocessor::ProjectCard card;
    if (config.include_project_card && !config.repo_root.empty()) {
        card = preprocessor::ProjectCardBuilder::build(index, config.repo_root);
        server.set_project_card(&card);
    }

    std::cerr << "[INFO] MCP server ready on stdio (JSON-RPC 2.0)\n";
    server.serve_stdio(std::cin, std::cout);
    return 0;
}

int main(int argc, char* argv[]) {
    bool serve_mode = false;
    bool mcp_mode = false;
    bool health_mode = false;
    std::string config_path = "config.json";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_help();
            return 0;
        }
        if (std::strcmp(argv[i], "--version") == 0 || std::strcmp(argv[i], "-v") == 0) {
            std::cout << "LLM Preprocessor v" << VERSION << "\n";
            return 0;
        }
        if (std::strcmp(argv[i], "--serve") == 0) {
            serve_mode = true;
            continue;
        }
        if (std::strcmp(argv[i], "--mcp") == 0) {
            mcp_mode = true;
            continue;
        }
        if (std::strcmp(argv[i], "--health") == 0) {
            health_mode = true;
            continue;
        }
        config_path = argv[i];
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);

    int exit_code = 0;
    try {
        preprocessor::Config config = preprocessor::ConfigLoader::load(config_path);

        if (health_mode) {
            // Phase 12 self-check: verify config loads + ONNX assets exist.
            bool ok = true;
            std::cout << "config: OK (" << config_path << ")\n";
            if (std::filesystem::exists(config.model_path)) {
                std::cout << "model: OK (" << config.model_path << ")\n";
            } else {
                std::cout << "model: MISSING (" << config.model_path << ")\n";
                ok = false;
            }
            if (std::filesystem::exists(config.vocab_path)) {
                std::cout << "vocab: OK (" << config.vocab_path << ")\n";
            } else {
                std::cout << "vocab: MISSING (" << config.vocab_path << ")\n";
                ok = false;
            }
            std::cout << (ok ? "HEALTHY" : "DEGRADED") << "\n";
            curl_global_cleanup();
            return ok ? 0 : 1;
        }
        if (serve_mode) {
            exit_code = run_serve(config);
            curl_global_cleanup();
            return exit_code;
        }
        if (mcp_mode) {
            exit_code = run_mcp(config);
            curl_global_cleanup();
            return exit_code;
        }

        preprocessor::ChatHistoryStore history_store(config.db_path);
        preprocessor::PromptCompiler compiler(config.system_prompt);
        preprocessor::HeuristicLLMTokenizer llm_tokenizer;

        preprocessor::ApiParams api_params;
        api_params.model = config.api_model;
        api_params.temperature = config.temperature;
        api_params.max_tokens = config.max_tokens;
        bool use_api_payload = api_params.model.has_value();

        std::unique_ptr<preprocessor::IntentRouter> router;
        bool routing_enabled = false;

        if (std::filesystem::exists(config.model_path) &&
            std::filesystem::exists(config.vocab_path)) {
            auto tokenizer = std::make_shared<preprocessor::Tokenizer>(config.vocab_path);
            auto engine = std::make_shared<preprocessor::EmbeddingEngine>(config.model_path, tokenizer);
            router = std::make_unique<preprocessor::IntentRouter>(config.similarity_threshold, engine);

            for (const auto& [name, examples] : config.intents) {
                for (const auto& example : examples) {
                    router->add_intent(name, example);
                }
                std::cout << "  Registered intent: " << name
                          << " (" << examples.size() << " examples)\n";
            }
            routing_enabled = true;
        } else {
            std::cout << "[INFO] Model files not found - semantic routing disabled.\n";
            std::cout << "       model_path: " << config.model_path << "\n";
            std::cout << "       vocab_path: " << config.vocab_path << "\n";
        }

        std::cout << "\nLLM Preprocessor ready. Type your input (or 'quit' to exit).\n\n";

        std::string line;
        while (true) {
            std::cout << "> ";
            if (!std::getline(std::cin, line)) {
                break;
            }

            std::string user_input = preprocessor::TextSanitizer::sanitize(line);
            if (user_input.empty()) {
                continue;
            }
            if (user_input == "quit" || user_input == "exit") {
                break;
            }

            if (routing_enabled) {
                auto matched = router->route(user_input);
                if (matched) {
                    std::cout << "[ACTION] " << matched->intent_name
                              << " (score: " << matched->score << ")\n\n";
                    history_store.add_message("user", user_input);
                    history_store.add_message("assistant", "Executed local action: " + matched->intent_name);
                    continue;
                }
            }

            std::string retrieved_context;
            auto urls = preprocessor::ContextGatherer::extract_urls(line);
            for (const auto& url : urls) {
                try {
                    std::cout << "[FETCH] " << url << "\n";
                    retrieved_context += preprocessor::ContextGatherer::fetch_url(url);
                } catch (const std::exception& e) {
                    std::cerr << "[WARN] Failed to fetch " << url << ": " << e.what() << "\n";
                }
            }

            auto history = history_store.get_recent_history(config.history_limit);

            if (use_api_payload) {
                std::vector<std::pair<std::string, std::string>> empty_history;
                auto display = compiler.build_payload_json(user_input, retrieved_context, empty_history, api_params);
                const std::string dumped = display.dump(4);
                std::cout << "\n=== LLM Payload ===\n" << dumped << "\n";
                std::cout << "[~tokens: " << llm_tokenizer.count_tokens(dumped) << "]\n";
            } else {
                std::vector<std::pair<std::string, std::string>> empty_history;
                std::string display_payload = compiler.build_payload(user_input, retrieved_context, empty_history);
                std::cout << "\n=== LLM Payload ===\n" << display_payload << "\n";
                std::cout << "[~tokens: " << llm_tokenizer.count_tokens(display_payload) << "]\n";
            }
            if (!history.empty()) {
                std::cout << "(+ " << history.size() << " history messages included in payload)\n";
            }
            std::cout << "\n";

            history_store.add_message("user", user_input);
            history_store.prune(config.history_limit * 2);
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        exit_code = 1;
    }

    curl_global_cleanup();
    return exit_code;
}
