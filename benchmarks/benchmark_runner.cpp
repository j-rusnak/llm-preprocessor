// LLM Preprocessor — Benchmark Runner
//
// Runs a diverse set of test prompts through the routing pipeline,
// collects latency, accuracy, and similarity data, then outputs
// structured JSON to stdout for visualization by visualize.py.
//
// Run from the project root (where config.json and models/ live):
//   .\build\benchmark_runner.exe > benchmarks\results\benchmark_data.json

#include "config_loader.hpp"
#include "embedding_engine.hpp"
#include "intent_router.hpp"
#include "text_sanitizer.hpp"
#include "tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::high_resolution_clock;

namespace {

struct TestCase {
    std::string category;
    std::string input;
    std::string expected_intent; // empty = should NOT match any intent
};

int count_words(const std::string& s) {
    std::istringstream iss(s);
    std::string w;
    int n = 0;
    while (iss >> w) ++n;
    return n;
}

} // anonymous namespace

int main() {
    // ----------------------------------------------------------------
    // Prerequisites
    // ----------------------------------------------------------------
    const std::string config_path = "config.json";
    if (!std::filesystem::exists(config_path)) {
        std::cerr << "Error: config.json not found. Run from the project root.\n";
        return 1;
    }

    auto config = preprocessor::ConfigLoader::load(config_path);

    if (!std::filesystem::exists(config.model_path) ||
        !std::filesystem::exists(config.vocab_path)) {
        std::cerr << "Error: Model files not found.\n"
                  << "  model_path: " << config.model_path << "\n"
                  << "  vocab_path: " << config.vocab_path << "\n";
        return 1;
    }

    auto tokenizer = std::make_shared<preprocessor::Tokenizer>(config.vocab_path);
    auto engine    = std::make_shared<preprocessor::EmbeddingEngine>(config.model_path, tokenizer);

    // Warm-up ONNX Runtime (first inference is often slower).
    engine->generate_embedding("warmup");

    // ----------------------------------------------------------------
    // Register intents (same as main app)
    // ----------------------------------------------------------------
    preprocessor::IntentRouter router(config.similarity_threshold, engine);

    struct IntentEmbedding {
        std::string name;
        std::vector<float> embedding;
    };
    std::vector<IntentEmbedding> all_intent_embeddings;
    int total_examples = 0;

    for (const auto& [name, examples] : config.intents) {
        for (const auto& example : examples) {
            router.add_intent(name, example);
            auto emb = engine->generate_embedding(example);
            all_intent_embeddings.push_back({name, std::move(emb)});
            ++total_examples;
        }
    }
    std::cerr << "Registered " << config.intents.size() << " intents ("
              << total_examples << " examples)\n";

    // ----------------------------------------------------------------
    // Define test prompts across five complexity categories
    // ----------------------------------------------------------------
    std::vector<TestCase> test_cases = {
        // ---- Direct commands (clean, should match easily) ----
        {"direct_command", "mute the sound",           "ACTION_MUTE"},
        {"direct_command", "turn up the volume",       "ACTION_INCREASE_VOLUME"},
        {"direct_command", "turn down the volume",     "ACTION_DECREASE_VOLUME"},
        {"direct_command", "open the web browser",     "ACTION_OPEN_BROWSER"},
        {"direct_command", "take a screenshot",        "ACTION_TAKE_SCREENSHOT"},
        {"direct_command", "lock my computer",         "ACTION_LOCK_SCREEN"},
        {"direct_command", "open a file",              "ACTION_OPEN_FILE"},
        {"direct_command", "close this file",          "ACTION_CLOSE_FILE"},

        // ---- Noisy commands (filler words, slang — tests stop-word filtering) ----
        {"noisy_command", "hey bro can you mute that",                      "ACTION_MUTE"},
        {"noisy_command", "yo dude turn up the volume please",              "ACTION_INCREASE_VOLUME"},
        {"noisy_command", "could you please turn the volume down for me",   "ACTION_DECREASE_VOLUME"},
        {"noisy_command", "hey can you open the browser real quick",        "ACTION_OPEN_BROWSER"},
        {"noisy_command", "yo take a screenshot of this thing",             "ACTION_TAKE_SCREENSHOT"},
        {"noisy_command", "ok buddy lock the screen for me please",         "ACTION_LOCK_SCREEN"},

        // ---- Complex natural language (longer sentences) ----
        {"complex_command", "i was wondering if you could perhaps mute the audio output",            "ACTION_MUTE"},
        {"complex_command", "would it be possible to increase the sound level a bit",                "ACTION_INCREASE_VOLUME"},
        {"complex_command", "the music is too loud so please reduce the volume",                     "ACTION_DECREASE_VOLUME"},
        {"complex_command", "i need to browse the internet so open a browser window",                "ACTION_OPEN_BROWSER"},
        {"complex_command", "i really need you to capture what is on my screen right now",           "ACTION_TAKE_SCREENSHOT"},
        {"complex_command", "i am stepping away so please secure my workstation by locking it",      "ACTION_LOCK_SCREEN"},

        // ---- Non-matching queries (should NOT route) ----
        {"no_match", "what is the meaning of life",                ""},
        {"no_match", "tell me about quantum physics",              ""},
        {"no_match", "how do i make pasta carbonara",              ""},
        {"no_match", "what will the weather be like tomorrow",     ""},
        {"no_match", "explain the theory of relativity",           ""},
        {"no_match", "write me a poem about the ocean",            ""},

        // ---- Edge cases (very short, very long, ambiguous) ----
        {"edge_case", "mute",                                                                                      "ACTION_MUTE"},
        {"edge_case", "volume",                                                                                     ""},
        {"edge_case", "the",                                                                                        ""},
        {"edge_case", "please do something about the incredibly loud noise by lowering it",                         "ACTION_DECREASE_VOLUME"},
        {"edge_case", "mute the sound and open the browser",                                                        "ACTION_MUTE"},
    };

    constexpr int NUM_RUNS   = 10;  // repetitions per prompt for timing
    constexpr int EMBED_RUNS = 20;  // repetitions for embedding-only timing

    // ================================================================
    // Phase 1: Routing benchmarks — latency and accuracy
    // ================================================================
    std::cerr << "Phase 1: Routing benchmarks (" << test_cases.size()
              << " inputs x " << NUM_RUNS << " runs)...\n";

    json routing_results = json::array();

    for (const auto& tc : test_cases) {
        std::string sanitized = preprocessor::TextSanitizer::sanitize(tc.input);

        std::vector<long long> latencies;
        std::string matched_intent;
        float score  = 0.0f;
        bool matched = false;

        for (int run = 0; run < NUM_RUNS; ++run) {
            auto start  = Clock::now();
            auto result = router.route(sanitized);
            auto end    = Clock::now();

            long long us = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
            latencies.push_back(us);

            if (run == 0 && result) {
                matched_intent = result->intent_name;
                score   = result->score;
                matched = true;
            }
        }

        std::sort(latencies.begin(), latencies.end());
        double avg_us = std::accumulate(latencies.begin(), latencies.end(), 0.0)
                        / static_cast<double>(latencies.size());

        bool correct;
        if (tc.expected_intent.empty()) {
            correct = !matched;
        } else {
            correct = (matched && matched_intent == tc.expected_intent);
        }

        json entry;
        entry["category"]        = tc.category;
        entry["input"]           = tc.input;
        entry["sanitized"]       = sanitized;
        entry["expected_intent"] = tc.expected_intent;
        entry["matched"]         = matched;
        entry["matched_intent"]  = matched_intent;
        entry["score"]           = score;
        entry["correct"]         = correct;
        entry["word_count"]      = count_words(sanitized);
        entry["latency"] = {
            {"avg_us", avg_us},
            {"min_us", latencies.front()},
            {"max_us", latencies.back()},
            {"p50_us", latencies[latencies.size() / 2]},
            {"p95_us", latencies[static_cast<size_t>(latencies.size() * 0.95)]},
            {"all_us", latencies}
        };

        routing_results.push_back(entry);
    }

    // ================================================================
    // Phase 2: Raw similarity analysis (for threshold sweep)
    // ================================================================
    std::cerr << "Phase 2: Similarity analysis...\n";

    json similarity_analysis = json::array();

    for (const auto& tc : test_cases) {
        std::string sanitized = preprocessor::TextSanitizer::sanitize(tc.input);
        auto input_emb = engine->generate_embedding(sanitized);

        // Max similarity per unique intent name
        std::unordered_map<std::string, float> max_scores;
        for (const auto& ie : all_intent_embeddings) {
            float sim = preprocessor::detail::cosine_similarity(input_emb, ie.embedding);
            auto it = max_scores.find(ie.name);
            if (it == max_scores.end() || sim > it->second) {
                max_scores[ie.name] = sim;
            }
        }

        std::string best_intent;
        float best_score = -1.0f;
        json scores_obj;
        for (const auto& [name, s] : max_scores) {
            scores_obj[name] = std::round(s * 10000.0f) / 10000.0f;
            if (s > best_score) {
                best_score  = s;
                best_intent = name;
            }
        }

        json entry;
        entry["input"]            = tc.input;
        entry["expected_intent"]  = tc.expected_intent;
        entry["best_intent"]      = best_intent;
        entry["best_score"]       = std::round(best_score * 10000.0f) / 10000.0f;
        entry["scores_by_intent"] = scores_obj;
        similarity_analysis.push_back(entry);
    }

    // ================================================================
    // Phase 3: Intent-to-intent similarity matrix
    // ================================================================
    std::cerr << "Phase 3: Intent similarity matrix...\n";

    std::vector<std::string> intent_labels;
    std::vector<std::vector<float>> representative_embeddings;
    for (const auto& [name, examples] : config.intents) {
        intent_labels.push_back(name);
        representative_embeddings.push_back(engine->generate_embedding(examples[0]));
    }

    json sim_matrix = json::array();
    for (size_t i = 0; i < representative_embeddings.size(); ++i) {
        json row = json::array();
        for (size_t j = 0; j < representative_embeddings.size(); ++j) {
            float s = preprocessor::detail::cosine_similarity(
                representative_embeddings[i], representative_embeddings[j]);
            row.push_back(std::round(s * 1000.0f) / 1000.0f);
        }
        sim_matrix.push_back(row);
    }

    // ================================================================
    // Phase 4: Embedding generation timing vs input length
    // ================================================================
    std::cerr << "Phase 4: Embedding timing...\n";

    std::vector<std::string> timing_inputs = {
        "hi",
        "mute it",
        "mute the sound",
        "open the web browser",
        "turn down the volume please",
        "could you please mute the audio",
        "hey can you open the web browser for me",
        "would it be possible to take a screenshot of my entire screen",
        "i was wondering if perhaps you could lower the volume because it is way too loud right now",
    };

    json embedding_timing = json::array();
    for (const auto& input : timing_inputs) {
        std::vector<long long> times;
        for (int i = 0; i < EMBED_RUNS; ++i) {
            auto start = Clock::now();
            engine->generate_embedding(input);
            auto end = Clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count());
        }
        std::sort(times.begin(), times.end());
        double avg = std::accumulate(times.begin(), times.end(), 0.0)
                     / static_cast<double>(times.size());

        json entry;
        entry["input"]      = input;
        entry["word_count"] = count_words(input);
        entry["avg_us"]     = avg;
        entry["min_us"]     = times.front();
        entry["max_us"]     = times.back();
        entry["p50_us"]     = times[times.size() / 2];
        embedding_timing.push_back(entry);
    }

    // ================================================================
    // Phase 5: Tokenization timing
    // ================================================================
    std::cerr << "Phase 5: Tokenization timing...\n";

    json tokenization_timing = json::array();
    for (const auto& input : timing_inputs) {
        std::vector<long long> times;
        for (int i = 0; i < EMBED_RUNS; ++i) {
            auto start = Clock::now();
            tokenizer->encode(input);
            auto end = Clock::now();
            times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count());
        }
        std::sort(times.begin(), times.end());
        double avg = std::accumulate(times.begin(), times.end(), 0.0)
                     / static_cast<double>(times.size());

        json entry;
        entry["input"]      = input;
        entry["word_count"] = count_words(input);
        entry["avg_us"]     = avg;
        entry["p50_us"]     = times[times.size() / 2];
        entry["token_count"] = static_cast<int>(tokenizer->encode(input).size());
        tokenization_timing.push_back(entry);
    }

    // ================================================================
    // Assemble final JSON output
    // ================================================================
    json output;
    output["config"] = {
        {"threshold",            config.similarity_threshold},
        {"num_runs",             NUM_RUNS},
        {"embed_runs",           EMBED_RUNS},
        {"num_intents",          static_cast<int>(config.intents.size())},
        {"total_intent_examples", total_examples},
        {"model_path",           config.model_path},
        {"num_test_cases",       static_cast<int>(test_cases.size())}
    };
    output["routing_results"]           = routing_results;
    output["similarity_analysis"]       = similarity_analysis;
    output["intent_similarity_matrix"]  = {
        {"labels", intent_labels},
        {"matrix", sim_matrix}
    };
    output["embedding_timing"]      = embedding_timing;
    output["tokenization_timing"]   = tokenization_timing;

    std::cout << output.dump(2) << std::endl;
    std::cerr << "Benchmark complete.\n";
    return 0;
}
