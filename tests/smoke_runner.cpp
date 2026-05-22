// Phase 0 end-to-end smoke runner.
//
// Drives every new module against an in-memory synthetic "repo" and prints
// PASS/FAIL lines. This is the manual integration framework the user can run
// to validate the full Phase 0 pipeline (chunk -> hash -> index -> retrieve
// -> tokenize -> watch).
//
// It does NOT depend on the ONNX model — embeddings are simulated with a
// deterministic hash-based projection so the smoke runner stays fast and
// hermetic. Real-embedding tests live in test_embedding_engine.cpp.
//
// Usage:
//   ./smoke_runner             # run all
//   ./smoke_runner --verbose   # extra detail per stage

#include "chat_history_store.hpp"
#include "code_chunker.hpp"
#include "file_watcher.hpp"
#include "llm_tokenizer.hpp"
#include "vector_store.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Stats {
    int passed = 0;
    int failed = 0;
};

void report(Stats& s, const std::string& name, bool ok, const std::string& detail = "") {
    if (ok) ++s.passed; else ++s.failed;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
    if (!detail.empty()) std::cout << " - " << detail;
    std::cout << "\n";
}

// Deterministic 256-dim projection from chunk-id -> unit vector. Stands in
// for a real embedding model so the smoke runner is hermetic.
std::vector<float> fake_embed(std::uint64_t seed, std::size_t dim = 64) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(dim);
    float sq = 0.0f;
    for (auto& x : v) { x = d(rng); sq += x * x; }
    const float n = std::sqrt(sq);
    if (n > 0.0f) for (auto& x : v) x /= n;
    return v;
}

std::vector<std::pair<std::string, std::string>> make_fake_repo() {
    return {
        {"src/math.cpp",
         "int add(int a, int b) { return a + b; }\n"
         "int sub(int a, int b) { return a - b; }\n"
         "int mul(int a, int b) { return a * b; }\n"},
        {"src/io.cpp",
         "void read_file(const std::string& p) { /* ... */ }\n"
         "void write_file(const std::string& p) { /* ... */ }\n"},
        {"include/math.hpp",
         "#pragma once\nint add(int, int);\nint sub(int, int);\nint mul(int, int);\n"},
    };
}

void stage_chunker(Stats& s, bool verbose) {
    std::cout << "[chunker]\n";
    preprocessor::LineWindowChunker chunker(/*window*/ 3, /*overlap*/ 1);
    std::size_t total = 0;
    for (const auto& [path, src] : make_fake_repo()) {
        auto chunks = chunker.chunk(path, src);
        if (verbose) {
            std::cout << "    " << path << " -> " << chunks.size() << " chunks\n";
        }
        total += chunks.size();
    }
    report(s, "chunker produced chunks", total > 0,
           std::to_string(total) + " total");
}

void stage_vector_index_and_search(Stats& s, bool verbose) {
    std::cout << "[vector_store + chunker]\n";
    preprocessor::LineWindowChunker chunker(3, 1);
    preprocessor::VectorStore store(64, 1024);

    std::vector<preprocessor::CodeChunk> all;
    for (const auto& [path, src] : make_fake_repo()) {
        auto cs = chunker.chunk(path, src);
        for (auto& c : cs) {
            store.add(c.id, fake_embed(c.id));
        }
        all.insert(all.end(), cs.begin(), cs.end());
    }

    report(s, "indexed all chunks", store.size() == all.size(),
           "indexed " + std::to_string(store.size()));

    // Self-recall: query each chunk -> top hit should be itself.
    int recall = 0;
    for (const auto& c : all) {
        auto hits = store.search(fake_embed(c.id), 1);
        if (!hits.empty() && hits.front().id == c.id) ++recall;
    }
    report(s, "self-recall @ 1", recall == static_cast<int>(all.size()),
           std::to_string(recall) + "/" + std::to_string(all.size()));

    // Persistence round-trip.
    auto tmp = fs::temp_directory_path() / "llm_pp_smoke_index.bin";
    store.save(tmp.string());
    try {
        preprocessor::VectorStore reloaded(64, 1024, tmp.string());
        bool ok = reloaded.size() == store.size();
        if (ok && !all.empty()) {
            auto hits = reloaded.search(fake_embed(all.front().id), 1);
            ok = !hits.empty() && hits.front().id == all.front().id;
        }
        report(s, "persisted index round-trips", ok);
    } catch (const std::exception& e) {
        report(s, "persisted index round-trips", false, e.what());
    }
    std::error_code ec;
    fs::remove(tmp, ec);
    (void)verbose;
}

void stage_history(Stats& s, bool) {
    std::cout << "[chat_history_store]\n";
    try {
        preprocessor::ChatHistoryStore store(":memory:");
        store.add_message("user", "fix the null deref");
        store.add_message("assistant", "patched it");
        auto h = store.get_recent_history(10);
        report(s, "history round-trip", h.size() == 2 && h[1].second == "patched it");
    } catch (const std::exception& e) {
        report(s, "history round-trip", false, e.what());
    }
}

void stage_tokenizer(Stats& s, bool) {
    std::cout << "[llm_tokenizer]\n";
    preprocessor::HeuristicLLMTokenizer tk;
    auto small = tk.count_tokens("hello world");
    auto big = tk.count_tokens(std::string(4000, 'x'));
    report(s, "tokenizer monotonic", big > small,
           "small=" + std::to_string(small) + " big=" + std::to_string(big));
}

void stage_file_watcher(Stats& s, bool verbose) {
    std::cout << "[file_watcher]\n";
    auto dir = fs::temp_directory_path() /
               ("llm_pp_smoke_fw_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);

    std::atomic<int> events{0};
    preprocessor::FileWatcher watcher;
    long id = watcher.add_watch(dir.string(), [&](const preprocessor::FileEvent& ev) {
        events.fetch_add(1, std::memory_order_relaxed);
        if (verbose) std::cout << "    event: " << ev.path << "\n";
    });
    if (id < 0) {
        report(s, "file_watcher attaches", false, "add_watch returned " + std::to_string(id));
        return;
    }

    for (int i = 0; i < 3; ++i) {
        std::ofstream(dir / ("f" + std::to_string(i) + ".txt")) << i;
    }
    for (int i = 0; i < 30 && events.load() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    watcher.remove_watch(id);
    report(s, "file_watcher fires", events.load() > 0,
           std::to_string(events.load()) + " events");

    std::error_code ec;
    fs::remove_all(dir, ec);
}

} // namespace

int main(int argc, char** argv) {
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verbose") == 0 || std::strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
    }

    std::cout << "=== LLM Preprocessor :: Phase 0 Smoke Runner ===\n\n";

    Stats s;
    stage_chunker(s, verbose);
    stage_vector_index_and_search(s, verbose);
    stage_history(s, verbose);
    stage_tokenizer(s, verbose);
    stage_file_watcher(s, verbose);

    std::cout << "\nSummary: " << s.passed << " passed, " << s.failed << " failed.\n";
    return s.failed == 0 ? 0 : 1;
}
