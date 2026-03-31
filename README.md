# LLM Preprocessor

A high-performance C++17 middleware that intercepts user inputs, routes simple commands to local OS actions via semantic matching, and enriches complex queries with external context before handing them off to an LLM. The goal is to minimize expensive API calls by bypassing the LLM entirely for tasks that can be resolved locally.

## Architecture

```
User Input
    │
    ▼
TextSanitizer ──► Tokenizer ──► EmbeddingEngine (ONNX Runtime)
                                        │
                                        ▼
                                  IntentRouter
                                   ╱        ╲
                          Match found     No match
                              │               │
                              ▼               ▼
                       Local Action     ContextGatherer ──► MemoryEngine
                       (skip LLM)              │                  │
                                               ▼                  ▼
                                           PromptCompiler
                                               │
                                               ▼
                                      JSON Payload (to host app)
```

## Pipeline Modules

| Module | Header | Description |
|---|---|---|
| **ConfigLoader** | `config_loader.hpp` | Loads and validates JSON configuration (model paths, thresholds, intents). |
| **TextSanitizer** | `text_sanitizer.hpp` | Normalizes input — lowercases, collapses whitespace, trims. |
| **Tokenizer** | `tokenizer.hpp` | WordPiece tokenizer compatible with BERT-based models. Dynamically resolves `[CLS]`/`[SEP]`/`[UNK]` IDs from the vocabulary and truncates at 512 tokens. |
| **EmbeddingEngine** | `embedding_engine.hpp` | Generates vector embeddings from text via ONNX Runtime inference. Supports `.onnx` and `.ort` model formats with attention-mask-aware mean pooling. |
| **IntentRouter** | `intent_router.hpp` | Compares input embeddings against registered intents using cosine similarity. Strips common stop words and uses sliding-window subphrase extraction (capped at 15 ONNX inferences) to match commands in longer sentences. Returns a `RouteResult` with intent name and confidence score. Supports an action callback, plus runtime `remove_intent()` / `clear_intents()`. Thread-safe via `EmbeddingEngine` mutex. |
| **ContextGatherer** | `context_gatherer.hpp` | Fetches external context from URLs (`libcurl`, RAII-wrapped handles) and extracts URLs from user input. Restricted to HTTP/HTTPS with a streaming-enforced 10 MB download limit. |
| **MemoryEngine** | `memory_engine.hpp` | SQLite-backed conversation history. Stores, retrieves, updates (`update_last_message`), clears, and auto-prunes messages. Supports move semantics. |
| **PromptCompiler** | `prompt_compiler.hpp` | Assembles the final JSON payload. `build_payload()` returns a messages-only string (backward-compatible). `build_payload_json()` returns a `nlohmann::json` object with optional `model`/`temperature`/`max_tokens` for a complete API request body. |

## Tech Stack

- **C++17** (strictly enforced)
- **CMake 3.15+** with **vcpkg** manifest mode
- **ONNX Runtime 1.23.2** — local embedding inference (official pre-built binary)
- **libcurl** — HTTP fetching
- **SQLite3** — conversation memory
- **nlohmann/json** — JSON construction
- **Google Test** — unit testing (76 tests across 8 suites)

## Project Structure

```
├── CMakeLists.txt
├── vcpkg.json
├── config.json                 (runtime configuration)
├── include/
│   ├── config_loader.hpp
│   ├── context_gatherer.hpp
│   ├── embedding_engine.hpp
│   ├── intent_router.hpp
│   ├── memory_engine.hpp
│   ├── prompt_compiler.hpp
│   ├── text_sanitizer.hpp
│   └── tokenizer.hpp
├── src/
│   ├── main.cpp
│   ├── config_loader.cpp
│   ├── context_gatherer.cpp
│   ├── embedding_engine.cpp
│   ├── intent_router.cpp
│   ├── memory_engine.cpp
│   ├── prompt_compiler.cpp
│   ├── text_sanitizer.cpp
│   └── tokenizer.cpp
├── tests/
│   ├── test_config_loader.cpp
│   ├── test_context_gatherer.cpp
│   ├── test_embedding_engine.cpp
│   ├── test_intent_router.cpp
│   ├── test_memory_engine.cpp
│   ├── test_prompt_compiler.cpp
│   ├── test_text_sanitizer.cpp
│   └── test_tokenizer.cpp
├── benchmarks/
│   ├── benchmark_runner.cpp    (C++ benchmark executable)
│   ├── visualize.py            (Python chart generator)
│   ├── run_benchmarks.ps1      (PowerShell orchestration)
│   └── results/                (generated JSON + PNGs)
├── models/
│   ├── model.onnx / model.ort  (ONNX embedding model)
│   └── vocab.txt               (WordPiece vocabulary)
└── onnxruntime-win-x64-1.23.2/ (pre-built ONNX Runtime SDK)
```

## Prerequisites

- A C++17 compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
- [CMake 3.15+](https://cmake.org/)
- [vcpkg](https://vcpkg.io/) — package manager
- **ONNX Runtime 1.23.2** — download the [official pre-built release](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2) and extract it to the project root (CMake auto-selects the platform-appropriate directory name)
- A **BERT-based ONNX embedding model** (e.g., `all-MiniLM-L6-v2`)

## Setting Up the Model

The preprocessor requires a sentence-embedding ONNX model and its WordPiece vocabulary.

```bash
# Install Python dependencies
pip install optimum[onnxruntime] sentence-transformers

# Export model to ONNX format
python -c "
from optimum.onnxruntime import ORTModelForFeatureExtraction
m = ORTModelForFeatureExtraction.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', export=True)
m.save_pretrained('models')
"

# Download the vocabulary file
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
t.save_vocabulary('models')
"
```

This creates `models/model.onnx` (~80 MB) and `models/vocab.txt` (~232 KB).

> **Tip:** Add `models/` to your `.gitignore` — don't commit large binary files.

## Build

Building requires a **Visual Studio Developer Command Prompt** (or equivalent) so the MSVC environment variables are set.

```powershell
# Open a VS Developer PowerShell, then:
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build

# Copy the ONNX Runtime DLL next to the built executables
copy onnxruntime-win-x64-1.23.2\lib\onnxruntime.dll build\
```

## Testing

See [Running the Test Suite](#running-the-test-suite) below for the full guide.

```bash
cd build
ctest --output-on-failure
```

## Configuration

The preprocessor is driven by a `config.json` file:

```json
{
    "model_path": "models/model.ort",
    "vocab_path": "models/vocab.txt",
    "db_path": "history.db",
    "system_prompt": "You are a helpful AI assistant.",
    "similarity_threshold": 0.65,
    "history_limit": 10,
    "intents": [
        {
            "name": "ACTION_DECREASE_VOLUME",
            "examples": ["turn down the volume", "lower the volume", "make it quieter"]
        },
        {
            "name": "ACTION_OPEN_BROWSER",
            "examples": ["open the web browser", "launch a browser", "start the browser"]
        }
    ]
}
```

Each intent supports multiple synonym examples via the `"examples"` array. The router registers every example as a separate embedding — the best match across all examples determines the intent. A single `"example"` string is also accepted for backward compatibility.

| Key | Description | Default |
|---|---|---|
| `model_path` | Path to the ONNX/ORT embedding model | *(required)* |
| `vocab_path` | Path to the WordPiece vocab file | *(required)* |
| `db_path` | SQLite database file for conversation history | `"history.db"` |
| `system_prompt` | System message prepended to every LLM payload | `"You are a helpful assistant."` |
| `similarity_threshold` | Cosine similarity cutoff for intent matching (0.0–1.0) | `0.65` |
| `history_limit` | Max conversation turns to include in payload | `10` |
| `intents` | Array of `{name, examples}` objects for semantic routing | `[]` |
| `api_model` | *(Optional)* Model name for complete API payload (e.g., `"gpt-4"`) | — |
| `api_endpoint` | *(Optional)* API endpoint URL | — |
| `temperature` | *(Optional)* Sampling temperature (0.0–2.0) | — |
| `max_tokens` | *(Optional)* Max tokens in LLM response | — |

## Running

```powershell
.\build\preprocessor_app.exe                  # uses config.json
.\build\preprocessor_app.exe my_config.json   # custom config path
.\build\preprocessor_app.exe --help            # show usage
.\build\preprocessor_app.exe --version         # show version
```

When `api_model` is set in config, payloads are emitted as complete API request bodies (`{model, messages, temperature, max_tokens}`). Without it, the old messages-only format is used.

The interactive loop accepts free-text input:

```
LLM Preprocessor ready. Type your input (or 'quit' to exit).

> open the web browser
[ACTION] ACTION_OPEN_BROWSER

> what is the meaning of life?

=== LLM Payload ===
[
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "what is the meaning of life?"}
]

> quit
```

- Inputs matching a registered intent trigger a **local action** (no LLM call).
- Unmatched inputs produce a **JSON payload** for the host application to forward to an LLM.
- The terminal display shows a clean payload without conversation history to reduce clutter; the full history is still included when the payload is sent to the LLM.
- URLs in the input are automatically fetched and injected as RAG context.
- If model files are missing, semantic routing is gracefully disabled and only the payload path is active.

## Usage as a Library

```cpp
#include "config_loader.hpp"
#include "intent_router.hpp"
#include "embedding_engine.hpp"
#include "tokenizer.hpp"
#include "prompt_compiler.hpp"
#include "memory_engine.hpp"
#include "context_gatherer.hpp"
#include "text_sanitizer.hpp"

// Load config
auto config = preprocessor::ConfigLoader::load("config.json");

// Initialize pipeline
auto tokenizer = std::make_shared<preprocessor::Tokenizer>(config.vocab_path);
auto engine = std::make_shared<preprocessor::EmbeddingEngine>(config.model_path, tokenizer);
preprocessor::IntentRouter router(config.similarity_threshold, engine);
preprocessor::MemoryEngine memory(config.db_path);
preprocessor::PromptCompiler compiler(config.system_prompt);

// Register intents (multiple synonym examples per intent)
for (const auto& [name, examples] : config.intents) {
    for (const auto& example : examples) {
        router.add_intent(name, example);
    }
}

// Process input
std::string input = preprocessor::TextSanitizer::sanitize(raw_input);
auto matched = router.route(input);
if (matched) {
    // Handle locally — no LLM call needed
} else {
    auto urls = preprocessor::ContextGatherer::extract_urls(raw_input);
    std::string context;
    for (const auto& url : urls) {
        context += preprocessor::ContextGatherer::fetch_url(url);
    }
    auto history = memory.get_recent_history(config.history_limit);
    std::string payload = compiler.build_payload(input, context, history);
    // Send payload to your LLM...
}
```

## Benchmarks & Visualizations

The project includes a full benchmarking and visualization pipeline for evaluating the semantic router's latency, accuracy, and similarity characteristics. This is useful for presentations, reports, and tuning.

### Overview

| Component | Location | Purpose |
|---|---|---|
| **Benchmark Runner** | `benchmarks/benchmark_runner.cpp` | C++ executable that runs 31 test prompts through the routing pipeline, collecting latency, accuracy, similarity scores, and embedding timing. Outputs structured JSON. |
| **Visualization Script** | `benchmarks/visualize.py` | Python script that reads the JSON output and generates 10 presentation-ready PNG charts. |
| **Orchestration Script** | `benchmarks/run_benchmarks.ps1` | PowerShell script that runs both steps end-to-end. |
| **Results Directory** | `benchmarks/results/` | Output directory for JSON data and PNG charts. |

### Prerequisites

Ensure the project is already built (see [Build](#build) above), then install the Python dependencies:

```bash
pip install matplotlib numpy
```

### Quick Start (All-In-One)

From the project root, run the PowerShell orchestration script:

```powershell
.\benchmarks\run_benchmarks.ps1
```

This will:
1. Verify `benchmark_runner.exe` exists in `build/`
2. Run the C++ benchmark → `benchmarks/results/benchmark_data.json`
3. Generate all 10 PNG charts → `benchmarks/results/`

### Step-by-Step (Manual)

#### 1. Build the benchmark executable

The benchmark target is included in the CMake build. If you haven't built yet:

```powershell
# From a VS Developer PowerShell:
cmake -B build -G Ninja `
    -DCMAKE_BUILD_TYPE=Debug `
    -DCMAKE_TOOLCHAIN_FILE="C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake"

cmake --build build
```

Verify the executable exists:

```powershell
Test-Path .\build\benchmark_runner.exe   # should be True
```

#### 2. Run the benchmark

The benchmark must be run from the **project root** so it can find `config.json` and `models/`:

```powershell
# Create results directory
New-Item -ItemType Directory -Path benchmarks\results -Force | Out-Null

# Run and capture JSON output
.\build\benchmark_runner.exe > benchmarks\results\benchmark_data.json
```

The benchmark runs 5 phases:
1. **Routing Benchmarks** — 31 test inputs × 10 runs each, measuring latency and correctness
2. **Similarity Analysis** — per-input similarity scores against all 8 intents
3. **Intent Similarity Matrix** — 8×8 cosine similarity between intents
4. **Embedding Timing** — 9 different input lengths × 20 runs each
5. **Tokenization Timing** — encoding speed for the same 9 inputs

Progress is printed to `stderr`; JSON data goes to `stdout`.

#### 3. Generate visualizations

```powershell
python benchmarks\visualize.py benchmarks\results\benchmark_data.json benchmarks\results
```

Arguments:
- **Arg 1** (required): Path to the JSON data file
- **Arg 2** (optional): Output directory for PNGs (default: `benchmarks/results`)

### Generated Charts

| # | File | Chart Type | What It Shows |
|---|------|-----------|---------------|
| 1 | `01_latency_by_category.png` | Bar chart (± std) | Average routing latency across 5 input categories |
| 2 | `02_latency_vs_words.png` | Scatter + trend line | How routing latency scales with input word count |
| 3 | `03_accuracy_by_category.png` | Bar chart | Percentage of correctly routed inputs per category |
| 4 | `04_score_distribution.png` | Box plot | Distribution of best cosine similarity scores, with threshold overlay |
| 5 | `05_similarity_heatmap.png` | Heatmap (8×8) | Cosine similarity between all registered intents (with formula) |
| 6 | `06_threshold_curve.png` | Multi-line plot | Accuracy, Precision, Recall, and F1 across thresholds 0.40–0.95 |
| 7 | `07_embedding_timing.png` | Bar chart (± error) | ONNX embedding generation time vs input length |
| 8 | `08_api_comparison.png` | Log-scale bars | Local routing latency vs estimated LLM API round-trip times |
| 9 | `09_per_input_scores.png` | Heatmap (31×8) | Every test input's similarity score against every intent |
| 10 | `10_summary_dashboard.png` | Text dashboard | Configuration, accuracy, latency (P50/P95), throughput, optimal threshold |

### Test Categories

The 31 benchmark inputs span 5 complexity levels:

| Category | Count | Examples |
|---|---|---|
| **Direct Commands** | 8 | `"mute the sound"`, `"turn up the volume"`, `"open a file"` |
| **Noisy Commands** | 6 | `"hey bro can you mute that"`, `"yo dude turn up the volume please"` |
| **Complex Sentences** | 6 | `"i was wondering if you could perhaps mute the audio"` |
| **Non-Matching** | 6 | `"what is the meaning of life"`, `"tell me about quantum physics"` |
| **Edge Cases** | 5 | `"mute"` (1 word), `"volume"`, `"mute the sound and open the browser"` (multi-intent) |

### Benchmark JSON Schema

The JSON output contains these top-level sections:

```
{
  "config":                  { ... },  // threshold, num_intents, num_runs, etc.
  "routing_results":         [ ... ],  // 31 entries with latency, correctness, scores
  "similarity_analysis":     [ ... ],  // per-input scores against all intents
  "intent_similarity_matrix": { ... }, // 8×8 matrix with labels
  "embedding_timing":        [ ... ],  // 9 entries with avg/min/max/p50
  "tokenization_timing":     [ ... ]   // 9 entries with timing + token count
}
```

Each routing result includes: `category`, `input`, `expected_intent`, `matched_intent`, `score`, `correct`, `word_count`, and `latency` object with `avg_us`, `min_us`, `max_us`, `p50_us`, `p95_us`.

## Running the Test Suite

### Quick Run

```powershell
cd build
ctest --output-on-failure
```

### Verbose Output

```powershell
cd build
ctest --output-on-failure -V
```

### Run a Single Test Suite

```powershell
cd build
.\preprocessor_tests.exe --gtest_filter="TextSanitizerTest.*"
.\preprocessor_tests.exe --gtest_filter="IntentRouterTest.*"
.\preprocessor_tests.exe --gtest_filter="MemoryEngineTest.*"
.\preprocessor_tests.exe --gtest_filter="PromptCompilerTest.*"
.\preprocessor_tests.exe --gtest_filter="UrlExtractionTest.*"
.\preprocessor_tests.exe --gtest_filter="ConfigLoaderTest.*"
.\preprocessor_tests.exe --gtest_filter="TokenizerTest.*"
.\preprocessor_tests.exe --gtest_filter="EmbeddingEngineTest.*"
```

### Run a Single Test

```powershell
.\preprocessor_tests.exe --gtest_filter="ConfigLoaderTest.LoadValidConfig"
```

### List All Tests

```powershell
.\preprocessor_tests.exe --gtest_list_tests
```

### Test Suites

| Suite | Tests | What It Covers |
|---|---|---|
| **TextSanitizerTest** | 5 | Whitespace collapsing, case normalization, trimming |
| **IntentRouterTest** | 5 | Cosine similarity routing, edge cases, empty/identical embeddings |
| **MemoryEngineTest** | 10 | SQLite CRUD, ordering, history limits, move semantics, update, clear, prune |
| **PromptCompilerTest** | 7 | JSON payload construction, `build_payload_json`, API params |
| **UrlExtractionTest** | 6 | URL detection in text (http/https, mixed content) |
| **ConfigLoaderTest** | 18 | Config validation, defaults, multi-example parsing, backward compat, bounds checking |
| **TokenizerTest** | 12 | WordPiece encoding, special tokens, truncation, subwords |
| **EmbeddingEngineTest** | 13 | Shape, normalization, similarity, multi-example routing, sliding-window, stop-words |

**Total: 76 unit tests + 11 integration tests (EmbeddingEngine) = 87 tests**

## Complete Workflow Reference

A full build-test-benchmark-visualize cycle from a clean state:

```powershell
# 0. Open a VS Developer PowerShell (MSVC environment)
& "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64

# 1. Configure
cmake -B build -G Ninja `
    -DCMAKE_BUILD_TYPE=Debug `
    -DCMAKE_TOOLCHAIN_FILE="C:/Users/you/vcpkg/scripts/buildsystems/vcpkg.cmake"

# 2. Build everything (app + tests + benchmark)
cmake --build build

# 3. Copy ONNX Runtime DLL (Windows only)
Copy-Item onnxruntime-win-x64-1.23.2\lib\onnxruntime.dll build\

# 4. Run the test suite
cd build
ctest --output-on-failure
cd ..

# 5. Run the benchmark
New-Item -ItemType Directory -Path benchmarks\results -Force | Out-Null
.\build\benchmark_runner.exe > benchmarks\results\benchmark_data.json

# 6. Generate visualizations
pip install matplotlib numpy   # first time only
python benchmarks\visualize.py benchmarks\results\benchmark_data.json benchmarks\results

# 7. Run the application
.\build\preprocessor_app.exe config.json
```

## License

See [LICENSE](LICENSE) for details.
