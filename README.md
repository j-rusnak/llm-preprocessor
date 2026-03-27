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
| **Tokenizer** | `tokenizer.hpp` | WordPiece tokenizer compatible with BERT-based models. Loads a `vocab.txt` file and produces token ID sequences. |
| **EmbeddingEngine** | `embedding_engine.hpp` | Generates vector embeddings from text via ONNX Runtime inference. Supports `.onnx` and `.ort` model formats with mean pooling over token outputs. |
| **IntentRouter** | `intent_router.hpp` | Compares input embeddings against registered intents using cosine similarity. Returns a local action name if the similarity exceeds a configurable threshold. |
| **ContextGatherer** | `context_gatherer.hpp` | Fetches external context from URLs (`libcurl`) and extracts URLs from user input. Restricted to HTTP/HTTPS with a 10 MB download limit. |
| **MemoryEngine** | `memory_engine.hpp` | SQLite-backed conversation history. Stores and retrieves recent message pairs for multi-turn context. |
| **PromptCompiler** | `prompt_compiler.hpp` | Assembles the final JSON payload (system prompt + history + context-enriched user message) ready to send to any LLM API. |

## Tech Stack

- **C++17** (strictly enforced)
- **CMake 3.15+** with **vcpkg** manifest mode
- **ONNX Runtime 1.23.2** — local embedding inference (official pre-built binary)
- **libcurl** — HTTP fetching
- **SQLite3** — conversation memory
- **nlohmann/json** — JSON construction
- **Google Test** — unit testing (32 tests across 6 suites)

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
│   ├── test_intent_router.cpp
│   ├── test_memory_engine.cpp
│   ├── test_prompt_compiler.cpp
│   └── test_text_sanitizer.cpp
├── models/
│   ├── model.onnx / model.ort  (ONNX embedding model)
│   └── vocab.txt               (WordPiece vocabulary)
└── onnxruntime-win-x64-1.23.2/ (pre-built ONNX Runtime SDK)
```

## Prerequisites

- A C++17 compiler (MSVC recommended on Windows)
- [CMake 3.15+](https://cmake.org/)
- [vcpkg](https://vcpkg.io/) — package manager
- **ONNX Runtime 1.23.2** — download the [official pre-built release](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2) and extract it to the project root as `onnxruntime-win-x64-1.23.2/`
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

```bash
cd build
ctest --output-on-failure
```

All 32 tests across 6 suites should pass:
- **TextSanitizerTest** (5) — whitespace, case normalization
- **IntentRouterTest** (5) — cosine similarity edge cases
- **MemoryEngineTest** (5) — SQLite CRUD, ordering, limits
- **PromptCompilerTest** (4) — JSON payload construction
- **UrlExtractionTest** (6) — URL parsing from text
- **ConfigLoaderTest** (7) — config validation and defaults

## Configuration

The preprocessor is driven by a `config.json` file:

```json
{
    "model_path": "models/model.ort",
    "vocab_path": "models/vocab.txt",
    "db_path": "history.db",
    "system_prompt": "You are a helpful AI assistant.",
    "similarity_threshold": 0.75,
    "history_limit": 10,
    "intents": [
        { "name": "ACTION_DECREASE_VOLUME", "example": "turn down the volume" },
        { "name": "ACTION_OPEN_BROWSER",    "example": "open the web browser" }
    ]
}
```

| Key | Description | Default |
|---|---|---|
| `model_path` | Path to the ONNX/ORT embedding model | *(required)* |
| `vocab_path` | Path to the WordPiece vocab file | *(required)* |
| `db_path` | SQLite database file for conversation history | `"history.db"` |
| `system_prompt` | System message prepended to every LLM payload | `"You are a helpful assistant."` |
| `similarity_threshold` | Cosine similarity cutoff for intent matching (0.0–1.0) | `0.75` |
| `history_limit` | Max conversation turns to include in payload | `10` |
| `intents` | Array of `{name, example}` pairs for semantic routing | `[]` |

## Running

```powershell
.\build\preprocessor_app.exe                  # uses config.json
.\build\preprocessor_app.exe my_config.json   # custom config path
```

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

// Register intents
for (const auto& [name, example] : config.intents) {
    router.add_intent(name, example);
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

## License

See [LICENSE](LICENSE) for details.
