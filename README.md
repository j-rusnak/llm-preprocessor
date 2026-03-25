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
| **TextSanitizer** | `text_sanitizer.hpp` | Normalizes input — lowercases, collapses whitespace, trims. |
| **Tokenizer** | `tokenizer.hpp` | WordPiece tokenizer compatible with BERT-based models. Loads a `vocab.txt` file and produces token ID sequences. |
| **EmbeddingEngine** | `embedding_engine.hpp` | Generates vector embeddings from text via ONNX Runtime inference. Accepts any ONNX embedding model. |
| **IntentRouter** | `intent_router.hpp` | Compares input embeddings against registered intents using cosine similarity. Returns a local action name if the similarity exceeds a configurable threshold. |
| **ContextGatherer** | `context_gatherer.hpp` | Fetches external context from local files (`std::fstream`) or URLs (`libcurl`). Restricted to HTTP/HTTPS with a 10 MB download limit. |
| **MemoryEngine** | `memory_engine.hpp` | SQLite-backed conversation history. Stores and retrieves recent message pairs for multi-turn context. |
| **PromptCompiler** | `prompt_compiler.hpp` | Assembles the final JSON payload (system prompt + history + context-enriched user message) ready to send to any LLM API. |

## Tech Stack

- **C++17** (strictly enforced)
- **CMake 3.15+** with **vcpkg** manifest mode
- **ONNX Runtime** — local embedding inference
- **libcurl** — HTTP fetching
- **SQLite3** — conversation memory
- **nlohmann/json** — JSON construction
- **Google Test** — unit testing

## Project Structure

```
├── CMakeLists.txt
├── vcpkg.json
├── include/
│   ├── context_gatherer.hpp
│   ├── embedding_engine.hpp
│   ├── intent_router.hpp
│   ├── memory_engine.hpp
│   ├── prompt_compiler.hpp
│   ├── text_sanitizer.hpp
│   └── tokenizer.hpp
├── src/
│   ├── main.cpp
│   ├── context_gatherer.cpp
│   ├── embedding_engine.cpp
│   ├── intent_router.cpp
│   ├── memory_engine.cpp
│   ├── prompt_compiler.cpp
│   ├── text_sanitizer.cpp
│   └── tokenizer.cpp
├── tests/
│   ├── test_intent_router.cpp
│   └── test_text_sanitizer.cpp
└── models/
    ├── model.onnx          (your ONNX embedding model)
    └── vocab.txt           (WordPiece vocabulary file)
```

## Prerequisites

- A C++17 compiler (MSVC, GCC, or Clang)
- [CMake 3.15+](https://cmake.org/)
- [vcpkg](https://vcpkg.io/) — package manager

## Build

```bash
# Clone and bootstrap vcpkg (if not already installed)
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg && bootstrap-vcpkg.bat   # Windows
cd vcpkg && ./bootstrap-vcpkg.sh  # Linux/macOS

# Configure and build (vcpkg manifest mode auto-installs dependencies)
cmake -B build -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

## Testing

```bash
cd build
ctest --output-on-failure
```

## Usage

This project is designed as **middleware** — it does not call an LLM directly. The host application is responsible for LLM inference. The preprocessor:

1. Receives raw user input
2. Sanitizes and embeds the text
3. Attempts to route to a local action (e.g., "turn down volume")
4. If no match, gathers external context and conversation history
5. Compiles a JSON payload the host app sends to its LLM

```cpp
#include "intent_router.hpp"
#include "embedding_engine.hpp"
#include "tokenizer.hpp"
#include "prompt_compiler.hpp"
#include "memory_engine.hpp"
#include "context_gatherer.hpp"

// Initialize
auto tokenizer = std::make_shared<preprocessor::Tokenizer>("models/vocab.txt");
auto engine = std::make_shared<preprocessor::EmbeddingEngine>("models/model.onnx", tokenizer);
preprocessor::IntentRouter router(0.75f, engine);
preprocessor::MemoryEngine memory("history.db");
preprocessor::PromptCompiler compiler("You are a helpful AI assistant.");

// Register local actions
router.add_intent("ACTION_DECREASE_VOLUME", "turn down the volume");

// Route input
auto matched = router.route(user_input);
if (matched) {
    // Handle locally — no LLM call needed
} else {
    // Build LLM payload
    auto context = preprocessor::ContextGatherer::fetch_url("http://example.com");
    auto history = memory.get_recent_history(5);
    std::string payload = compiler.build_payload(user_input, context, history);
    // Send payload to your LLM...
}
```

## License

See [LICENSE](LICENSE) for details.
