# Production Packaging Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean generated artifacts out of version control and make the project easier to install, package, and run from a safe example config.

**Architecture:** Keep runtime assets local and ignored, install the compiled app/library plus ONNX Runtime redistributables through CMake, and ship a conservative loopback-only config example. Keep retrieval improvements to focused eval fixtures that follow the existing `GraphAwareRetriever` and `Effectiveness_Retrieval` test patterns.

**Tech Stack:** C++17, CMake, Google Test, nlohmann/json, vcpkg dependencies, ONNX Runtime prebuilt binaries.

---

### Task 1: Safe Example Config

**Files:**
- Create: `config.example.json`
- Modify: `tests/test_config_loader.cpp`

- [x] **Step 1: Write the failing test**

Add a `ConfigLoaderTest.LoadsProductionExampleConfig` test that locates `config.example.json` from either the repo root or the CTest build working directory, loads it with `ConfigLoader::load`, and asserts loopback binding, auth/rate-limit settings, request-size limit, model-calibrated token accounting, symbol graph settings, prompt rewriter settings, and no unsafe remote proxy flag.

- [x] **Step 2: Run test to verify it fails**

Run: `.\build\preprocessor_tests.exe --gtest_filter=ConfigLoaderTest.LoadsProductionExampleConfig`

Expected: FAIL because `config.example.json` does not exist yet.

- [x] **Step 3: Add the config example**

Create `config.example.json` with non-secret placeholders, loopback proxy binding, `X-Preprocessor` auth guidance through placeholder bearer tokens, rate limiting, bounded request size, cache/vector export limits, model-calibrated token metrics, symbol graph retrieval, structural fast path, and heuristic prompt rewriting.

- [x] **Step 4: Run test to verify it passes**

Run: `.\build\preprocessor_tests.exe --gtest_filter=ConfigLoaderTest.LoadsProductionExampleConfig`

Expected: PASS.

### Task 2: CMake Install Package

**Files:**
- Modify: `CMakeLists.txt`
- Modify: `cmake/LLMPreprocessorConfig.cmake.in`
- Modify: `README.md`
- Modify: `docs/TESTING.md`

- [x] **Step 1: Capture current install gap**

Run: `cmake --install build --prefix build\install-before`

Expected: install completes but `build\install-before\lib\cmake\LLMPreprocessor\LLMPreprocessorConfig.cmake` is missing.

- [x] **Step 2: Add package config and ONNX Runtime install wiring**

Use `CMakePackageConfigHelpers` to generate and install `LLMPreprocessorConfig.cmake` and `LLMPreprocessorConfigVersion.cmake`. Install ONNX Runtime shared library files into `${CMAKE_INSTALL_BINDIR}` and import libraries into `${CMAKE_INSTALL_LIBDIR}` on Windows. Define an imported `onnxruntime` target in the package config before loading exported targets.

- [x] **Step 3: Run install smoke check**

Run: `cmake --build build` then `cmake --install build --prefix build\install-check`

Expected: package config files are installed and ONNX Runtime runtime files are present.

### Task 3: Tracked Artifact Cleanup

**Files:**
- Git index only for ignored runtime artifacts.

- [x] **Step 1: List tracked ignored files**

Run: `git ls-files -ci --exclude-standard`

Expected: generated/runtime assets such as `history.db`, `models/*`, `onnxruntime-win-x64-1.23.2/*`, `onnxruntime.zip`, and test output files are listed.

- [x] **Step 2: Untrack ignored files without deleting local copies**

Run: `git rm --cached` for the listed ignored files.

Expected: files remain on disk and appear as deletions in git status.

### Task 4: Retrieval Eval Continuation

**Files:**
- Modify: `tests/test_effectiveness.cpp`
- Optionally modify: `tests/test_graph_aware_retriever.cpp`

- [x] **Step 1: Add one focused failing eval fixture**

Add a test that uses a realistic cross-file language-specific query shape not already covered by the recent graph-aware ranking work.

- [x] **Step 2: Implement only if the test exposes a small ranking gap**

If the test already passes, keep it as additional coverage. If it fails, make the smallest graph-ranking adjustment that preserves existing tests.

### Task 5: Verification

**Files:**
- No source changes.

- [x] **Step 1: Build**

Run: `cmake --build build`

Expected: build succeeds.

- [x] **Step 2: Focused tests**

Run focused config and retrieval filters touched by this work.

Expected: all focused tests pass.

- [x] **Step 3: Full gates**

Run: `ctest --test-dir build --output-on-failure`, `.\build\smoke_runner.exe`, and `.\build\effectiveness_runner.exe`

Expected: all pass.
