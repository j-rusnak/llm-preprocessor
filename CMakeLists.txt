cmake_minimum_required(VERSION 3.15)
project(LLMPreprocessor VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# --- 1. Find the Packages (Downloaded by vcpkg) ---
find_package(nlohmann_json CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# --- ONNX Runtime (official pre-built binary) ---
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-1.23.2")
    else()
        set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x86-1.23.2")
    endif()
elseif(APPLE)
    set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-universal2-1.23.2")
else()
    set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.23.2")
endif()

add_library(onnxruntime SHARED IMPORTED)
if(WIN32)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${ORT_ROOT}/lib/onnxruntime.lib"
        IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
    )
else()
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ORT_ROOT}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
    )
    if(APPLE)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ORT_ROOT}/lib/libonnxruntime.dylib"
        )
    endif()
endif()

# --- 2. Define your Core Preprocessor Library ---
#
# Phase 0 modules:
#   chat_history_store  - SQLite chat memory (replaces MemoryEngine)
#   vector_store        - HNSW ANN index for code-chunk embeddings
#   llm_tokenizer       - downstream-LLM token budgeting (heuristic for now)
#   code_chunker        - line-window chunker; tree-sitter slot in Phase 1
#   file_watcher        - efsw wrapper for incremental re-indexing
add_library(preprocessor_lib
    src/intent_router.cpp
    src/embedding_engine.cpp
    src/context_gatherer.cpp
    src/tokenizer.cpp
    src/chat_history_store.cpp
    src/vector_store.cpp
    src/llm_tokenizer.cpp
    src/code_chunker.cpp
    src/file_watcher.cpp
    src/prompt_compiler.cpp
    src/text_sanitizer.cpp
    src/config_loader.cpp
)

target_include_directories(preprocessor_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# --- Phase 0 dependencies (vcpkg manifest mode) ---
find_package(hnswlib CONFIG REQUIRED)
find_package(xxHash CONFIG REQUIRED)
find_package(efsw CONFIG REQUIRED)

# Link the external dependencies to your library
target_link_libraries(preprocessor_lib PUBLIC
    nlohmann_json::nlohmann_json
    CURL::libcurl
    onnxruntime
    unofficial::sqlite3::sqlite3
    hnswlib::hnswlib
    xxHash::xxhash
    efsw::efsw
)

# --- 3. Define the Main Executable ---
add_executable(preprocessor_app src/main.cpp)
target_link_libraries(preprocessor_app PRIVATE preprocessor_lib)

# --- 4. Define the Test Executable ---
include(GoogleTest)

add_executable(preprocessor_tests
    tests/test_text_sanitizer.cpp
    tests/test_intent_router.cpp
    tests/test_chat_history_store.cpp
    tests/test_prompt_compiler.cpp
    tests/test_context_gatherer.cpp
    tests/test_config_loader.cpp
    tests/test_tokenizer.cpp
    tests/test_vector_store.cpp
    tests/test_llm_tokenizer.cpp
    tests/test_code_chunker.cpp
    tests/test_file_watcher.cpp
    # tests/test_embedding_engine.cpp
)

target_link_libraries(preprocessor_tests PRIVATE
    preprocessor_lib
    GTest::gtest_main
)

gtest_add_tests(TARGET preprocessor_tests)

# --- 5. Benchmark executable ---
add_executable(benchmark_runner benchmarks/benchmark_runner.cpp)
target_link_libraries(benchmark_runner PRIVATE preprocessor_lib)

# --- 6. Install targets ---
include(GNUInstallDirs)

install(TARGETS preprocessor_lib
    EXPORT LLMPreprocessorTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS preprocessor_app
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT LLMPreprocessorTargets
    FILE LLMPreprocessorTargets.cmake
    NAMESPACE LLMPreprocessor::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LLMPreprocessor
)

# --- 7. Copy ONNX Runtime DLL next to executables on Windows ---
if(WIN32)
    foreach(_tgt preprocessor_app preprocessor_tests benchmark_runner)
        add_custom_command(TARGET ${_tgt} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ORT_ROOT}/lib/onnxruntime.dll"
                $<TARGET_FILE_DIR:${_tgt}>)
    endforeach()
endif()

# --- 8. Integration smoke-test runner (Phase 0 framework) ---
#
# Exercises the new modules end-to-end against a small synthetic repo. Not a
# unit test; intended to be run manually or in CI to validate the full Phase 0
# pipeline (chunk -> embed -> index -> retrieve -> tokenize).
add_executable(smoke_runner tests/smoke_runner.cpp)
target_link_libraries(smoke_runner PRIVATE preprocessor_lib)