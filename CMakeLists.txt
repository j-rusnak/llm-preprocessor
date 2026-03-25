cmake_minimum_required(VERSION 3.15)
project(LLMPreprocessor VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# --- 1. Find the Packages (Downloaded by vcpkg) ---
find_package(nlohmann_json CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# --- 2. Define your Core Preprocessor Library ---
add_library(preprocessor_lib
    src/intent_router.cpp
    src/embedding_engine.cpp
    src/context_gatherer.cpp
    src/tokenizer.cpp
    src/memory_engine.cpp
    src/prompt_compiler.cpp
    src/text_sanitizer.cpp
)

target_include_directories(preprocessor_lib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link the external dependencies to your library
target_link_libraries(preprocessor_lib PUBLIC
    nlohmann_json::nlohmann_json
    CURL::libcurl
    onnxruntime
    unofficial::sqlite3::sqlite3
)

# --- 3. Define the Main Executable ---
add_executable(preprocessor_app src/main.cpp)
target_link_libraries(preprocessor_app PRIVATE preprocessor_lib)

# --- 4. Define the Test Executable ---
include(GoogleTest)

add_executable(preprocessor_tests
    tests/test_text_sanitizer.cpp
    tests/test_intent_router.cpp
)

target_link_libraries(preprocessor_tests PRIVATE
    preprocessor_lib
    GTest::gtest_main
)

gtest_discover_tests(preprocessor_tests)