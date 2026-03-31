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
add_library(preprocessor_lib
    src/intent_router.cpp
    src/embedding_engine.cpp
    src/context_gatherer.cpp
    src/tokenizer.cpp
    src/memory_engine.cpp
    src/prompt_compiler.cpp
    src/text_sanitizer.cpp
    src/config_loader.cpp
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
    tests/test_memory_engine.cpp
    tests/test_prompt_compiler.cpp
    tests/test_context_gatherer.cpp
    tests/test_config_loader.cpp
    tests/test_tokenizer.cpp
    tests/test_embedding_engine.cpp
)

target_link_libraries(preprocessor_tests PRIVATE
    preprocessor_lib
    GTest::gtest_main
)

gtest_add_tests(TARGET preprocessor_tests)