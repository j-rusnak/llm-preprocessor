include(CMakePackageConfigHelpers)

find_package(LLMPreprocessor CONFIG REQUIRED)

add_executable(package_smoke main.cpp)
target_link_libraries(package_smoke PRIVATE LLMPreprocessor::preprocessor_lib)

if(WIN32)
    message(STATUS "Checking ONNX Runtime install redistributable beside package smoke app")
endif()
