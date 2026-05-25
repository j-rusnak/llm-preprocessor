cmake_minimum_required(VERSION 3.20)
project(LLMPreprocessorPackage LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(preprocessor_runtime STATIC src/runtime.cpp)
target_compile_features(preprocessor_runtime PUBLIC cxx_std_17)
target_include_directories(preprocessor_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS preprocessor_runtime
    EXPORT LLMPreprocessorTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
    ${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.23.2/lib/onnxruntime.dll
    DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_package_config_file(
    cmake/LLMPreprocessorConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/LLMPreprocessorConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LLMPreprocessor)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/LLMPreprocessorConfigVersion.cmake
    VERSION 1.0.0
    COMPATIBILITY SameMajorVersion)

set(CPACK_PACKAGE_NAME "llm-preprocessor")
set(CPACK_PACKAGE_VENDOR "LLM Preprocessor")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
include(CPack)
