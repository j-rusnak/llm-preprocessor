set(RELEASE_STAGING_DIR "${CMAKE_BINARY_DIR}/release-staging")
set(RELEASE_ARTIFACT_DIR "${CMAKE_BINARY_DIR}/artifacts")

add_custom_target(stage_release_package
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RELEASE_STAGING_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:llm_preprocessor>" "${RELEASE_STAGING_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/examples" "${RELEASE_STAGING_DIR}/examples"
    COMMENT "Stage llm-preprocessor release package with runtime examples")

add_custom_command(OUTPUT "${RELEASE_ARTIFACT_DIR}/llm-preprocessor.sha256"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${RELEASE_ARTIFACT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E sha256sum "${RELEASE_STAGING_DIR}/llm-preprocessor.zip"
            > "${RELEASE_ARTIFACT_DIR}/llm-preprocessor.sha256"
    DEPENDS stage_release_package
    COMMENT "Generate release artifact checksum for package publishing")

add_custom_target(release_checksum ALL
    DEPENDS "${RELEASE_ARTIFACT_DIR}/llm-preprocessor.sha256")
