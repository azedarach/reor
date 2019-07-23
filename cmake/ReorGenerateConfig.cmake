function(generate_reor_config_files)

  include(CMakePackageConfigHelpers)

  configure_package_config_file(
    "${CMAKE_MODULE_PATH}/ReorConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/${REOR_INSTALL_CMAKE_DIR}/ReorConfig.cmake"
    INSTALL_DESTINATION "${REOR_INSTALL_CMAKE_DIR}"
    PATH_VARS
    REOR_INSTALL_CMAKE_DIR
    REOR_INSTALL_INCLUDE_DIR
    )

  install(FILES "${PROJECT_BINARY_DIR}/${REOR_INSTALL_CMAKE_DIR}/ReorConfig.cmake"
    DESTINATION ${REOR_INSTALL_CMAKE_DIR})

  install(EXPORT ReorTargets
    NAMESPACE Reor::
    DESTINATION ${REOR_INSTALL_CMAKE_DIR})

  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${REOR_INSTALL_CMAKE_DIR}/ReorConfigVersion.cmake"
    VERSION "${REOR_VERSION}"
    COMPATIBILITY SameMajorVersion)

  install(FILES "${PROJECT_BINARY_DIR}/${REOR_INSTALL_CMAKE_DIR}/ReorConfigVersion.cmake"
    DESTINATION ${REOR_INSTALL_CMAKE_DIR})

endfunction()
