cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(REOR VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include(ReorBuildOptions)

if(NOT REOR_INSTALL_INCLUDE_DIR)
  set(REOR_INSTALL_INCLUDE_DIR include
    CACHE PATH
    "The directory relative to CMAKE_INSTALL_PREFIX where Reor headers are installed")
endif()

if(NOT REOR_INSTALL_CMAKE_DIR)
  set(REOR_INSTALL_CMAKE_DIR share/reor
    CACHE PATH
    "The directory relative to CMAKE_INSTALL_PREFIX where Reor CMake files are installed")
endif()

find_package(Eigen3 3.1 REQUIRED)

if(ENABLE_COMPILER_WARNINGS)
  include(add_cxx_flag_if_supported)
  include(CompilerWarnings)
endif()

add_library(reor INTERFACE)
target_include_directories(reor INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${REOR_INSTALL_INCLUDE_DIR}>)

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(reor INTERFACE
    ${CXX_WARNING_FLAGS})
endif()

if(TARGET Eigen3::Eigen)
  target_link_libraries(reor
    INTERFACE
    Eigen3::Eigen)
endif()

install(DIRECTORY reor DESTINATION ${REOR_INSTALL_INCLUDE_DIR})

install(TARGETS
  reor
  EXPORT ReorTargets)

add_library(Reor::reor ALIAS reor)

export(TARGETS reor
  NAMESPACE Reor::
  FILE "${PROJECT_BINARY_DIR}/${REOR_INSTALL_CMAKE_DIR}/ReorTargets.cmake")

export(PACKAGE REOR)

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test EXCLUDE_FROM_ALL)
endif()

include(ReorGenerateConfig)
generate_reor_config_files()
