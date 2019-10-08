cmake_minimum_required(VERSION 3.1.3 FATAL_ERROR)

project(Reor)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include(ReorBuildOptions)

if(NOT REOR_INSTALL_INCLUDE_DIR)
  set(REOR_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}/reor_cxx"
    CACHE PATH
    "The directory relative to CMAKE_INSTALL_PREFIX where Reor headers are installed")
endif()

find_package(Eigen3 3.1 REQUIRED)

if(ENABLE_COMPILER_WARNINGS)
  include(add_cxx_flag_if_supported)
  include(CompilerWarnings)
endif()

set(REOR_CATCH_INCLUDE_DIR lib/catch/include)
set(REOR_BINDINGS_SOURCE_DIR src/reor)
set(REOR_CXX_SOURCE_DIR ${BINDINGS_SOURCE_DIR}/reor_cxx)
set(REOR_TEST_DIR tests)

set(REOR_CXX_HEADERS
  ${REOR_CXX_SOURCE_DIR}/backends/eigen_backend.hpp
  ${REOR_CXX_SOURCE_DIR}/backends/eigen_type_traits.hpp
  ${REOR_CXX_SOURCE_DIR}/backend_interface.hpp
  ${REOR_CXX_SOURCE_DIR}/backends.hpp
  ${REOR_CXX_SOURCE_DIR}/furthest_sum.hpp
  ${REOR_CXX_SOURCE_DIR}/gpnh_l2_spa.hpp
  ${REOR_CXX_SOURCE_DIR}/kernel_aa.hpp
  ${REOR_CXX_SOURCE_DIR}/numerics_helpers.hpp
  ${REOR_CXX_SOURCE_DIR}/spg.hpp
)

set(REOR_BINDINGS_SRC
  ${REOR_BINDINGS_SOURCE_DIR}/bindings.cpp
)

set(REOR_TEST_SRC
   ${REOR_TEST_DIR}/test_main.cpp
   ${REOR_TEST_DIR}/test_eigen_backend.cpp
   ${REOR_TEST_DIR}/test_furthest_sum.cpp
   ${REOR_TEST_DIR}/test_gpnh_l2_spa.cpp
   ${REOR_TEST_DIR}/test_kernel_aa.cpp
)

if(TARGET Eigen3::Eigen)
  set(REOR_TEST_SRC
    ${REOR_TEST_SRC}
    ${REOR_TEST_DIR}/test_eigen_backend.cpp
)
endif()

add_library(reor_cxx INTERFACE)
target_include_directories(reor_cxx INTERFACE
  $<BUILD_INTERFACE:${REOR_CXX_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${REOR_INSTALL_INCLUDE_DIR}>)

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(reor_cxx INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_WARNING_FLAGS}>)
endif()

if(TARGET Eigen3::Eigen)
  target_link_libraries(reor_cxx
    INTERFACE
    Eigen3::Eigen)
endif()

add_subdirectory(lib/pybind11)
pybind11_add_module(reor_ext ${REOR_BINDINGS_SRC})

target_include_directories(reor_ext PUBLIC
  $<BUILD_INTERFACE:${REOR_CXX_SOURCE_DIR} ${REOR_BINDINGS_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${REOR_INSTALL_INCLUDE_DIR}>)
target_link_libraries(reor_ext PUBLIC reor_cxx)

if(TARGET Eigen3::Eigen)
  target_link_libraries(reor_ext
    INTERFACE
    Eigen3::Eigen)

  target_compile_definitions(reor_ext PUBLIC HAVE_EIGEN)
endif()

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(reor_ext PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_WARNING_FLAGS}>)
endif()

add_executable(reor_cxx_tests ${REOR_TEST_SRC})

set_property(TARGET reor_cxx_tests PROPERTY CXX_STANDARD 14)

target_include_directories(reor_cxx_tests PUBLIC
   $<BUILD_INTERFACE:${REOR_CXX_SOURCE_DIR} ${REOR_BINDINGS_SOURCE_DIR} ${REOR_TEST_DIR} ${REOR_CATCH_INCLUDE_DIR}>
   $<INSTALL_INTERFACE:${REOR_INSTALL_INCLUDE_DIR}>
)

target_link_libraries(reor_cxx_tests PUBLIC reor_cxx)

if(TARGET Eigen3::Eigen)
  target_link_libraries(reor_cxx_tests
    INTERFACE
    Eigen3::Eigen)

  target_compile_definitions(reor_cxx_tests PUBLIC HAVE_EIGEN)
endif()

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(reor_cxx_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_WARNING_FLAGS}>)
endif()

add_library(Reor::reor_cxx ALIAS reor_cxx)
