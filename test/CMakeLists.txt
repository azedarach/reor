set(REOR_TEST_SOURCES
  catch_main.cpp
  test_gpnh_regularizer.cpp
  test_l2_spa.cpp
  test_trivial_matrix_factorization_regularizer.cpp)

if(TARGET Eigen3::Eigen)
  set(REOR_TEST_SOURCES
    ${REOR_TEST_SOURCES}
    test_eigen_backend.cpp)
endif()

add_executable(test_reor
  ${REOR_TEST_SOURCES})

target_include_directories(test_reor
  PRIVATE catch)

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(test_reor
    PRIVATE
    ${CXX_WARNING_FLAGS})
endif()

if(TARGET Eigen3::Eigen)
  target_compile_definitions(test_reor
    PRIVATE HAVE_EIGEN)
endif()


set_target_properties(test_reor PROPERTIES
  CXX_STANDARD 11
  CXX_EXTENSIONS OFF)

target_link_libraries(test_reor
  Reor::reor)

include(Catch)
catch_discover_tests(test_reor)

add_executable(Reor::test_reor ALIAS test_reor)

add_custom_target(check COMMAND "ctest")
add_dependencies(check test_reor)
