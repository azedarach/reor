add_executable(run_lorenz63_l2spa
  l2spa_fit_wrappers.cpp
  lorenz63_rk4_model.cpp
  run_lorenz63_l2spa.cpp)

target_link_libraries(run_lorenz63_l2spa
  Reor::reor)

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(run_lorenz63_l2spa
    PRIVATE "${CXX_WARNING_FLAGS}")
endif()

set_target_properties(run_lorenz63_l2spa PROPERTIES
  CXX_STANDARD 11
  CXX_EXTENSIONS OFF)

add_executable(Reor::run_lorenz63_l2spa ALIAS run_lorenz63_l2spa)

add_dependencies(examples run_lorenz63_l2spa)
