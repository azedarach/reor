add_executable(factor_csv_l2spa
  csv_utils.cpp
  factor_csv_l2spa.cpp
  l2spa_fit_wrappers.cpp)

target_link_libraries(factor_csv_l2spa
  Reor::reor)

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(factor_csv_l2spa
    PRIVATE "${CXX_WARNING_FLAGS}")
endif()

set_target_properties(factor_csv_l2spa PROPERTIES
  CXX_STANDARD 11
  CXX_EXTENSIONS OFF)

add_executable(Reor::factor_csv_l2spa ALIAS factor_csv_l2spa)

add_dependencies(examples factor_csv_l2spa)

