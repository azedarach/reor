add_executable(fit_csv_l2trspa
  csv_utils.cpp
  fit_csv_l2trspa.cpp
  l2trspa_fit_wrappers.cpp)

target_link_libraries(fit_csv_l2trspa
  Reor::reor)

if(ENABLE_COMPILER_WARNINGS)
  target_compile_options(fit_csv_l2trspa
    PRIVATE "${CXX_WARNING_FLAGS}")
endif()

set_target_properties(fit_csv_l2trspa PROPERTIES
  CXX_STANDARD 11
  CXX_EXTENSIONS OFF)

add_executable(Reor::fit_csv_l2trspa ALIAS fit_csv_l2trspa)

add_dependencies(examples fit_csv_l2trspa)
