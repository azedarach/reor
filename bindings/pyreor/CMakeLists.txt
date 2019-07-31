add_subdirectory(pybind11)

pybind11_add_module(pyreor_ext
  src/pyreor.cpp)

target_link_libraries(pyreor_ext
  PRIVATE Reor::reor)

if(TARGET Eigen3::Eigen)
  target_compile_definitions(pyreor_ext
    PRIVATE HAVE_EIGEN)
endif()

file(COPY
  _random_matrix.py
  _spa.py
  _validation.py
  DESTINATION
  ${CMAKE_CURRENT_BINARY_DIR}
  )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
  COPYONLY)

add_dependencies(bindings pyreor_ext)
