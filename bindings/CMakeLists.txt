add_custom_target(bindings)

if(ENABLE_PYTHON_BINDINGS)
  add_subdirectory(pyreor)
endif()
