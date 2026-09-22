find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(runtime_test_sources ${GGLBOT_SOURCES})
list(FILTER runtime_test_sources EXCLUDE REGEX "/RLBotMain\\.cpp$")

function(add_runtime_tests device runtime_root use_cuda)
  set(target GGLBotRuntimeTests${device})
  add_executable(${target} RuntimeTests.cpp ${runtime_test_sources})
  configure_gglbot_target(${target} "${runtime_root}" ${use_cuda})
  add_test(NAME runtime-${device} COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/run_test.py"
    "$<TARGET_FILE:${target}>" "${runtime_root}/lib" "${CMAKE_CURRENT_BINARY_DIR}/models-${device}"
  )
endfunction()

if(GGLBOT_USE_CUDA)
  add_runtime_tests(CUDA "${GGLBOT_RUNTIME_ROOT}" TRUE)
else()
  add_runtime_tests(CPU "${GGLBOT_RUNTIME_ROOT}" FALSE)
endif()
