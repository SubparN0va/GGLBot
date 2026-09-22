cmake_minimum_required(VERSION 3.22)
project(GGLBotLauncherTests LANGUAGES CXX)

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME bob-build-helper COMMAND "${Python3_EXECUTABLE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/test_bob_build.py")

foreach(device CPU CUDA)
  if(device STREQUAL "CUDA" AND NOT WIN32)
    continue()
  endif()
  add_executable(GGLBotLauncherTest${device} ../launcher/GGLBotLauncher.cpp)
  target_compile_features(GGLBotLauncherTest${device} PRIVATE cxx_std_20)
  target_compile_definitions(GGLBotLauncherTest${device} PRIVATE GGLBOT_USE_CUDA=$<STREQUAL:${device},CUDA>)
  if(WIN32)
    target_link_libraries(GGLBotLauncherTest${device} PRIVATE shell32)
  endif()
  add_executable(TestCore${device} TestCore.cpp)
  target_compile_features(TestCore${device} PRIVATE cxx_std_20)
  target_compile_definitions(TestCore${device} PRIVATE TEST_DEVICE="${device}")
  add_test(NAME launcher-${device} COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_launcher.py"
    "$<TARGET_FILE:GGLBotLauncherTest${device}>" "$<TARGET_FILE:TestCore${device}>" "${device}")
endforeach()
