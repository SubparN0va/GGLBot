cmake_minimum_required(VERSION 3.22)

project(GGLBot)

set(DEVICE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/rlbot/device.txt")
include(cmake/ReadDevice.cmake)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${DEVICE_FILE}")
if(GGLBOT_DEVICE STREQUAL "cuda")
  if(NOT WIN32)
    message(FATAL_ERROR "CUDA builds currently support Windows only. Select cpu for Linux.")
  endif()
  set(GGLBOT_USE_CUDA TRUE)
  set(GGLBOT_CORE GGLBotCoreCUDA)
  set(runtime_variable LIBTORCH_CUDA_ROOT)
  set(runtime_subdir libtorch)
  set(runtime_marker torch_cuda)
else()
  set(GGLBOT_USE_CUDA FALSE)
  set(GGLBOT_CORE GGLBotCoreCPU)
  set(runtime_variable LIBTORCH_CPU_ROOT)
  set(runtime_subdir torch-archive/torch)
  set(runtime_marker torch_cpu)
endif()

if(WIN32)
  # The shared runtimes use the release MSVC ABI, including for builds with debug symbols.
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR "Use Release or RelWithDebInfo with the shared LibTorch runtimes (see CMakePresets.json).")
  endif()
  if(CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_CONFIGURATION_TYPES "Release;RelWithDebInfo" CACHE STRING "Supported LibTorch configurations" FORCE)
  endif()
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
endif()

# Resolve only the selected SDK. Explicit CMake and environment overrides are exclusive.
if(NOT DEFINED ${runtime_variable})
  if(DEFINED ENV{${runtime_variable}} AND NOT "$ENV{${runtime_variable}}" STREQUAL "")
    set(default_runtime "$ENV{${runtime_variable}}")
  else()
    if(WIN32)
      set(data_home "$ENV{LOCALAPPDATA}")
    elseif(DEFINED ENV{XDG_DATA_HOME} AND NOT "$ENV{XDG_DATA_HOME}" STREQUAL "")
      set(data_home "$ENV{XDG_DATA_HOME}")
    else()
      set(data_home "$ENV{HOME}/.local/share")
    endif()
    set(default_runtime "${data_home}/RLBot5/bots/${runtime_subdir}")
    set(directory "${CMAKE_CURRENT_SOURCE_DIR}/rlbot")
    foreach(level RANGE 1 5)
      set(candidate "${directory}/${runtime_subdir}")
      if(EXISTS "${candidate}/lib/${runtime_marker}.lib" OR EXISTS "${candidate}/lib/lib${runtime_marker}.so")
        set(default_runtime "${candidate}")
        break()
      endif()
      get_filename_component(directory "${directory}" DIRECTORY)
    endforeach()
  endif()
  set(${runtime_variable} "${default_runtime}" CACHE PATH "Selected LibTorch SDK root")
endif()
set(GGLBOT_RUNTIME_ROOT "${${runtime_variable}}")
message(STATUS "GGLBot: ${GGLBOT_DEVICE}, ${runtime_variable}=${GGLBOT_RUNTIME_ROOT}")

add_subdirectory(cpp-interface EXCLUDE_FROM_ALL)

file(GLOB_RECURSE GGLBOT_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
file(GLOB_RECURSE GGLBOT_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
)

# The launcher locates the selected runtime before loading Torch.
add_executable(GGLBot launcher/GGLBotLauncher.cpp)
target_compile_features(GGLBot PRIVATE cxx_std_20)
target_compile_definitions(GGLBot PRIVATE GGLBOT_USE_CUDA=$<BOOL:${GGLBOT_USE_CUDA}>)
if(WIN32)
  target_link_libraries(GGLBot PRIVATE shell32)
endif()

set(GGLBOT_DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rlbot")
set(GGLBOT_RUNTIME_DIR "${GGLBOT_DEPLOY_DIR}/000-runtime")

function(configure_gglbot_target target runtime_root use_cuda)
  target_compile_features(${target} PRIVATE cxx_std_20)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/inc" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/src")
  target_compile_definitions(${target} PRIVATE GGLBOT_USE_CUDA=$<BOOL:${use_cuda}>)
  target_link_libraries(${target} PRIVATE RLBotCPP-static)

  if(WIN32)
    # Direct linking avoids requiring a Windows CUDA toolkit inside the Linux cross-compiler.
    set(torch_libraries torch torch_cpu c10)
    if(use_cuda)
      list(APPEND torch_libraries torch_cuda c10_cuda)
    endif()
    foreach(library IN LISTS torch_libraries)
      set(import_library "${runtime_root}/lib/${library}.lib")
      if(NOT EXISTS "${import_library}")
        message(FATAL_ERROR "${target}: missing ${import_library}. Install the selected SDK or set ${runtime_variable}.")
      endif()
      target_link_libraries(${target} PRIVATE "${import_library}")
    endforeach()
    target_include_directories(${target} SYSTEM PRIVATE "${runtime_root}/include" "${runtime_root}/include/torch/csrc/api/include")
    target_compile_definitions(${target} PRIVATE USE_DISTRIBUTED USE_C10D_GLOO)
    target_compile_options(${target} PRIVATE /permissive- /EHsc /bigobj)
    if(use_cuda)
      # Keep CUDA registration code even though the public API lives in torch_cpu.
      target_link_options(${target} PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")
    endif()
  else()
    set(Torch_DIR "${runtime_root}/share/cmake/Torch")
    find_package(Torch REQUIRED CONFIG)
    if(TARGET Torch::Torch)
      target_link_libraries(${target} PRIVATE Torch::Torch)
    elseif(TARGET torch)
      target_link_libraries(${target} PRIVATE torch)
    else()
      target_include_directories(${target} SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
      target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES})
    endif()
    # RLBotCPP passes std::string across the core boundary and must use Torch's ABI too.
    separate_arguments(torch_compile_options NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(${target} PRIVATE ${torch_compile_options})
    target_compile_options(RLBotCPP-static PUBLIC ${torch_compile_options})
  endif()
endfunction()

function(add_gglbot_core target runtime_root use_cuda)
  add_executable(${target} ${GGLBOT_SOURCES} ${GGLBOT_HEADERS})
  configure_gglbot_target(${target} "${runtime_root}" ${use_cuda})
  add_dependencies(GGLBot ${target})
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GGLBOT_RUNTIME_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${target}>" "${GGLBOT_RUNTIME_DIR}/$<TARGET_FILE_NAME:${target}>"
    VERBATIM
  )
endfunction()

add_gglbot_core(${GGLBOT_CORE} "${GGLBOT_RUNTIME_ROOT}" ${GGLBOT_USE_CUDA})

add_custom_command(TARGET GGLBot POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GGLBOT_DEPLOY_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:GGLBot>" "${GGLBOT_DEPLOY_DIR}/$<TARGET_FILE_NAME:GGLBot>"
  VERBATIM
)

option(GGLBOT_BUILD_RUNTIME_TESTS "Build inference and live-packet regression tests" OFF)
if(GGLBOT_BUILD_RUNTIME_TESTS)
  enable_testing()
  add_subdirectory(tests/runtime)
endif()
