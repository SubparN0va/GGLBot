cmake_minimum_required(VERSION 3.22)

project(GGLBot)

add_subdirectory(cpp-interface EXCLUDE_FROM_ALL)

# Ensure RLBotCPP-static uses the same std::string ABI as PyTorch/LibTorch.
# TorchConfig.cmake propagates _GLIBCXX_USE_CXX11_ABI=0 to consumers of
# Torch::Torch, but RLBotCPP-static doesn't link Torch directly so it needs
# the flag set explicitly to avoid undefined symbol errors at link time.
if(NOT WIN32)
  target_compile_definitions(RLBotCPP-static PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
endif()

# ---- GGLBot core (Torch-linked executable) ----
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

add_executable(GGLBotCore
  ${GGLBOT_SOURCES}
  ${GGLBOT_HEADERS}
)

target_compile_features(GGLBotCore PRIVATE cxx_std_20)

target_include_directories(GGLBotCore PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(GGLBotCore PRIVATE RLBotCPP-static)

# ---- GGLBot launcher (no Torch dependency) ----
add_executable(GGLBot launcher/GGLBotLauncher.cpp)
target_compile_features(GGLBot PRIVATE cxx_std_20)
add_dependencies(GGLBot GGLBotCore)

if(WIN32)
  target_link_libraries(GGLBot PRIVATE shell32)
endif()

# Deploy the launcher beside bot.toml and keep the core in its runtime folder.
set(GGLBOT_DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rlbot")
set(GGLBOT_RUNTIME_DIR "${GGLBOT_DEPLOY_DIR}/000-runtime")
add_custom_command(TARGET GGLBotCore POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GGLBOT_RUNTIME_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:GGLBotCore>"
          "${GGLBOT_RUNTIME_DIR}/$<TARGET_FILE_NAME:GGLBotCore>"
  VERBATIM
)
add_custom_command(TARGET GGLBot POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GGLBOT_DEPLOY_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:GGLBot>"
          "${GGLBOT_DEPLOY_DIR}/$<TARGET_FILE_NAME:GGLBot>"
  VERBATIM
)

# Allow override: -DLIBTORCH_ROOT="C:/path/to/torch-archive/torch"
if(NOT DEFINED LIBTORCH_ROOT)
  if(WIN32)
    set(LIBTORCH_ROOT "$ENV{LOCALAPPDATA}/RLBot5/bots/torch-archive/torch")
  else()
    message(FATAL_ERROR "LIBTORCH_ROOT not set (and non-Windows build). Set -DLIBTORCH_ROOT=...")
  endif()
endif()

# TorchConfig.cmake lives here in the botpack/torch-archive layout:
set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch")

if(NOT EXISTS "${Torch_DIR}/TorchConfig.cmake")
  message(FATAL_ERROR
    "Could not find TorchConfig.cmake at: ${Torch_DIR}/TorchConfig.cmake\n"
    "Check LIBTORCH_ROOT='${LIBTORCH_ROOT}' (expected botpack layout with share/cmake/Torch)."
)
endif()

find_package(Torch REQUIRED CONFIG)

if(TARGET Torch::Torch)
  set(_TORCH_LINK Torch::Torch)
elseif(TARGET torch)
  set(_TORCH_LINK torch)
else()
  set(_TORCH_LINK ${TORCH_LIBRARIES})
endif()

target_link_libraries(GGLBotCore PRIVATE ${_TORCH_LINK})
