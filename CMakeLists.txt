cmake_minimum_required(VERSION 3.22)

project(GGLBot)

set(CMAKE_CXX_STANDARD 20) 

add_subdirectory(cpp-interface EXCLUDE_FROM_ALL)

# ---- GGLBot (EXE) ----
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

add_executable(GGLBot
  ${GGLBOT_SOURCES} 
  ${GGLBOT_HEADERS}
) 

target_include_directories(GGLBot PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"      
  "${CMAKE_CURRENT_SOURCE_DIR}/inc"  
  "${CMAKE_CURRENT_SOURCE_DIR}/src" 
)

target_link_libraries(GGLBot PRIVATE RLBotCPP-static)

# Copy the exe to the rlbot/ folder so it doesn't have to be done manually
set(GGLBOT_DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rlbot")
add_custom_command(TARGET GGLBot POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${GGLBOT_DEPLOY_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:GGLBot>"
          "${GGLBOT_DEPLOY_DIR}/$<TARGET_FILE_NAME:GGLBot>"
  VERBATIM
)

# Allow override: -DLIBTORCH_ROOT="C:/path/to/libtorch_cpu"
if(NOT DEFINED LIBTORCH_ROOT)
    if(WIN32)
        set(LIBTORCH_ROOT "$ENV{LOCALAPPDATA}/RLBot5/bots/libtorch_cpu")
    else()
        message(FATAL_ERROR "LIBTORCH_ROOT not set (and non-Windows build). Set -DLIBTORCH_ROOT=...") 
    endif()
endif() 

# TorchConfig.cmake usually lives here in the official libtorch distribution:
set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch")

if(NOT EXISTS "${Torch_DIR}/TorchConfig.cmake")
message(FATAL_ERROR
    "Could not find TorchConfig.cmake at: ${Torch_DIR}/TorchConfig.cmake\n"
    "Check LIBTORCH_ROOT='${LIBTORCH_ROOT}' (expected libtorch layout with share/cmake/Torch)."
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

target_link_libraries(GGLBot PRIVATE ${_TORCH_LINK})