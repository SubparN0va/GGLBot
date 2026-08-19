# Cross-compilation toolchain: Clang → x86_64-pc-windows-msvc
# with CRT + Windows SDK provided by xwin.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_VERSION 10.0)

set(CMAKE_C_COMPILER /usr/local/bin/clang-cl-c)
set(CMAKE_CXX_COMPILER /usr/local/bin/clang-cl)
set(CMAKE_C_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)

set(CMAKE_LINKER lld-link)
set(CMAKE_AR llvm-lib)
set(CMAKE_RANLIB llvm-lib)

# llvm-lib is the MSVC-compatible static librarian (understands /out: etc.)
set(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")

# In clang-cl mode, MSVC-style flags are understood natively.
# Add xwin CRT/SDK headers via -imsvc (clang-cl equivalent of -isystem).
# Use libc++ for C++ standard library (xwin doesn't ship C++ headers).
set(CMAKE_C_FLAGS_INIT
  "-MD /imsvc/tmp/xwin/crt/include /imsvc/tmp/xwin/sdk/include/ucrt /imsvc/tmp/xwin/sdk/include/um /imsvc/tmp/xwin/sdk/include/shared /clang:-Wno-unused-command-line-argument"
)
set(CMAKE_CXX_FLAGS_INIT
  "-MD -Xclang -stdlib=libc++ /imsvc/tmp/xwin/crt/include /imsvc/tmp/xwin/sdk/include/ucrt /imsvc/tmp/xwin/sdk/include/um /imsvc/tmp/xwin/sdk/include/shared /clang:-Wno-unused-command-line-argument"
)

# Linker flags: point at xwin CRT/SDK libraries (MSVC /LIBPATH: style for clang-cl)
set(CMAKE_EXE_LINKER_FLAGS_INIT
  "/LIBPATH:/tmp/xwin/crt/lib/x86_64 /LIBPATH:/tmp/xwin/sdk/lib/um/x86_64 /LIBPATH:/tmp/xwin/sdk/lib/ucrt/x86_64"
)

# Use Release config for try_compile tests (avoid msvcrtd.lib debug CRT)
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)

# Ensure .lib import libraries are found
set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .dll .dll.a .a)

# Tell CMake search to prefer xwin + libtorch
set(CMAKE_FIND_ROOT_PATH
  /tmp/xwin/crt /tmp/xwin/sdk /tmp/botpack-win/torch-archive/torch
)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Use dynamic MSVC runtime (/MD, already in flags above)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
