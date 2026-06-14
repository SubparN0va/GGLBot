FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# ============================================================
# Install build tools: g++, cmake, make, git, curl, unzip,
#                      plus Clang 19 + Ninja for Windows cross-compile
# ============================================================
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        build-essential \
        g++ \
        cmake \
        make \
        ninja-build \
        git \
        curl \
        ca-certificates \
        unzip \
        pkg-config \
        wget \
        gnupg \
        lsb-release \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install LLVM/Clang 19 from the official LLVM repository
# (Ubuntu 24.04's default Clang 18 is too old for xwin's CRT headers)
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    add-apt-repository -y "deb https://apt.llvm.org/$(lsb_release -sc)/ llvm-toolchain-$(lsb_release -sc)-19 main" && \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        clang-19 \
        lld-19 \
        llvm-19-dev \
        libc++-19-dev \
        libc++abi-19-dev \
        clang-tidy-19 \
    && rm -rf /var/lib/apt/lists/*

# Set up symlinks so that /usr/bin/clang, /usr/bin/clang++, /usr/bin/lld etc. point to version 19
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-19 100 && \
    update-alternatives --install /usr/bin/ld.lld ld.lld /usr/bin/ld.lld-19 100 && \
    update-alternatives --install /usr/bin/llvm-ar llvm-ar /usr/bin/llvm-ar-19 100 && \
    update-alternatives --install /usr/bin/llvm-ranlib llvm-ranlib /usr/bin/llvm-ranlib-19 100 && \
    update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-19 100 && \
    update-alternatives --install /usr/bin/lld-link lld-link /usr/bin/lld-link-19 100 && \
    update-alternatives --install /usr/bin/llvm-dlltool llvm-dlltool /usr/bin/llvm-dlltool-19 100 && \
    update-alternatives --install /usr/bin/llvm-lib llvm-lib /usr/bin/llvm-lib-19 100

RUN git --version && g++ --version | head -1 && cmake --version | head -1 && clang++-19 --version | head -1

# ============================================================
# Install Rust + xwin – downloads the Windows CRT + SDK headers
# so Clang can target x86_64-pc-windows-msvc on Linux.
# ============================================================
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path
ENV PATH="/root/.cargo/bin:$PATH"

RUN cargo install xwin --locked

# Accepting the Microsoft license is required to download the SDK
# Also list the structure so we can fix paths
RUN xwin --accept-license splat --output /tmp/xwin && \
    echo "=== xwin output structure ===" && \
    find /tmp/xwin -type d | head -40 && \
    echo "=== xwin lib files ===" && \
    find /tmp/xwin -name "*.lib" | head -20

# ============================================================
# Download and extract LibTorch – Linux version (CPU)
# ============================================================
RUN echo "Downloading LibTorch CPU (Linux) ..." && \
    curl -fsSL -o /tmp/libtorch-linux.zip \
        "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip" && \
    echo "Extracting ..." && \
    unzip -q /tmp/libtorch-linux.zip -d /tmp/ && \
    mv /tmp/libtorch /tmp/libtorch-linux && \
    rm -f /tmp/libtorch-linux.zip && \
    echo "LibTorch (Linux) ready"

RUN test -d /tmp/libtorch-linux && \
    test -f /tmp/libtorch-linux/share/cmake/Torch/TorchConfig.cmake

# ============================================================
# Download and extract LibTorch – Windows version (CPU, MSVC)
# ============================================================
RUN echo "Downloading LibTorch CPU (Windows) ..." && \
    curl -fsSL -o /tmp/libtorch-win.zip \
        "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.7.0%2Bcpu.zip" && \
    echo "Extracting ..." && \
    unzip -q /tmp/libtorch-win.zip -d /tmp/ && \
    mv /tmp/libtorch /tmp/libtorch-win && \
    rm -f /tmp/libtorch-win.zip && \
    echo "LibTorch (Windows) ready"

RUN test -d /tmp/libtorch-win && \
    test -f /tmp/libtorch-win/share/cmake/Torch/TorchConfig.cmake

# ============================================================
# Copy project into container
# ============================================================
WORKDIR /src
COPY . /src

# ============================================================
# Create a clang-cl wrapper so Clang accepts MSVC-style flags (/EHsc etc.)
RUN printf '#!/bin/bash\nexec /usr/bin/clang++ --driver-mode=cl -target x86_64-pc-windows-msvc "$@"\n' > /usr/local/bin/clang-cl && \
    printf '#!/bin/bash\nexec /usr/bin/clang --driver-mode=cl -target x86_64-pc-windows-msvc "$@"\n' > /usr/local/bin/clang-cl-c && \
    chmod +x /usr/local/bin/clang-cl /usr/local/bin/clang-cl-c

# Create CMake toolchain file for Windows cross-compilation
# Uses Clang in MSVC-compatible driver mode + xwin CRT/SDK
# ============================================================
RUN mkdir -p /src/cmake && \
    cat > /src/cmake/toolchain-msvc.cmake << 'TOOLCHAIN_EOF'
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
# Tell CMake to use llvm-lib in MSVC mode for static libraries
set(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> /OUT:<TARGET> <OBJECTS>")

# In clang-cl mode, MSVC-style flags are understood natively.
# Add xwin CRT/SDK headers via -imsvc (clang-cl equivalent of -isystem).
# Use libc++ for C++ standard library (xwin doesn't ship C++ headers).
# /Zc:preprocessor is MSVC-only and clang-cl rejects it; suppress the warning.
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
  /tmp/xwin/crt /tmp/xwin/sdk /tmp/libtorch-win
)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Use dynamic MSVC runtime (/MD, already in flags above)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
TOOLCHAIN_EOF

# ============================================================
# Build Linux GGLBot  (native g++)
# ============================================================
RUN cmake -S . -B build-linux -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBTORCH_ROOT=/tmp/libtorch-linux \
    && cmake --build build-linux --parallel "$(nproc)"

# ============================================================
# Build Windows GGLBot  (cross-compiled with Clang + MSVC ABI)
# ============================================================
# Patch cpp-interface to not pass /Zc:preprocessor (clang-cl rejects it)
RUN sed -i '/Zc:preprocessor/d' /src/cpp-interface/library/CMakeLists.txt

RUN cmake -S . -B build-win -G "Ninja" \
        -DCMAKE_TOOLCHAIN_FILE=/src/cmake/toolchain-msvc.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBTORCH_ROOT=/tmp/libtorch-win \
    && cmake --build build-win --parallel "$(nproc)"

# ============================================================
# Package everything for bob
# Structure:
#   /out/
#     x86_64-unknown-linux-gnu/   ← Linux binary + SOs + config
#     x86_64-pc-windows-msvc/     ← Windows binary + DLLs + config
# ============================================================

# --- Linux package -------------------------------------------------------
RUN mkdir -p /out/x86_64-unknown-linux-gnu

# Copy the built Linux GGLBot
RUN if [ ! -f /src/build-linux/GGLBot ]; then \
        echo "ERROR: Expected /src/build-linux/GGLBot but it is missing."; \
        exit 1; \
    fi && \
    cp /src/build-linux/GGLBot /out/x86_64-unknown-linux-gnu/GGLBot

# Copy libtorch runtime SOs for Linux (minimal inference set)
RUN if [ -d /tmp/libtorch-linux/lib ]; then \
        cp /tmp/libtorch-linux/lib/libtorch_cpu.so     /out/x86_64-unknown-linux-gnu/ 2>/dev/null || true; \
        cp /tmp/libtorch-linux/lib/libc10.so            /out/x86_64-unknown-linux-gnu/ 2>/dev/null || true; \
        cp /tmp/libtorch-linux/lib/libtorch.so          /out/x86_64-unknown-linux-gnu/ 2>/dev/null || true; \
        cp /tmp/libtorch-linux/lib/libtorch_global_deps.so /out/x86_64-unknown-linux-gnu/ 2>/dev/null || true; \
        cp /tmp/libtorch-linux/lib/libgomp*.so*         /out/x86_64-unknown-linux-gnu/ 2>/dev/null || true; \
    fi

# --- Windows package -----------------------------------------------------
RUN mkdir -p /out/x86_64-pc-windows-msvc

# Copy the built Windows GGLBot.exe
RUN if [ ! -f /src/build-win/GGLBot.exe ]; then \
        echo "ERROR: Expected /src/build-win/GGLBot.exe but it is missing."; \
        find /src/build-win -name "GGLBot*" -type f 2>/dev/null || true; \
        exit 1; \
    fi && \
    cp /src/build-win/GGLBot.exe /out/x86_64-pc-windows-msvc/GGLBot.exe

# Copy the LibTorch runtime DLL closure used by the Windows bot.
RUN cat >/tmp/copy-pe-dll-closure.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

out_dir="$1"
search_dir="$2"
shift 2

if [[ ! -d "$search_dir" ]]; then
    echo "ERROR: Missing $search_dir" >&2
    exit 1
fi

objdump="$(command -v llvm-objdump-19 || command -v llvm-objdump || command -v objdump || true)"
if [ -z "$objdump" ]; then
    echo "ERROR: Need llvm-objdump or objdump to inspect Windows DLL imports" >&2
    exit 1
fi

queue_file=/tmp/pe-dll-queue.txt
scanned_file=/tmp/pe-dll-scanned.txt
: > "$queue_file"
: > "$scanned_file"

for seed in "$@"; do
    if [ -f "$seed" ]; then
        printf '%s\n' "$seed" >> "$queue_file"
    else
        src="$(find "$search_dir" -maxdepth 1 -type f -iname "$seed" -print -quit)"
        if [ -n "$src" ]; then
            cp -n "$src" "$out_dir/"
            printf '%s\n' "$out_dir/$(basename "$src")" >> "$queue_file"
        fi
    fi
done

while [ -s "$queue_file" ]; do
    binary="$(sed -n '1p' "$queue_file")"
    tail -n +2 "$queue_file" > "$queue_file.next" || true
    mv "$queue_file.next" "$queue_file"
    [ -f "$binary" ] || continue

    key="$(basename "$binary" | tr '[:upper:]' '[:lower:]')"
    if grep -Fxq "$key" "$scanned_file"; then
        continue
    fi
    printf '%s\n' "$key" >> "$scanned_file"

    while IFS= read -r dep; do
        dep="$(printf '%s' "$dep" | tr -d '\r')"
        dep_key="$(printf '%s' "$dep" | tr '[:upper:]' '[:lower:]')"
        if grep -Fxq "$dep_key" "$scanned_file"; then
            continue
        fi

        src="$(find "$search_dir" -maxdepth 1 -type f -iname "$dep" -print -quit)"
        if [ -n "$src" ]; then
            cp -n "$src" "$out_dir/"
            printf '%s\n' "$out_dir/$(basename "$src")" >> "$queue_file"
        fi
    done < <("$objdump" -p "$binary" | sed -n 's/^[[:space:]]*DLL Name: //p')
done

echo "Copied Windows runtime DLLs:"
find "$out_dir" -maxdepth 1 -type f -iname '*.dll' -printf '%f\n' | sort -f
EOF
RUN chmod +x /tmp/copy-pe-dll-closure.sh && \
    /tmp/copy-pe-dll-closure.sh \
        /out/x86_64-pc-windows-msvc \
        /tmp/libtorch-win/lib \
        /out/x86_64-pc-windows-msvc/GGLBot.exe \
        torch.dll \
        torch_cpu.dll \
        torch_global_deps.dll \
        c10.dll

# --- Shared assets (logo, models, config) --------------------------------
# Copy logo.png to parent folder (if it exists)
RUN if [ -f /src/rlbot/logo.png ]; then \
        cp /src/rlbot/logo.png /out/logo.png; \
        echo "Copied logo.png"; \
    else \
        echo "No logo.png found"; \
    fi

# Copy any *.lt model files from rlbot (recursively) beside both binaries
RUN if [ -d /src/rlbot ]; then \
        shopt -s globstar nullglob; \
        for f in /src/rlbot/**/*.lt; do \
            cp "$f" /out/x86_64-unknown-linux-gnu/; \
            cp "$f" /out/x86_64-pc-windows-msvc/; \
            echo "Copied $f"; \
        done; \
    else \
        echo "No rlbot folder found at /src/rlbot"; \
    fi

# Copy config files (bot.toml, loadout.toml, run.sh / run.bat) beside both
RUN for f in bot.toml loadout.toml; do \
        if [ -f "/src/rlbot/$f" ]; then \
            cp "/src/rlbot/$f" "/out/x86_64-unknown-linux-gnu/$f"; \
            cp "/src/rlbot/$f" "/out/x86_64-pc-windows-msvc/$f"; \
            echo "Copied $f"; \
        fi; \
    done

# Copy platform-specific run scripts
RUN if [ -f "/src/rlbot/run.sh" ]; then \
        cp "/src/rlbot/run.sh" "/out/x86_64-unknown-linux-gnu/run.sh"; \
    fi
RUN if [ -f "/src/rlbot/run.bat" ]; then \
        cp "/src/rlbot/run.bat" "/out/x86_64-pc-windows-msvc/run.bat"; \
    fi

# List output contents for fun
RUN echo "=== Linux package ===" && ls -la /out/x86_64-unknown-linux-gnu/ && \
    echo "" && \
    echo "=== Windows package ===" && ls -la /out/x86_64-pc-windows-msvc/ && \
    echo "" && \
    echo "=== /out/ root ===" && ls -la /out/ 2>/dev/null

# Emit tar to stdout for bob (includes both platform directories)
ENTRYPOINT ["tar", "-C", "/out", "-cf", "-", \
    "x86_64-unknown-linux-gnu", \
    "x86_64-pc-windows-msvc"]
