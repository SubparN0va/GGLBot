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
# Create a clang-cl wrapper so Clang accepts MSVC-style flags (/EHsc etc.)
# This is static — no source dependency — so we do it early for caching.
# ============================================================
RUN mkdir -p /usr/local/bin && \
    printf '#!/bin/bash\nexec /usr/bin/clang++ --driver-mode=cl -target x86_64-pc-windows-msvc "$@"\n' > /usr/local/bin/clang-cl && \
    printf '#!/bin/bash\nexec /usr/bin/clang --driver-mode=cl -target x86_64-pc-windows-msvc "$@"\n' > /usr/local/bin/clang-cl-c && \
    chmod +x /usr/local/bin/clang-cl /usr/local/bin/clang-cl-c

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
# Download and extract PyTorch botpack – Linux version (CPU)
# ============================================================
RUN echo "Downloading PyTorch botpack (Linux) ..." && \
    curl -fsSL -o /tmp/botpack-linux.tar.xz \
        "https://github.com/VirxEC/pytorch-archive/releases/download/r-1/botpack_x86_64-linux.tar.xz" && \
    echo "Extracting ..." && \
    mkdir -p /tmp/botpack-linux && \
    tar -xJf /tmp/botpack-linux.tar.xz -C /tmp/botpack-linux && \
    rm -f /tmp/botpack-linux.tar.xz && \
    echo "PyTorch botpack (Linux) ready"

RUN test -d /tmp/botpack-linux/torch-archive/torch && \
    test -f /tmp/botpack-linux/torch-archive/torch/share/cmake/Torch/TorchConfig.cmake

# ============================================================
# Download and extract PyTorch botpack – Windows version (CPU, MSVC)
# ============================================================
RUN echo "Downloading PyTorch botpack (Windows) ..." && \
    curl -fsSL -o /tmp/botpack-win.tar.xz \
        "https://github.com/VirxEC/pytorch-archive/releases/download/r-1/botpack_x86_64-windows.tar.xz" && \
    echo "Extracting ..." && \
    mkdir -p /tmp/botpack-win && \
    tar -xJf /tmp/botpack-win.tar.xz -C /tmp/botpack-win && \
    rm -f /tmp/botpack-win.tar.xz && \
    echo "PyTorch botpack (Windows) ready"

RUN test -d /tmp/botpack-win/torch-archive/torch && \
    test -f /tmp/botpack-win/torch-archive/torch/share/cmake/Torch/TorchConfig.cmake

# ============================================================
# Copy project into container
# ============================================================
WORKDIR /src
COPY . /src

# ============================================================
# Build Linux GGLBot  (native g++)
# ============================================================
RUN cmake -S . -B build-linux -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBTORCH_ROOT=/tmp/botpack-linux/torch-archive/torch \
    && cmake --build build-linux --parallel "$(nproc)"

# ============================================================
# Build Windows GGLBot  (cross-compiled with Clang + MSVC ABI)
# ============================================================
# Patch cpp-interface to not pass /Zc:preprocessor (clang-cl rejects it)
RUN sed -i '/Zc:preprocessor/d' /src/cpp-interface/library/CMakeLists.txt

RUN cmake -S . -B build-win -G "Ninja" \
        -DCMAKE_TOOLCHAIN_FILE=/src/cmake/toolchain-msvc.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DLIBTORCH_ROOT=/tmp/botpack-win/torch-archive/torch \
    && cmake --build build-win --parallel "$(nproc)"

# ============================================================
# Package everything for bob
# ============================================================

# --- Linux package -------------------------------------------------------
RUN mkdir -p /out/x86_64-unknown-linux-gnu

# Copy the Linux launcher and Torch-linked core
RUN mkdir -p /out/x86_64-unknown-linux-gnu/000-runtime
RUN if [ ! -f /src/build-linux/GGLBot ] || [ ! -f /src/build-linux/GGLBotCore ]; then \
        echo "ERROR: Expected Linux launcher/core outputs but one is missing."; \
        exit 1; \
    fi && \
    cp /src/build-linux/GGLBot /out/x86_64-unknown-linux-gnu/GGLBot && \
    cp /src/build-linux/GGLBotCore /out/x86_64-unknown-linux-gnu/000-runtime/GGLBotCore

# --- Windows package -----------------------------------------------------
RUN mkdir -p /out/x86_64-pc-windows-msvc

# Copy the Windows launcher and Torch-linked core
RUN mkdir -p /out/x86_64-pc-windows-msvc/000-runtime
RUN if [ ! -f /src/build-win/GGLBot.exe ] || [ ! -f /src/build-win/GGLBotCore.exe ]; then \
        echo "ERROR: Expected Windows launcher/core outputs but one is missing."; \
        find /src/build-win -name "GGLBot*" -type f 2>/dev/null || true; \
        exit 1; \
    fi && \
    cp /src/build-win/GGLBot.exe /out/x86_64-pc-windows-msvc/GGLBot.exe && \
    cp /src/build-win/GGLBotCore.exe /out/x86_64-pc-windows-msvc/000-runtime/GGLBotCore.exe

# Copy any *.lt model files from rlbot into both platform directories
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

# Emit tar to stdout for bob. Sorting places 000-runtime before the launcher,
# so bob selects the launcher as the platform entry point.
ENTRYPOINT ["tar", "--sort=name", "-C", "/out", "-cf", "-", \
    "x86_64-unknown-linux-gnu", \
    "x86_64-pc-windows-msvc"]
