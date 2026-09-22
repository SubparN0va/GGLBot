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

# Stage only the selected Windows SDK; the helper supplies headers and import libraries.
COPY build-support/libtorch/ /tmp/libtorch-win/
COPY cmake/ReadDevice.cmake /tmp/ReadDevice.cmake
COPY rlbot/device.txt /tmp/gglbot-device.txt
RUN cmake -DDEVICE_FILE=/tmp/gglbot-device.txt -DOUTPUT_FILE=/tmp/gglbot-device \
        -P /tmp/ReadDevice.cmake && \
    test -f /tmp/libtorch-win/local/lib/torch.lib

# Linux CPU needs its platform's bot-pack archive. CUDA builds never fetch CPU archives.
ARG LIBTORCH_LINUX_CPU_URL=https://github.com/VirxEC/pytorch-archive/releases/download/r-1/botpack_x86_64-linux.tar.xz
RUN if [ "$(cat /tmp/gglbot-device)" = cpu ]; then \
        curl -fsSL -o /tmp/botpack-linux.tar.xz "$LIBTORCH_LINUX_CPU_URL" && \
        mkdir -p /tmp/botpack-linux && \
        tar -xJf /tmp/botpack-linux.tar.xz -C /tmp/botpack-linux && \
        rm -f /tmp/botpack-linux.tar.xz; \
    fi

WORKDIR /src
COPY . /src

RUN if [ "$(cat /tmp/gglbot-device)" = cpu ]; then \
        cmake -S . -B build-linux -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DGGLBOT_DEVICE=cpu \
            -DLIBTORCH_CPU_ROOT=/tmp/botpack-linux/torch-archive/torch && \
        cmake --build build-linux --target GGLBot --parallel "$(nproc)" && \
        mkdir -p /out/x86_64-unknown-linux-gnu/000-runtime && \
        cp build-linux/GGLBot /out/x86_64-unknown-linux-gnu/ && \
        cp build-linux/GGLBotCoreCPU /out/x86_64-unknown-linux-gnu/000-runtime/; \
    fi

RUN sed -i '/Zc:preprocessor/d' /src/cpp-interface/library/CMakeLists.txt
RUN device="$(cat /tmp/gglbot-device)" && \
    if [ "$device" = cuda ]; then root=LIBTORCH_CUDA_ROOT; core=GGLBotCoreCUDA; \
    else root=LIBTORCH_CPU_ROOT; core=GGLBotCoreCPU; fi && \
    cmake -S . -B build-win -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE=/src/cmake/toolchain-msvc.cmake \
        -DCMAKE_BUILD_TYPE=Release -DGGLBOT_DEVICE="$device" \
        -D"$root"=/tmp/libtorch-win/local && \
    cmake --build build-win --target GGLBot --parallel "$(nproc)" && \
    mkdir -p /out/x86_64-pc-windows-msvc/000-runtime && \
    cp build-win/GGLBot.exe /out/x86_64-pc-windows-msvc/ && \
    cp "build-win/$core.exe" /out/x86_64-pc-windows-msvc/000-runtime/

RUN shopt -s globstar nullglob; \
    for platform in /out/*; do \
        for model in /src/rlbot/**/*.lt; do cp "$model" "$platform/"; done; \
    done

# Sorting puts the core before the launcher, which bob uses as the entry point.
# Export only platforms produced by this build.
ENTRYPOINT ["tar", "--sort=name", "-C", "/out", "-cf", "-", "."]
