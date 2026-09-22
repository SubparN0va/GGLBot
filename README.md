# GGLBot - Bob GPU Branch
Build a CPU bot for Windows and Linux, or a CUDA GPU bot for Windows. Each build contains one core per supported platform and uses the selected shared LibTorch runtime from the bot pack.

## 1. Prepare your bot

* Clone recursively: `git clone --branch bob-gpu https://github.com/SubparN0va/GGLBot --recurse-submodules`.
* Set the observation builder, action parser, model configuration, and tick skip in `src/RLBotMain.cpp` to match training.
* Update includes in `src/RLBotClient.h` when adding builders or parsers. Include paths must match filename case for Linux builds.
* Put your `.lt` models in `rlbot/`.
* Set `project_name` in `bob.toml`, then customize `rlbot/bot.toml` and `rlbot/loadout.toml`. An optional `logo.png` also goes in `rlbot/`.

## 2. Choose CPU or GPU

Set `rlbot/device.txt` before building:

| Value | Bob package | Required Windows SDK |
| --- | --- | --- |
| `cpu` (checked in) | Windows and Linux CPU | Bot pack's `torch-archive/torch` |
| `cuda` | Windows GPU only | Bot pack's `libtorch` |

The selection is compiled into the launcher and core. Rebuild to change devices. A CUDA build requires a working NVIDIA GPU and compatible driver; initialization failures report an error and exit. Linux GPU builds are currently unsupported.

### Shared LibTorch locations

Builds and launchers search beside `rlbot/` (the launcher searches beside itself), through four parent directories, then the standard RLBot bot pack:

| Runtime | Standard location |
| --- | --- |
| Windows CPU | `%LOCALAPPDATA%/RLBot5/bots/torch-archive/torch` |
| Windows CUDA | `%LOCALAPPDATA%/RLBot5/bots/libtorch` |
| Linux CPU | `$XDG_DATA_HOME/RLBot5/bots/torch-archive/torch`, or `~/.local/share/RLBot5/bots/torch-archive/torch` |

CPU uses the bot pack's CPU Torch installation. GPU uses the existing Windows release CUDA LibTorch installation, including its `include/`, `lib/`, and `build-version`. The CUDA archive includes its own `torch_cpu` library; a separate CPU archive is not needed. Match the build SDK to the runtime on the machine running the bot.

For a custom location, set `LIBTORCH_CPU_ROOT` or `LIBTORCH_CUDA_ROOT` to the selected SDK's root (the directory containing `include/` and `lib/`). The Python helper and launcher read these environment variables. CMake accepts the same variables as `-D` options, or reads the environment on the first configure. An explicit override is exclusive; an invalid path reports an error.

## 3. Build with bob

Install [bob the bot builder](https://github.com/swz-git/bob), Docker Desktop with Linux containers, and Python 3.10 or newer. From the repository root, run:

```text
python scripts/build_bob.py
```

If bob is not on `PATH`, pass `--bob "<path to bob.exe>"`. To override the selected Windows SDK for this build, pass `--libtorch "<path to torch or libtorch>"`; this takes precedence over the environment variable. Use `--out-dir "<directory>"` to change the default `bob_build/` output.

The helper finds the selected local SDK, temporarily stages its headers and required import libraries for Docker, and removes the staged files when bob finishes. Windows LibTorch is never downloaded. DLLs stay in the shared bot pack and are not included in the container or submission. Use the helper for both CPU and CUDA builds.

CPU builds also download and cache the Linux CPU bot-pack archive for the Linux executable. Its URL is `LIBTORCH_LINUX_CPU_URL` in `cpp.Dockerfile`; change that default if the host uses a different Linux archive. CUDA builds skip this download and all Linux/CPU compilation. Docker still downloads and caches compiler tools as needed.

Install the selected shared runtime on the machine running the bot, then add `bob_build/` in RLBot v5. The output may remain outside the bot-pack directory.

## 4. Build locally with Visual Studio or CMake

Install Visual Studio's C++ tools, CMake, and Ninja. Open the project folder and choose the preset for the device you want:

| Preset | Output |
| --- | --- |
| `windows-cpu-release` | Windows CPU, optimized |
| `windows-cpu-relwithdebinfo` | Windows CPU, optimized with debug symbols |
| `windows-cuda-release` | Windows GPU, optimized |
| `windows-cuda-relwithdebinfo` | Windows GPU, optimized with debug symbols |
| `linux-cpu-release` | Linux CPU, optimized |
| `linux-cpu-relwithdebinfo` | Linux CPU, optimized with debug symbols |

A preset's device takes precedence over `rlbot/device.txt`. From an x64 Visual Studio developer prompt, for example:

```text
cmake --preset windows-cpu-relwithdebinfo
cmake --build --preset windows-cpu-relwithdebinfo --target GGLBot
```

For a custom SDK, append `-DLIBTORCH_CPU_ROOT="<path to torch>"` or `-DLIBTORCH_CUDA_ROOT="<path to libtorch>"` to the configure command. You can also save overrides in an ignored `CMakeUserPresets.json`. Without a preset, CMake uses `rlbot/device.txt` unless you pass `-DGGLBOT_DEVICE=cpu` or `-DGGLBOT_DEVICE=cuda`.

Windows builds use Release or RelWithDebInfo because the shared libraries use the release MSVC ABI. If Visual Studio uses an old `CMakeSettings.json`, enable CMakePresets integration and reconfigure. The first configure fetches cpp-interface dependencies; build once to generate FlatBuffers headers for IntelliSense.

Results go under `out/build/<preset>/`. The build copies the launcher to `rlbot/` and the selected core to `rlbot/000-runtime/`. Run through RLBot v5 to supply the connection environment. Sources under `inc/` and `src/` belong to the core; the launcher must remain independent of Torch.

## 5. Submit your bot

Zip these files and directories:

* `cmake/`, `build-support/libtorch/README.md`, `cpp-interface/`, `inc/`, `launcher/`, and `src/`
* `rlbot/` with models and configuration
* `scripts/build_bob.py`, `.dockerignore`, `bob.toml`, `CMakeLists.txt`, `CMakePresets.json`, and `cpp.Dockerfile`

Exclude `.git/`, `.vs/`, `out/`, `bob_build/`, generated executables, `rlbot/000-runtime/`, and generated `build-support/libtorch/local/`. The host builds with `scripts/build_bob.py` and the selected Windows SDK installed. Test your models with the runtime and hardware you will use.

## Checks

The helper and launcher checks cover exclusive device selection, local discovery and overrides, missing SDKs, staging cleanup, cache fingerprints, argument forwarding, and failure handling:

```text
cmake -S tests -B out/build/launcher-tests -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/launcher-tests
ctest --test-dir out/build/launcher-tests --output-on-failure
```

Use an x64 Visual Studio developer prompt on Windows. These checks require Python and CMake, but no LibTorch or GPU. The helper checks can also run with `python tests/test_bob_build.py`.

