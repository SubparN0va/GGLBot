# GGLBot - Bob GPU Branch
The main purpose of this branch of GGLBot is to submit your bot to tournaments or Rocket Host since it allows the host to build your bot using [bob the bot builder](https://github.com/swz-git/bob). If you are just wanting to play your bot in RLBot, the main branch is likely what you want to use since it is a simpler process.

The key feature of the bob-gpu branch is that it can run your bot on the GPU instead of only the CPU, which usually leads to much faster inference times.

## 1. Preparing your bot

* Clone recursively: `git clone --branch bob-gpu https://github.com/SubparN0va/GGLBot --recurse-submodules`.
* Update `src/RLBotMain.cpp` with the observation builder, action parser, model configuration, and tick skip used during training.
* Update the includes in `src/RLBotClient.h` when adding builders or parsers. Match filename case exactly; bob builds in a Linux container.
* Put your `.lt` models in `rlbot/`.
* GPU is the default. For CPU, change `rlbot/device.txt` from `gpu` to `cpu`. GPU bots support Windows; CPU bots support Windows and Linux.
* Set `project_name` in `bob.toml` and customize `rlbot/bot.toml` and `rlbot/loadout.toml`. Put an optional `logo.png` in `rlbot/`.

## 2. Submitting your bot

With Python 3.10 or newer installed, run this from the repository root:

```text
python scripts/package_bot.py
```

This packages everything you need to run your bot into a `submission.zip` file. The script includes your source, models, and configuration, and leaves out build outputs and LibTorch. No build is required.

Send the generated `submission.zip` to the host. The host builds your bot using the bob instructions below.

## 3. Building your bot (optional)

To test your bot before submitting it, set up LibTorch and use either build method below.

### LibTorch setup

For **GPU**, put the complete Windows release CUDA LibTorch folder here:

```text
%LOCALAPPDATA%/RLBot5/bots/libtorch/
```

Keep its `include/`, `lib/`, and `build-version` intact. For a fully accurate comparison, use the same version as the host. Running the bot requires a compatible NVIDIA GPU and driver.

For **CPU**, the build uses the bot pack's existing `%LOCALAPPDATA%/RLBot5/bots/torch-archive/torch/` installation.

For a custom location, set `LIBTORCH_CUDA_ROOT` (GPU) or `LIBTORCH_CPU_ROOT` (CPU) to the folder containing `include/` and `lib/`. These environment variables work for both building and running the bot.

### Build with bob

* Install [bob](https://github.com/swz-git/bob), Docker Desktop with Linux containers, and Python 3.10 or newer.
* From the repository root, run:

```text
python scripts/build_bob.py --bob "<path to bob.exe>"
```

Omit `--bob` if bob is on `PATH`. To use a different LibTorch folder for this build, add `--libtorch "<path to torch or libtorch>"`.

The helper reuses your local Windows LibTorch installation. CPU builds also download and cache the Linux CPU bot-pack archive; GPU builds skip that download. LibTorch DLLs stay in the shared bot pack.

Add the resulting `bob_build/` folder in RLBot v5 to run the bot.

### Build with Visual Studio or CMake

Install Visual Studio's C++ tools, CMake, and Ninja. Open the project folder and select `windows-gpu-release` or `windows-cpu-release`. The preset's device takes precedence over `rlbot/device.txt`.

From an x64 Visual Studio developer prompt, you can also run:

```text
cmake --preset windows-gpu-release
cmake --build --preset windows-gpu-release --target GGLBot
```

Use the corresponding `-relwithdebinfo` preset for debug symbols. Windows Debug builds are unsupported by the shared release libraries. For a custom LibTorch folder, you can also pass `-DLIBTORCH_CUDA_ROOT="<path>"` or `-DLIBTORCH_CPU_ROOT="<path>"` when configuring.

The build copies the bot into `rlbot/`. Add that folder in RLBot v5 to test it. Rebuild when changing devices.

On Linux, use `linux-cpu-release` or `linux-cpu-relwithdebinfo` with the CPU bot-pack runtime installed. Its default location is `$XDG_DATA_HOME/RLBot5/bots/torch-archive/torch/`, or `~/.local/share/RLBot5/bots/torch-archive/torch/`.

## Additional notes

### Batched inference

Batched inference evaluates matching teammates together in a single model call to reduce inference overhead. Each car gets its own action from its own observations and action mask, and retains its own delayed controls. Batching itself adds no shared decision-making or team coordination.

Enable batched inference with `hivemind = true` in `rlbot/bot.toml`. `hivemind` is RLBot's required configuration key for grouping matching teammates into one process. Set `hivemind = false` to run each car in a separate process.
