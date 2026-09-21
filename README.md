# GGLBot - Bob Branch
The main purpose of this branch of GGLBot is to submit your bot to tournaments or Rocket Host since it allows the host to build your bot using bob the bot builder. If you are just wanting to play your bot in RLBot, the `main` branch is likely what you want to use since it is a simpler process.

## Instructions
The instructions are broken down into three steps:
1. Preparing your bot
2. Submitting your bot
3. (Optional) Building your bot with bob

If you are submitting your bot, you only need to worry about step one and two. The host will build your bot for you.

### 1. Preparing your bot for submission
* Clone the bob branch of this repo recursively: `git clone --branch bob https://github.com/SubparN0va/GGLBot --recurse-submodules`
* Update `RLBotMain.cpp` with your Obs Builder, Action Parser, and InferUnit config
  * If creating new obs or parser files, make sure you update the `#include` at the top of `RLBotClient.h`. The Docker builds in a Linux container, which means the headers are case sensitive. Make sure they match exactly, and keep in mind if you're using MSVC it is NOT case sensitive, which means if you get it wrong you won't know until the host builds it in the Linux container.
* Put your models (.lt files) into the `rlbot\` folder (these will be copied to the output folder automatically at build time)
* When the host builds your bot, the docker will download LibTorch **2.14.0 + CUDA 12.6** for Windows (`libtorch-win-shared-with-deps-2.14.0+cu126.zip`) and cross-compile the Windows bot against it. Only the Windows binary is built – the tournament runs on Windows, so the Linux build from the original branch was removed to halve the host's build time. (The CMake and launcher code paths for Linux are still present if anyone needs them.)
  * The core runs inference on the GPU. There is deliberately **no CPU fallback**: if CUDA is unavailable at startup the bot prints an error and exits, so a broken GPU setup is caught immediately rather than silently running slow.
  * The Windows LibTorch URL in `cpp.Dockerfile` must be the **exact same archive** the host has extracted into `%LOCALAPPDATA%\RLBot5\bots\libtorch\`. Do not change it unless the host changes their installed version.
* Update the `project_name` in bob.toml, and all of bot.toml and loadout.toml to your preference
* If you're using a logo, name it `logo.png` and put it in the `rlbot\` folder

The build produces a small native launcher named `GGLBot` and a Torch-linked core in `000-runtime\`. On Windows the launcher searches for the shared CUDA LibTorch runtime at `libtorch\lib` (it must contain `torch_cuda.dll`), first near the bot and then under `%LOCALAPPDATA%\RLBot5\bots\`. On Linux it searches for RLBot's CPU `torch-archive` as before. This allows a `bob_build\` package to run from any location without bundling the multi-GB LibTorch runtime.

Note: At this point, if you want, you can build your bot using your IDE to ensure it compiles – see **Opening the project in Visual Studio** below, which has two setup steps that are *not optional*. Running it locally additionally needs an NVIDIA GPU that the cu126 build supports (GeForce 10-series through 40-series; **RTX 50-series is not supported by cu126** – use the matching `cu130` zip for local testing on those cards). Test the executable through RLBot v5 so it receives the required RLBot connection environment. If you do build your bot before submitting it, make sure you don't include the `out\` or `.vs\` folders and don't include generated executables or the `rlbot\000-runtime\` folder when creating your .zip file.

### Opening the project in Visual Studio
Opening the folder in Visual Studio ("Open a local folder" / CMake project) **will show errors on almost every header** until two things have happened. This is expected; it is not a broken checkout. Do them in this order:

1. **Extract the CUDA LibTorch zip *before* opening the folder.** Visual Studio runs CMake the moment the folder opens, and CMake expects the same zip the host uses at `%LOCALAPPDATA%\RLBot5\bots\libtorch\` (so that `%LOCALAPPDATA%\RLBot5\bots\libtorch\lib\torch_cuda.lib` exists). No CUDA Toolkit install is needed – only the zip. If it is missing, CMake stops with `Could not find torch.lib in ...`, IntelliSense has no include paths, and *everything* project-related is underlined – `RLGymCPP/GameStates/...` "cannot be opened", `FList` unknown, and so on. To see the actual message: **View → Output**, "Show output from: CMake". After extracting the zip, run **Project → Delete Cache and Reconfigure**. (Alternative: set `-DLIBTORCH_ROOT=<path to extracted libtorch>` in the CMake settings.)
   * The first configure also needs internet access: cpp-interface downloads flatbuffers and RLBot's schema and compiles `flatc` during configure. Expect it to take a minute or two.
2. **Build once: Build → Build All.** `rlbot/Bot.h` includes `interfacepacket_generated.h`, which does not exist in the source tree – it is generated into the build folder by running `flatc` on RLBot's schema as part of the build. Until the first build, everything in `rlbot::flat::` (`GamePacket`, `BallPrediction`, ...) and every type in `RLBotClient.h` that uses them is an IntelliSense error. Building the `rlbot-generated` target alone is enough to generate the headers, but a full build also confirms the CUDA link works on your machine.

After both steps, close and reopen any file that still shows squiggles so IntelliSense rescans. Other things to know:
* **Don't let Visual Studio edit `CMakeLists.txt` when you add files.** `GGLBotCore` picks up everything under `inc\` and `src\` automatically via `file(GLOB_RECURSE ...)`. If VS offers to add new sources to the CMake targets, decline – it tends to append them to *both* executables, and the launcher (`add_executable(GGLBot launcher/GGLBotLauncher.cpp)`) must stay exactly that line: it has no Torch and no `inc\` include path, so it will not compile with bot sources added.
* **Header paths are case sensitive on the host.** MSVC accepts `RLGymCPP/Gamestates/StateUtil.h`; the Linux container that bob builds in does not (the folder is `GameStates`). VS will not warn you. Check the case of every `#include` you add.
* The default VS configuration (`x64-Debug`) builds `RelWithDebInfo` with the Ninja generator and drops the results in `out\build\x64-Debug\`; the post-build step also copies `GGLBot.exe` to `rlbot\` and `GGLBotCore.exe` to `rlbot\000-runtime\`. Both locations are gitignored and excluded from the Docker build.

### 2. Submitting your bot
* Package the following files and folders into a .zip file:
  * `cmake\` (the Windows cross-compile toolchain – **the build fails without it**)
  * `cpp-interface\`
  * `inc\`
  * `launcher\`
  * `rlbot\`
  * `src\`
  * `.dockerignore`
  * `bob.toml`
  * `CMakeLists.txt`
  * `cpp.Dockerfile`
* Do **not** include `.git\`, `.vs\`, `out\`, `bob_build\`, `rlbot\000-runtime\`, `bob.exe` or any other `.exe`
* Quick check before sending – from inside the folder, this must succeed: `bob build bob.toml`. If it builds for you it will build for the host
* Send the .zip file to the host

### 3. Building your bot with bob
Note: If you're submitting your bot to a tournament or Rocket Host, you can skip this step! The host will build your bot for you using the instructions below.
* Install bob the bot builder from `https://github.com/swz-git/bob`
* Install Docker Desktop and make sure it's set to Linux containers
* In command prompt, navigate to your GGLBot root directory. Then enter `<path to bob.exe> build bob.toml`
* Install the shared CUDA LibTorch runtime (one time, shared by every GPU bot):
  * Download `https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-2.14.0%2Bcu126.zip` (~5 GB)
  * Extract it into `%LOCALAPPDATA%\RLBot5\bots\` so that `%LOCALAPPDATA%\RLBot5\bots\libtorch\lib\torch_cuda.dll` exists
  * Only the NVIDIA driver is required (CUDA 12.6 needs driver 560 or newer); the zip bundles cuBLAS, cuDNN and the CUDA runtime, so no CUDA Toolkit install is needed
* In RLBot v5, add the `bob_build\` folder; it can remain outside the RLBot bots directory
* On startup the bot prints `GGLBot: CUDA available (1 device) -> running inference on GPU`. If it prints `GGLBot: CUDA is NOT available - refusing to start.` instead, the GPU/driver/runtime setup needs fixing before it will run
