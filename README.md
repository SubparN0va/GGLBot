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
* When the host builds your bot, the docker will download and use Libtorch CPU Version 2.7.0 by default
  * If you need a specific version of LibTorch for your bot, update the Linux and Windows LibTorch download URLs in `cpp.Dockerfile` (lines 80 and 95).
* Update the `project_name` in bob.toml, and all of bot.toml and loadout.toml to your preference
* If you're using a logo, name it `logo.png` and put it in the `rlbot\` folder

Note: At this point, if you want, you can build your bot using your IDE to ensure it compiles (don't try to run the .exe directly, you'll get an error - this is normal because it must be started by run.bat instead). If you do build your bot before submitting it, make sure you don't include the `out\` or `.vs\` folders and don't include `GGLBot.exe` in the `rlbot\` folder when creating your .zip file.

### 2. Submitting your bot
* Package the following files and folders into a .zip file:
  * `cpp-interface\`
  * `inc\`
  * `rlbot\`
  * `src\`
  * `bob.toml`
  * `CMakeLists.txt`
  * `cpp.Dockerfile`
* Send the .zip file to the host

### 3. Building your bot with bob
Note: If you're submitting your bot to a tournament or Rocket Host, you can skip this step! The host will build your bot for you using the instructions below.
* Install bob the bot builder from `https://github.com/swz-git/bob`
* Install Docker Desktop and make sure it's set to Linux containers
* In command prompt, navigate to your GGLBot root directory. Then enter `<path to bob.exe> build bob.toml`
* In RLBot v5, add the `bob_build\` folder