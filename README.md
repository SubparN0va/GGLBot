# GGLBot
Creates everything you need to get your bot made with GigaLearn (GGL) up and running in RLBot v5 using bob the bot builder!

## Features
Uses bob the bot builder to generate your GGL bot inside a container and then outputs it to `bob_build\` folder. All you need to do is add libtorch, your Obs Builder, Action Parser, InferUnit config, and .lt models.

## Instructions
* Clone the bob branch of this repo recursively: `git clone --branch bob https://github.com/SubparN0va/GGLBot --recurse-submodules`
* Update `RLBotMain.cpp` with your Obs Builder, Action Parser, and InferUnit config
  * If creating new obs or parser files, make sure you update the `#include` at the top of `RLBotClient.h`
* Put libtorch in the root directory (only the necessary .dlls will be copied by bob to minimize size of the output)
* Put your models (.lt files) into the `rlbot\` folder (these will be copied to the output folder automatically)
* Update the bob.toml project_name, bot.toml and loadout.toml to your preference
* Install bob the bot builder from `https://github.com/swz-git/bob`
* Install Docker Desktop and make sure you switch to Windows containers
* In command prompt, navigate to your GGLBot root directory. Then enter `<path to bob.exe> build bob.toml`
* In RLBot v5, add the `bob_build\` folder