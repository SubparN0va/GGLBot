# GGLBot
Creates everything you need to get your bot made with GigaLearn (GGL) up and running in RLBot v5.

## Features
Creates a stripped down .exe file (~3MB) that runs your GGL bot without all the learning/optimization fluff. All you need to do is add your Obs Builder, Action Parser, InferUnit config, and .lt models.
Additionally, since the agent_id in the .exe is set by the bot.toml file and libtorch is in a central location, after you build your .exe, you can copy/paste the entire `rlbot\` folder after building. Then you only need to change the bot.toml and model files to quickly add different versions of your bot to RLBot - as long as they use the same obs/parser/model setup.

## Instructions
* Clone this repo recursively: `git clone https://github.com/SubparN0va/GGLBot --recurse-submodules`
* Update `RLBotMain.cpp` with your Obs Builder, Action Parser, and InferUnit config
  * If creating new obs or parser files, make sure you update the `#include` at the top of `RLBotClient.h`
* Make sure the project can locate libtorch (see note below)
* Build using Release mode. This will automatically place the .exe into the `rlbot\` folder
* Put your models (.lt files) into the `rlbot\` folder
* Update the bot.toml and loadout.toml to your preference
* In RLBot v5, add the `rlbot\` folder

## Note
This project expects the cpu version of libtorch to be located in `%LOCALAPPDATA%\RLBot5\bots\libtorch_cpu`. Eventually, RLBot v5 may ship with this, but for now you can add it manually. If you want to put it in a different location, make sure you update both the rlbot\run.bat (bot.toml uses run.bat to set the path to libtorch before opening the .exe) and CMakeLists.txt to point to it. If submitting a bot to a tournament, leave this at the default location.