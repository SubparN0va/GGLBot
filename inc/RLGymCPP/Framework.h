#pragma once

#include "RocketSim/RocketSim.h"
#include "RocketSim/Sim/GameEventTracker/GameEventTracker.h"

// Use RocketSim namespace
using namespace RocketSim;

// Define our own log
#define RG_LOG(s) { std::cout << s << std::endl; }

#define RG_NO_COPY(className) \
className(const className&) = delete;  \
className& operator= (const className&) = delete

#define RG_ERR_CLOSE(s) { \
std::string _errorStr = RS_STR("RG FATAL ERROR: " << s); \
RG_LOG(_errorStr); \
throw std::runtime_error(_errorStr); \
exit(EXIT_FAILURE); \
}

#ifndef RG_UNSAFE
#define RG_ASSERT(cond) { if (!(cond)) { RG_ERR_CLOSE("Assertion failed: " << #cond); } }
#else
#define RG_ASSERT(cond) {}
#endif

#define RG_DIVIDER std::string(40, '=')