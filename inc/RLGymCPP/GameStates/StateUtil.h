#pragma once
#include "RLGymCPP/Framework.h"

namespace RLGC {
	PhysState InvertPhys(const PhysState& physState, bool shouldInvert = true);
	PhysState MirrorPhysX(const PhysState& physState, bool shouldMirror = true);
}