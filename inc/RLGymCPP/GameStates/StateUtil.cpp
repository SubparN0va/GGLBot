#include "StateUtil.h"

PhysState RLGC::InvertPhys(const PhysState& physState, bool shouldInvert) {
	PhysState result = physState;

	if (shouldInvert) {
		constexpr Vec INV_VEC = Vec(-1, -1, 1);

		result.pos *= INV_VEC;
		for (int i = 0; i < 3; i++)
			result.rotMat[i] *= INV_VEC;
		result.vel *= INV_VEC;
		result.angVel *= INV_VEC;
	}

	return result;
}

PhysState RLGC::MirrorPhysX(const PhysState& physState, bool shouldMirror) {

	PhysState result = physState;

	if (shouldMirror) {
		result.pos.x *= -1;

		// Thanks Rolv, JPK, and Kaiyo!
		result.rotMat.forward *= Vec(-1, 1, 1);
		result.rotMat.right *= Vec(1, -1, -1);
		result.rotMat.up *= Vec(-1, 1, 1);

		result.vel.x *= -1;
		result.angVel *= Vec(1, -1, -1);
	}

	return result;
}