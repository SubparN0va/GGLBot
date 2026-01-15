#include "Player.h"

namespace RLGC {
	void Player::ResetBeforeStep() {
		this->eventState = {};
		debugRewardsWeighted.clear();
	}

	void Player::UpdateFromCar(Car* car, uint64_t tickCount, int tickSkip, const Action& prevAction, Player* prev) {

		this->prev = prev;
		if (prev)
			prev->prev = NULL;

		carId = car->id;
		team = car->team;
		*(CarState*)this = car->GetState();

		if (ballHitInfo.isValid) {
			ballTouchedStep = ballHitInfo.tickCountWhenHit >= (tickCount - tickSkip);
			ballTouchedTick = ballHitInfo.tickCountWhenHit == (tickCount - 1);
		} else {
			ballTouchedStep = ballTouchedTick = false;
		}

		this->prevAction = prevAction;
	}
}