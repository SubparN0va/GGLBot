#pragma once
#include "../Framework.h"
#include "../BasicTypes/Action.h"

namespace RLGC {
	struct PlayerEventState {
		bool goal, save, assist, shot, shotPass, bump, bumped, demo, demoed;

		PlayerEventState() {
			memset(this, 0, sizeof(*this));
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/player_data.py
	struct Player : CarState {

		Player* prev = NULL;

		int index = -1; // Index in the gamestate players array
		uint32_t carId;
		Team team;

		PlayerEventState eventState = {};

		bool ballTouchedStep; // True if the player touched the ball during any of tick of the step
		bool ballTouchedTick; // True if the player is touching the ball on the final tick of the step

		Action prevAction = {};

		// Called before updating to reset the per-step state
		void ResetBeforeStep();

		void UpdateFromCar(Car* car, uint64_t tickCount, int tickSkip, const Action& prevAction, Player* prev);
	};
}