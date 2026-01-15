#pragma once
#include "RLGymCPP/Gamestates/GameState.h"
#include "RLGymCPP/BasicTypes/Action.h"
#include "RLGymCPP/BasicTypes/Lists.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/obs_builders/obs_builder.py
namespace RLGC {
	// TODO: Only designed for discrete actions currently 
	class ActionParser {
	public:
		virtual Action ParseAction(int actionIdx, const Player& player, const GameState& state) = 0;
		virtual int GetActionAmount() = 0;

		// Returns true or false for each action, depending on if it is available in the current situation
		// Not using std::vector<bool> because it has major issues (see https://isocpp.org/blog/2012/11/on-vectorbool)
		virtual std::vector<uint8_t> GetActionMask(const Player& player, const GameState& state) {
			return std::vector<uint8_t>(GetActionAmount(), true);
		}
	};
}