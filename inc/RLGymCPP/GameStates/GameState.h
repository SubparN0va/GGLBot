#pragma once
#include "Player.h"
#include "../CommonValues.h"
#include "../BasicTypes/Action.h"

namespace RLGC {
	struct ScoreLine {
		int teamGoals[2] = { 0,0 };

		int operator[](size_t index) const {
			return teamGoals[index];
		}

		int& operator[](size_t index) {
			return teamGoals[index];
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/game_state.py
	struct GameState {

		GameState* prev = NULL;

		float deltaTime = 0; // Time that has passed since last update

		bool goalScored = false; // If the ball is in the goal
		int lastTouchCarID = -1;
		std::vector<Player> players;

		BallState ball;

		std::vector<bool> boostPads, boostPadsInv;
		std::vector<float> boostPadTimers, boostPadTimersInv;

		// Last arena we updated with
		// Can be used to determine current arena from within reward function, for example
		// NOTE: Could be null
		Arena* lastArena = NULL;

		// Last tick count when updated
		uint64_t lastTickCount = 0;

		void* userInfo = NULL;

		GameState() {
			boostPads = std::vector<bool>(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
			boostPadsInv = std::vector<bool>(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
			boostPadTimers = std::vector<float>(CommonValues::BOOST_LOCATIONS_AMOUNT, 0);
			boostPadTimersInv = std::vector<float>(CommonValues::BOOST_LOCATIONS_AMOUNT, 0);
		}
		explicit GameState(Arena* arena) {
			UpdateFromArena(arena, std::vector<Action>(arena->_cars.size()), NULL);
		}

		const auto& GetBoostPads(bool inverted) const {
			return inverted ? boostPadsInv : boostPads;
		}

		const auto& GetBoostPadTimers(bool inverted) const {
			return inverted ? boostPadTimers : boostPadTimersInv;
		}

		// Called before updating to reset the per-step state
		void ResetBeforeStep();

		void UpdateFromArena(Arena* arena, const std::vector<Action>& actions, GameState* prev);

		bool IsEmpty() const {
			return players.empty();
		}

		void MakeEmpty() {
			players.clear();
		}
	};
}