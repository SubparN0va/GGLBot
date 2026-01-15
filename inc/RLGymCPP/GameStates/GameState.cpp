#include "GameState.h"

#include "RocketSim/Math/Math.h"

using namespace RLGC;

static int boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
static bool boostPadIndexMapBuilt = false;
static std::mutex boostPadIndexMapMutex = {};
void _BuildBoostPadIndexMap(Arena* arena) {
	constexpr const char* ERROR_PREFIX = "_BuildBoostPadIndexMap(): ";
#ifdef RG_VERBOSE
	RG_LOG("Building boost pad index map...");
#endif

	if (arena->_boostPads.size() != CommonValues::BOOST_LOCATIONS_AMOUNT) {
		RG_ERR_CLOSE(
			ERROR_PREFIX << "Arena boost pad count does not match CommonValues::BOOST_LOCATIONS_AMOUNT " <<
			"(" << arena->_boostPads.size() << "/" << CommonValues::BOOST_LOCATIONS_AMOUNT << ")"
		);
	}
	
	bool found[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		Vec targetPos = CommonValues::BOOST_LOCATIONS[i];
		for (int j = 0; j < arena->_boostPads.size(); j++) {
			Vec padPos = arena->_boostPads[j]->config.pos;

			if (padPos.DistSq2D(targetPos) < 10) {
				if (!found[i]) {
					found[i] = true;
					boostPadIndexMap[i] = j;
				} else {
					RG_ERR_CLOSE(
						ERROR_PREFIX << "Matched duplicate boost pad at " << targetPos << "=" << padPos
					);
				}
				break;
			}
		}

		if (!found[i])
			RS_ERR_CLOSE(ERROR_PREFIX << "Failed to find matching pad at " << targetPos);
	}

#ifdef RG_VERBOSE
	RG_LOG(" > Done");
#endif
	boostPadIndexMapBuilt = true;
}

void RLGC::GameState::ResetBeforeStep() {
	for (auto& player : players)
		player.ResetBeforeStep();
}

void RLGC::GameState::UpdateFromArena(Arena* arena, const std::vector<Action>& actions, GameState* prev) {
	this->prev = prev;
	if (prev) {
		episodeId = prev->episodeId;
		episodeStep = prev->episodeStep + 1;
		prev->prev = NULL;
	}

	lastArena = arena;
	int tickSkip = RS_MAX(arena->tickCount - lastTickCount, 0);
	deltaTime = tickSkip * (1 / 120.f);

	ball = arena->ball->GetState();

	std::vector<Car*> cars;
	arena->GetCarsOrdered(cars);

	players.resize(cars.size());

	// Safety: actions should match the player count.
	// If you *know* they always match, you can omit this, but it's useful for catching bugs.
	if (!actions.empty() && actions.size() < cars.size()) {
		RG_ERR_CLOSE("GameState::UpdateFromArena(): actions.size() < cars.size() ("
			<< actions.size() << " < " << cars.size() << ")");
	}

	for (int i = 0; i < (int)cars.size(); i++) {
		auto& player = players[i];
		player.index = i;

		const Action& a = actions.empty() ? Action{} : actions[i]; // if actions can be empty sometimes
		player.UpdateFromCar(cars[i], arena->tickCount, tickSkip, a, prev ? &prev->players[i] : NULL);

		if (player.ballTouchedStep)
			lastTouchCarID = player.carId;
	}

	if (!boostPadIndexMapBuilt) {
		boostPadIndexMapMutex.lock();
		// Check again? This seems stupid but also makes sense to me
		//	Without this, we could lock as the index map is building, then go build again
		//	I would like to keep the mutex inside the if statement so it is only checked a few times
		if (!boostPadIndexMapBuilt) 
			_BuildBoostPadIndexMap(arena);
		boostPadIndexMapMutex.unlock();
	}

	int numBoostPads = arena->_boostPads.size();
	boostPads.resize(numBoostPads);
	boostPadsInv.resize(numBoostPads);
	boostPadTimers.resize(numBoostPads);
	boostPadTimersInv.resize(numBoostPads);
	for (int i = 0; i < arena->_boostPads.size(); i++) {
		int idx = boostPadIndexMap[i];
		int invIdx = boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1];

		auto state = arena->_boostPads[idx]->GetState();
		auto stateInv = arena->_boostPads[invIdx]->GetState();

		boostPads[i] = state.isActive;
		boostPadsInv[i] = stateInv.isActive;

		boostPadTimers[i] = state.cooldown;
		boostPadTimersInv[i] = stateInv.cooldown;
	}

	// Update goal scoring
	// If you don't have a GoalScoreCondition then that's not my problem lmao
	goalScored = arena->IsBallScored();

	lastTickCount = arena->tickCount;
}
