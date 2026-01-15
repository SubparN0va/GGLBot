#include "DefaultAction.h"

RLGC::DefaultAction::DefaultAction() {
	constexpr float
		// Boolean input
		R_B[] = { 0, 1 },

		R_F[] = { -1, 0, 1 };

	// TODO: Use std permutations here or whatever			

	// Ground
	for (float throttle : R_F) {
		for (float steer : R_F) {
			for (float boost : R_B) {
				for (float handbrake : R_B) {
					// Prevent useless throttle when boosting
					if (boost == 1 && throttle != 1)
						continue;

					actions.push_back(
						{
							throttle, steer, 0, steer, 0, 0, boost, handbrake
						}
					);
				}
			}
		}
	}

	int numGroundActions = actions.size();

	// Aerial
	for (float pitch : R_F) {
		for (float yaw : R_F) {
			for (float roll : R_F) {
				for (float jump : R_B) {
					for (float boost : R_B) {
						// Only need roll for sideflip
						if (jump == 1 && yaw != 0)
							continue;

						// Duplicate with ground
						if (pitch == roll && roll == jump && jump == 0)
							continue;

						// Enable handbrake for potential wavedashes
						float handbrake = (jump == 1) && (pitch != 0 || yaw != 0 || roll != 0);

						actions.push_back(
							{
								boost, yaw, pitch, yaw, roll, jump, boost, handbrake
							}
						);
					}
				}
			}
		}
	}

	groundMask.resize(actions.size());
	airMask.resize(actions.size());
	jumpMask.resize(actions.size());
	boostMask.resize(actions.size());

	for (int i = 0; i < actions.size(); i++) {
		Action& action = actions[i];

		if (action.jump)
			jumpMask[i] = true;

		if (action.boost)
			boostMask[i] = true;

		if (i < numGroundActions)
			groundMask[i] = true;

		if (i > numGroundActions && !action.jump)
			airMask[i] = true;

		// Add additional yaw-only actions to air mask
		// These actions were skipped during air action generation to prevent duplicates
		if (i < numGroundActions) {
			if (action.throttle == action.boost && (action.yaw != 0) == action.handbrake) {
				airMask[i] = true;
			}
		}
	}
}

std::vector<uint8_t> RLGC::DefaultAction::GetActionMask(const Player& player, const GameState& state) {
	auto result = std::vector<uint8_t>(actions.size(), false);

	auto fnApplyMask = [&](const std::vector<uint8_t>& mask, bool add) {
		if (add) {
			for (int i = 0; i < actions.size(); i++)
				result[i] |= mask[i];
		} else {
			for (int i = 0; i < actions.size(); i++)
				result[i] &= ~mask[i];
		}
	};

	if (player.isOnGround) {
		fnApplyMask(groundMask, true);
	} else {
		fnApplyMask(airMask, true);
	}

	if (player.boost == 0)
		fnApplyMask(boostMask, false);

	bool isTurtled = player.worldContact.hasContact && player.worldContact.contactNormal.z > 0.9f;
	if (player.HasFlipOrJump() || isTurtled)
		fnApplyMask(jumpMask, true);

	return result;
}
