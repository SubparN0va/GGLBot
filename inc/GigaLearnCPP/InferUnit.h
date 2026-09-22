#pragma once

#include <RLGymCPP/GameStates/GameState.h>
#include <RLGymCPP/BasicTypes/Action.h>
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>

#include "InferenceModelConfig.h"
#include <memory>
#include <filesystem>

namespace RLGC {
	class ObsBuilder;
	class ActionParser;
	struct Player;
	struct GameState;
	struct Action;
}

namespace GGL {

	struct ModelSet;

	struct RG_IMEXPORT InferUnit {
		int obsSize = 0;
		RLGC::ObsBuilder* obsBuilder = nullptr;      // not owned
		RLGC::ActionParser* actionParser = nullptr;  // not owned
		std::unique_ptr<ModelSet> models;
		bool useGPU = false;

		// NOTE: Reset() will never be called on your obs builder here.
		InferUnit(
			RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
			InferPartialModelConfig sharedHeadConfig, InferPartialModelConfig policyConfig,
			std::filesystem::path modelsFolder, bool useGPU);

		~InferUnit(); // frees models

		RLGC::Action InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature = 1);
		std::vector<RLGC::Action> BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature = 1);

		/** One model call for cars in the same packet, preserving the supplied index order. **/
		std::vector<RLGC::Action> BatchInferActions(const std::vector<unsigned>& indices, const RLGC::GameState& state, bool deterministic, float temperature = 1);

	private:
		/** Borrow observations' source objects so batching does not copy whole game states. **/
		std::vector<RLGC::Action> InferBatch(const std::vector<const RLGC::Player*>& players, const std::vector<const RLGC::GameState*>& states, bool deterministic, float temperature);
	};
}
