#include "InferUnit.h"

#include <GigaLearnCPP/Models.h>
#include <GigaLearnCPP/InferenceModels.h>

GGL::InferUnit::InferUnit(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	InferPartialModelConfig sharedHeadConfig, InferPartialModelConfig policyConfig,
	std::filesystem::path modelsFolder, bool useGPU) :
	obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser), useGPU(useGPU) {

	this->models = std::make_unique<ModelSet>();

	try {
		GGL::Infer::MakeInferenceModels(
			obsSize,
			actionParser->GetActionAmount(),
			sharedHeadConfig,
			policyConfig,
			useGPU ? torch::kCUDA : torch::kCPU,
			*this->models
		);
	}
	catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to construct models: " << e.what());
	}

	try {
		this->models->Load(modelsFolder, false, false); // loadOptims=false already
	}
	catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to load models: " << e.what());
	}
}

RLGC::Action GGL::InferUnit::InferAction(
	const RLGC::Player& player,
	const RLGC::GameState& state,
	bool deterministic,
	float temperature
) {
	return BatchInferActions({ player }, { state }, deterministic, temperature)[0];
}

std::vector<RLGC::Action> GGL::InferUnit::BatchInferActions(
	const std::vector<RLGC::Player>& players,
	const std::vector<RLGC::GameState>& states,
	bool deterministic,
	float temperature
) {
	RG_ASSERT(players.size() > 0 && states.size() > 0);
	RG_ASSERT(players.size() == states.size());

	int batchSize = (int)players.size();
	std::vector<float> allObs;
	std::vector<uint8_t> allActionMasks;

	for (int i = 0; i < batchSize; i++) {
		FList curObs = obsBuilder->BuildObs(players[i], states[i]);
		if ((int)curObs.size() != obsSize) {
			RG_ERR_CLOSE(
				"InferUnit: Obs builder produced an obs that differs from the provided size (expected: " << obsSize << ", got: " << curObs.size() << ")\n"
				"Make sure you provided the correct obs size to the InferUnit constructor.\n"
				"Also, make sure there aren't an incorrect number of players (there are " << states[i].players.size() << " in this state)"
			);
		}
		allObs += curObs;
		allActionMasks += actionParser->GetActionMask(players[i], states[i]);
	}

	std::vector<RLGC::Action> results;

	try {
		RG_NO_GRAD;

		auto device = useGPU ? torch::kCUDA : torch::kCPU;

		auto tObs = torch::tensor(allObs).reshape({ (int64_t)batchSize, (int64_t)obsSize }).to(device);
		auto tMasks = torch::tensor(allActionMasks).reshape({ (int64_t)batchSize, (int64_t)actionParser->GetActionAmount() }).to(device);

		torch::Tensor tActions, tLogProbs;

		GGL::Infer::InferActions(
			*models,
			tObs,
			tMasks,
			deterministic,
			temperature,
			/*halfPrec*/ false,
			&tActions,
			&tLogProbs
		);

		auto actionIndices = TENSOR_TO_VEC<int>(tActions);

		for (int i = 0; i < batchSize; i++)
			results.push_back(actionParser->ParseAction(actionIndices[i], players[i], states[i]));

	}
	catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when inferring model: " << e.what());
	}

	return results;
}


GGL::InferUnit::~InferUnit() = default;
