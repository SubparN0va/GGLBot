#include "InferUnit.h"

#include <GigaLearnCPP/Models.h>
#include <GigaLearnCPP/InferenceModels.h>

#include <stdexcept>

GGL::InferUnit::InferUnit(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	InferPartialModelConfig sharedHeadConfig, InferPartialModelConfig policyConfig,
	std::filesystem::path modelsFolder, bool useGPU) :
	obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser), useGPU(useGPU) {

	this->models = std::make_unique<ModelSet>();

	// Let startup handle CUDA initialization errors before announcing the selected device.
	GGL::Infer::MakeInferenceModels(
		obsSize,
		actionParser->GetActionAmount(),
		sharedHeadConfig,
		policyConfig,
		useGPU ? torch::kCUDA : torch::kCPU,
		*this->models
	);
	this->models->Load(modelsFolder, false, false); // loadOptims=false already

	if (useGPU) {
		// Validate the actual policy and synchronize to report CUDA failures at startup.
		RG_NO_GRAD;
		auto options = torch::TensorOptions().device(torch::kCUDA);
		auto sampleObs = torch::zeros({ 1, obsSize }, options);
		auto sampleMasks = torch::ones({ 1, actionParser->GetActionAmount() }, options.dtype(torch::kBool));
		torch::Tensor actions;
		GGL::Infer::InferActions(*models, sampleObs, sampleMasks, true, 1.0f, false, &actions, nullptr);
		actions.cpu();
	}
}

RLGC::Action GGL::InferUnit::InferAction(
	const RLGC::Player& player,
	const RLGC::GameState& state,
	bool deterministic,
	float temperature
) {
	return InferBatch({ &player }, { &state }, deterministic, temperature)[0];
}

std::vector<RLGC::Action> GGL::InferUnit::BatchInferActions(
	const std::vector<RLGC::Player>& players,
	const std::vector<RLGC::GameState>& states,
	bool deterministic,
	float temperature
) {
	RG_ASSERT(players.size() == states.size());
	std::vector<const RLGC::Player*> playerRefs;
	std::vector<const RLGC::GameState*> stateRefs;
	for (size_t i = 0; i < players.size(); ++i) {
		playerRefs.push_back(&players[i]);
		stateRefs.push_back(&states[i]);
	}
	return InferBatch(playerRefs, stateRefs, deterministic, temperature);
}

std::vector<RLGC::Action> GGL::InferUnit::BatchInferActions(
	const std::vector<unsigned>& indices,
	const RLGC::GameState& state,
	bool deterministic,
	float temperature
) {
	std::vector<const RLGC::Player*> playerRefs;
	playerRefs.reserve(indices.size());
	for (unsigned index : indices) {
		if (index >= state.players.size())
			throw std::out_of_range("InferUnit: batch player index is outside the game state");
		playerRefs.push_back(&state.players[index]);
	}
	return InferBatch(playerRefs, std::vector<const RLGC::GameState*>(indices.size(), &state), deterministic, temperature);
}

std::vector<RLGC::Action> GGL::InferUnit::InferBatch(
	const std::vector<const RLGC::Player*>& players,
	const std::vector<const RLGC::GameState*>& states,
	bool deterministic,
	float temperature
) {
	if (players.empty())
		return {};

	int batchSize = (int)players.size();
	std::vector<float> allObs;
	std::vector<uint8_t> allActionMasks;
	allObs.reserve(players.size() * obsSize);
	allActionMasks.reserve(players.size() * actionParser->GetActionAmount());

	for (int i = 0; i < batchSize; i++) {
		FList curObs = obsBuilder->BuildObs(*players[i], *states[i]);
		if ((int)curObs.size() != obsSize) {
			RG_ERR_CLOSE(
				"InferUnit: Obs builder produced an obs that differs from the provided size (expected: " << obsSize << ", got: " << curObs.size() << ")\n"
				"Make sure you provided the correct obs size to the InferUnit constructor.\n"
				"Also, make sure there aren't an incorrect number of players (there are " << states[i]->players.size() << " in this state)"
			);
		}
		allObs += curObs;
		allActionMasks += actionParser->GetActionMask(*players[i], *states[i]);
	}

	std::vector<RLGC::Action> results;
	results.reserve(players.size());

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
			results.push_back(actionParser->ParseAction(actionIndices[i], *players[i], *states[i]));

	}
	catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when inferring model: " << e.what());
	}

	return results;
}


GGL::InferUnit::~InferUnit() = default;
