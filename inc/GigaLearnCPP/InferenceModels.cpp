#include <GigaLearnCPP/InferenceModels.h>

#include <GigaLearnCPP/Models.h>

namespace {

	// Copied/minimized from PPOLearner::InferPolicyProbsFromModels
	torch::Tensor InferPolicyProbsFromModels(
		GGL::ModelSet& models,
		torch::Tensor obs,
		torch::Tensor actionMasks,
		float temperature,
		bool halfPrec
	) {
		actionMasks = actionMasks.to(torch::kBool);

		constexpr float ACTION_MIN_PROB = 1e-11f;
		constexpr float ACTION_DISABLED_LOGIT = -1e10f;

		// Guard against bad temperature
		if (!(temperature > 0.f)) temperature = 1.f;

		if (models["shared_head"])
			obs = models["shared_head"]->Forward(obs, halfPrec);

		auto logits = models["policy"]->Forward(obs, halfPrec) / temperature;

		auto probs = torch::softmax(
			logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(),
			-1
		);

		// Keep shape stable and avoid exact zeros
		return probs
			.view({ -1, models["policy"]->config.numOutputs })
			.clamp(ACTION_MIN_PROB, 1.0f);
	}

} // anonymous namespace

namespace GGL::Infer {

	void MakeInferenceModels(
		int obsSize,
		int numActions,
		const InferPartialModelConfig& sharedHeadConfig,
		const InferPartialModelConfig& policyConfig,
		torch::Device device,
		ModelSet& outModels
	) {
		// Policy: obs -> logits(numActions)
		ModelConfig fullPolicyConfig = policyConfig;
		fullPolicyConfig.numInputs = obsSize;
		fullPolicyConfig.numOutputs = numActions;

		if (sharedHeadConfig.IsValid()) {
			ModelConfig fullSharedHeadConfig = sharedHeadConfig;
			fullSharedHeadConfig.numInputs = obsSize;
			fullSharedHeadConfig.numOutputs = 0;

			RG_ASSERT(!sharedHeadConfig.addOutputLayer);

			int featSize = fullSharedHeadConfig.layerSizes.back();
			fullPolicyConfig.numInputs = featSize;

			outModels.Add(new Model("shared_head", fullSharedHeadConfig, device));
		}

		outModels.Add(new Model("policy", fullPolicyConfig, device));
	}

	void InferActions(
		ModelSet& models,
		torch::Tensor obs,
		torch::Tensor actionMasks,
		bool deterministic,
		float temperature,
		bool halfPrec,
		torch::Tensor* outActions,
		torch::Tensor* outLogProbs
	) {
		auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec);

		if (deterministic) {
			auto action = probs.argmax(1);
			if (outActions)  *outActions = action.flatten();
			if (outLogProbs) *outLogProbs = torch::Tensor(); // empty
		}
		else {
			auto action = torch::multinomial(probs, 1, true);
			auto logProb = torch::log(probs).gather(-1, action);

			if (outActions)  *outActions = action.flatten();
			if (outLogProbs) *outLogProbs = logProb.flatten();
		}
	}

} // namespace GGL::Infer
