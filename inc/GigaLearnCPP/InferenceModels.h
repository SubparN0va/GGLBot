#pragma once

#include <torch/torch.h>
#include <GigaLearnCPP/InferenceModelConfig.h>

namespace GGL {
	class ModelSet;
}

namespace GGL::Infer {

	// Builds only what inference needs: shared_head + policy
	void MakeInferenceModels(
		int obsSize,
		int numActions,
		const InferPartialModelConfig& sharedHeadConfig,
		const InferPartialModelConfig& policyConfig,
		torch::Device device,
		ModelSet& outModels
	);

	void InferActions(
		ModelSet& models,
		torch::Tensor obs,
		torch::Tensor actionMasks,
		bool deterministic,
		float temperature,
		bool halfPrec,
		torch::Tensor* outActions,
		torch::Tensor* outLogProbs
	);

} // namespace GGL::Infer
