#pragma once

#include "GigaLearnCPP/Framework.h"

namespace GGL {
	enum class ModelActivationType {
		RELU,
		LEAKY_RELU,
		SIGMOID,
		TANH
	};

	// Doesn't include inputs or outputs
	// Infer config removes optimType
	struct InferPartialModelConfig {
		std::vector<int> layerSizes = {};
		ModelActivationType activationType = ModelActivationType::RELU;
		bool addLayerNorm = true;
		bool addOutputLayer = true;

		bool IsValid() const {
			return !layerSizes.empty();
		}
	};

	struct ModelConfig : InferPartialModelConfig {
		int numInputs = -1;
		int numOutputs = -1;

		bool IsValid() const {
			return InferPartialModelConfig::IsValid() && numInputs > 0 && (numOutputs > 0 || !addOutputLayer);
		}

		ModelConfig(const InferPartialModelConfig& partialConfig) : InferPartialModelConfig(partialConfig) {}
	};
}