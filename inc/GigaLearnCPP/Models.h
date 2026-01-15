#pragma once

#include <RLGymCPP/Framework.h>
#include <GigaLearnCPP/FrameworkTorch.h>
#include <GigaLearnCPP/InferenceModelConfig.h>

#include <torch/torch.h>

namespace GGL {

	inline void AddActivationFunc(torch::nn::Sequential& seq, ModelActivationType type) {
		switch (type) {
		case ModelActivationType::RELU:       seq->push_back(torch::nn::ReLU());       return;
		case ModelActivationType::LEAKY_RELU: seq->push_back(torch::nn::LeakyReLU());  return;
		case ModelActivationType::SIGMOID:    seq->push_back(torch::nn::Sigmoid());    return;
		case ModelActivationType::TANH:       seq->push_back(torch::nn::Tanh());       return;
		}
		RG_ERR_CLOSE("Unknown activation function type: " << (int)type);
	}
	
	inline std::vector<int64_t> GetSeqParamNumels(const torch::nn::Sequential& seq) {
		std::vector<int64_t> out;
		for (const auto& p : seq->parameters(/*recurse=*/true))
			out.push_back(p.numel());
		return out;
	}

	class Model : public torch::nn::Module {
	public:
		std::string modelName;
		torch::Device device = torch::kCPU;

		// Main network + optional bf16 mirror for faster inference
		torch::nn::Sequential seq{ nullptr };
		torch::nn::Sequential seqHalf{ nullptr };
		bool _seqHalfOutdated = true;

		// Needed by InferenceModels.cpp (uses config.numOutputs)
		ModelConfig config = InferPartialModelConfig{};

		Model() = default;

		Model(const char* name, const ModelConfig& cfg, torch::Device dev)
			: modelName(name ? name : ""), device(dev), config(cfg) {

			if (!config.IsValid())
				RG_ERR_CLOSE("Failed to create model \"" << modelName << "\" with invalid config");

			seq = register_module("seq", torch::nn::Sequential());
			seqHalf = register_module("seqHalf", torch::nn::Sequential());

			int lastSize = config.numInputs;

			// Hidden layers
			for (int i = 0; i < (int)config.layerSizes.size(); i++) {
				const int sz = config.layerSizes[i];
				seq->push_back(torch::nn::Linear(lastSize, sz));
				if (config.addLayerNorm) {
					seq->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ (int64_t)sz })));
				}
				AddActivationFunc(seq, config.activationType);
				lastSize = sz;
			}

			// Output layer optional
			if (config.addOutputLayer) {
				seq->push_back(torch::nn::Linear(lastSize, config.numOutputs));
			}
			else {
				// If no output layer, output size == last hidden size
				config.numOutputs = lastSize;
			}

			seq->to(device);
			_seqHalfOutdated = true;
		}

		std::filesystem::path GetSuffixedSavePath(const std::filesystem::path& folder, const std::string& suffix) const {
			std::string filename = modelName + suffix;
			for (char& c : filename) c = (char)std::toupper((unsigned char)c);
			filename += ".lt";
			return folder / filename;
		}

		std::filesystem::path GetSavePath(const std::filesystem::path& folder) const {
			return GetSuffixedSavePath(folder, "");
		}

		static std::filesystem::path FindModelFile(const std::filesystem::path& folder, const std::string& nameUpperLt) {
			auto pUpper = folder / nameUpperLt;
			if (std::filesystem::exists(pUpper)) return pUpper;

			std::string lower = nameUpperLt;
			for (char& c : lower) c = (char)std::tolower((unsigned char)c);
			auto pLower = folder / lower;
			if (std::filesystem::exists(pLower)) return pLower;

			return pUpper; // default (for error message)
		}

		torch::Tensor Forward(torch::Tensor input, bool halfPrec) {
			// In inference builds, gradients should be off; but guard anyway.
			if (torch::GradMode::is_enabled())
				halfPrec = false;

			if (!halfPrec) {
				return seq->forward(input);
			}

			// Build/refresh bf16 mirror if needed
			if (_seqHalfOutdated) {
				_seqHalfOutdated = false;

				if (seqHalf->size() == 0) {
					for (auto& mod : *seq)
						seqHalf->push_back(mod.clone());
					seqHalf->to(RG_HALFPERC_TYPE, true);
				}
				else {
					auto fromParams = seq->parameters(true);
					auto toParams = seqHalf->parameters(true);
					RG_ASSERT(fromParams.size() == toParams.size());

					for (int i = 0; i < (int)fromParams.size(); i++) {
						auto scaled = fromParams[i].to(RG_HALFPERC_TYPE, true);
						toParams[i].copy_(scaled, true);
					}
				}
			}

			auto halfInput = input.to(RG_HALFPERC_TYPE);
			auto halfOut = seqHalf->forward(halfInput);
			return halfOut.to(torch::kFloat);
		}

		void Load(const std::filesystem::path& folder, bool allowNotExist) {
			auto expectedName = GetSavePath(folder).filename().string();
			auto path = FindModelFile(folder, expectedName);

			if (!std::filesystem::exists(path)) {
				if (allowNotExist) {
					RG_LOG("Warning: Model \"" << modelName << "\" not found in " << folder << " (skipping)");
					return;
				}
				RG_ERR_CLOSE("Model \"" << modelName << "\" does not exist in " << folder << " (looked for " << path << ")");
			}

			std::ifstream in(path, std::ios::binary);
			in >> std::noskipws;
			if (!in.good())
				RG_ERR_CLOSE("Failed to load from " << path << " (cannot access)");

			auto sizesBefore = GetSeqParamNumels(seq);

			try {
				torch::load(seq, in, device);
			}
			catch (const std::exception& e) {
				RG_ERR_CLOSE("Failed to load model \"" << modelName << "\" from " << path << "\nException: " << e.what());
			}

			auto sizesAfter = GetSeqParamNumels(seq);
			if (sizesBefore != sizesAfter) {
				RG_ERR_CLOSE(
					"Loaded model \"" << modelName << "\" has different parameter sizes than expected.\n"
					"This usually means your config (layer sizes / layernorm / activation / output layer) doesn't match the checkpoint."
				);
			}

			_seqHalfOutdated = true;
		}
	};

	class ModelSet {
	public:
		std::map<std::string, Model*> map;

		Model* operator[](const std::string& name) {
			auto it = map.find(name);
			return (it == map.end()) ? nullptr : it->second;
		}

		void Add(Model* model) {
			RG_ASSERT(model);
			map[model->modelName] = model;
		}

		// keep signature compatible with your InferUnit.cpp call: Load(folder, allowNotExist, loadOptims)
		void Load(const std::filesystem::path& folder, bool allowNotExist, bool /*loadOptimsIgnored*/) {
			for (auto& kv : map)
				kv.second->Load(folder, allowNotExist);
		}

		void Free() {
			for (auto& kv : map)
				delete kv.second;
			map.clear();
		}

		~ModelSet() {
			Free();
		}
	};

} // namespace GGL
