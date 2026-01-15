#pragma once
#include "RLGymCPP/Framework.h"

namespace RLGC {
	typedef std::vector<float> FList;
	typedef std::vector<int> IList;
}

// Vector append operator
template <typename T>
inline std::vector<T>& operator +=(std::vector<T>& vecA, const std::vector<T>& vecB) {
	vecA.insert(vecA.end(), vecB.begin(), vecB.end());
	return vecA;
}

inline RLGC::FList& operator +=(RLGC::FList& list, float val) {
	list.push_back(val);
	return list;
}

inline RLGC::FList& operator +=(RLGC::FList& list, const Vec& val) {
	list.push_back(val.x);
	list.push_back(val.y);
	list.push_back(val.z);
	return list;
}

///////////////

namespace RLGC {
	template <typename T>
	struct DimList2 {
		size_t size[2];
		size_t numel;
		std::vector<T> data;

		DimList2() {
			memset(size, 0, sizeof(size));
		}

		DimList2(size_t size0, size_t size1) {
			size[0] = size0;
			size[1] = size1;
			numel = size[0] * size[1];

			data.resize(numel);
		}

		size_t ResolveIdx(size_t idx0, size_t idx1) const {
			return idx0*size[1] + idx1;
		}

		T& At(size_t idx0, size_t idx1) { return data[ResolveIdx(idx0, idx1)]; }
		T At(size_t idx0, size_t idx1) const { return data[ResolveIdx(idx0, idx1)]; }

		std::vector<T> GetRow(size_t idx0) const {
			auto startItr = data.begin() + (idx0 * size[1]);
			return std::vector<T>(startItr, startItr + size[1]);
		}

		void Add(const std::vector<T>& newRow) {
			RG_ASSERT(size[1] == newRow.size());
			size[0]++;
			data.insert(data.end(), newRow.begin(), newRow.end());
		}

		void Set(size_t idx0, const std::vector<T>& newRow) {
			RG_ASSERT(size[1] == newRow.size());
			std::copy(newRow.begin(), newRow.end(), data.begin() + idx0 * size[1]);
		}

		bool Defined() const {
			return size[0] > 0;
		}
	};
}