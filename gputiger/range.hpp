#pragma once

#include <gputiger/params.hpp>
#include <gputiger/array.hpp>

struct range {
	float begin[NDIM];
	float end[NDIM];
	__device__
	void split(array<range, NCHILD>& children) const {
		const int& tid = threadIdx.x;
		if (tid < NCHILD) {
			for (int dim = 0; dim < NDIM; dim++) {
				float mid = (end[dim] / float(2) + begin[dim] / float(2));
				bool sw = (tid >> dim) & 1;
				children[tid].begin[dim] = sw ? mid : begin[dim];
				children[tid].end[dim] = sw ? end[dim] : mid;
			}
		}
		__syncthreads();
	}
	__device__
	bool in_range(array<float, NDIM> v) {
		bool rc = true;
		if (threadIdx.x == 0) {
			for (int dim = 0; dim < NDIM; dim++) {
				if (v[dim] < begin[dim] || v[dim] > end[dim]) {
					rc = false;
					break;
				}
			}
		}
		return rc;
	}
	template<class V>
	__device__
	bool in_range(array<fixed<V>, NDIM> v) {
		bool rc = true;
		if (threadIdx.x == 0) {
			for (int dim = 0; dim < NDIM; dim++) {
				if (v[dim].to_float() < begin[dim] || v[dim].to_float() > end[dim]) {
					rc = false;
					break;
				}
			}
		}
		return rc;
	}

};
