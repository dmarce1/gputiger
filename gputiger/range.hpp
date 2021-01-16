#pragma once

#include <gputiger/params.hpp>
#include <gputiger/array.hpp>

template<class T>
struct range {
	T begin[NDIM];
	T end[NDIM];
	__device__
	void split(array<range, NCHILD>& children) const {
		const int& tid = threadIdx.x;
		if (tid < NCHILD) {
			for (int dim = 0; dim < NDIM; dim++) {
				int mid = (end[dim] / T(2) + begin[dim] / T(2));
				bool sw = (tid >> dim) & 1;
				children[tid].begin[dim] = sw ? mid : begin[dim];
				children[tid].end[dim] = sw ? end[dim] : mid;
			}
		}
		__syncthreads();
	}
	__device__
	bool in_range(array<T, NDIM> v) {
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

};
