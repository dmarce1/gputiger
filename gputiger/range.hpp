#pragma once

#include <gputiger/params.hpp>
#include <gputiger/array.hpp>

template<class T>
struct range {
	array<T, NDIM> begin;
	array<T, NDIM> end;
	__device__
	void split(array<range, NCHILD>& children) const {
		const int& tid = threadIdx.x;
		if (tid < NCHILD) {
			for (int dim = 0; dim < NDIM; dim++) {
				int mid = (end[dim] + begin[dim]) / T(2);
				bool sw = (tid >> dim) & 1;
				children[tid].begin[dim] = sw ? mid : begin[dim];
				children[tid].end[dim] = sw ? end[dim] : mid;
			}
		}
		__syncthreads();
	}

};
