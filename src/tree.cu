#include <gputiger/tree.hpp>

__device__
           static stack<tree*>* arena;

__device__
           static mutex mtx;

__device__
          static tree_params params;

__device__
void tree::initialize(tree_params params_, tree* ptr, size_t bytes) {
	CUDA_CHECK(cudaMalloc(&arena, sizeof(stack<tree> )));
	params = params_;
	int count = bytes / sizeof(tree);
	for (int i = 0; i < count; i++) {
		arena->push(ptr + i);
	}
}
__device__ tree* tree::alloc() {
	lock_guard<mutex> lock(mtx);
	if (arena->size()) {
		tree* ptr = arena->top();
		arena->pop();
		return ptr;
	} else {
		printf("Out of tree memory!\n");
		__trap();
	}
}

__device__
void tree::free(tree* ptr) {
	lock_guard<mutex> lock(mtx);
	arena->push(ptr);
}

__device__ particle* sort_parts(particle* b, particle* e, pos_type xmid, int dim) {
	if (e == b) {
		return e;
	} else {
		particle* lo = b;
		particle* hi = e;
		while (lo < hi) {
			if (lo->x[dim] > xmid) {
				while (lo != hi) {
					hi--;
					if (hi->x[dim] < xmid) {
						particle tmp = *lo;
						*lo = *hi;
						*hi = tmp;
						break;
					}
				}
			}
			lo++;
		}
		return hi;
	}
}

struct tree_sort_type {
	array<tree*, NCHILD> tree_ptrs;
	array<range<pos_type>, NCHILD> boxes;
	array<particle*,NCHILD > begins;
	array<particle*, NCHILD> ends;
};

__global__
void tree_sort(tree_sort_type* trees, int depth) {
	const int& bid = blockIdx.x;
	trees->tree_ptrs[bid]->sort(trees->begins[bid], trees->ends[bid], trees->boxes[bid], depth);
	__syncthreads();
}

__device__ void tree::sort(particle* pbegin, particle* pend, const range<pos_type>& box_, int depth_) {
	const int& tid = threadIdx.x;
	__shared__ particle* begin[NCHILD];
	__shared__ particle* end[NCHILD];
	__shared__ array<range<pos_type>, NCHILD> cranges;
	if (tid == 0) {
		part_begin = pbegin;
		part_end = pend;
		depth = depth_;
		leaf = false;
		box = box_;
	}
	__syncthreads();
	pos_type midx;
	particle* mid;
	__syncthreads();
	if (pend - pbegin > params.parts_per_bucket) {
		if (tid == 0) {
			midx = (box.begin[2] + box.end[2]) / pos_type(2);
			mid = sort_parts(pbegin, pend, midx, 2);
			begin[0] = pbegin;
			end[0] = begin[1] = mid;
			end[1] = pend;
		}
		__syncthreads();
		if (tid < 2) {
			particle* b = begin[tid];
			particle* e = end[tid];
			midx = (box.begin[1] + box.end[1]) / pos_type(2);
			mid = sort_parts(b, e, midx, 1);
			begin[2 * tid] = b;
			end[2 * tid] = begin[2 * tid + 1] = mid;
			end[2 * tid + 1] = e;
		}
		__syncthreads();
		if (tid < 4) {
			particle* b = begin[tid];
			particle* e = end[tid];
			midx = (box.begin[0] + box.end[0]) / pos_type(2);
			mid = sort_parts(b, e, midx, 0);
			begin[2 * tid] = b;
			end[2 * tid] = begin[2 * tid + 1] = mid;
			end[2 * tid + 1] = e;
		}
		__syncthreads();
		if (tid == 0) {
			for (int ci = 0; ci < NCHILD; ci++) {
				children[ci] = alloc();
			}
		}
		__syncthreads();
		box.split(cranges);
		if (depth > params.kernel_depth) {
			for (int ci = 0; ci < NCHILD; ci++) {
				children[ci]->sort(begin[ci], end[ci], cranges[ci], depth + 1);
			}
		} else {
			__shared__ tree_sort_type* childdata;
			if (tid == 0) {
				CUDA_CHECK(cudaMalloc(&childdata, sizeof(tree_sort_type)));
			}
			__syncthreads();
			if (tid < NCHILD) {
				childdata->boxes[tid] = cranges[tid];
				childdata->tree_ptrs[tid] = children[tid];
				childdata->begins[tid] = begin[tid];
				childdata->ends[tid] = end[tid];
			}
			__syncthreads();
			if (tid == 0) {
				tree_sort<<<NCHILD,NCHILD>>>(childdata, depth+1);
				CUDA_CHECK(cudaGetLastError());
				CUDA_CHECK(cudaDeviceSynchronize());
				CUDA_CHECK(cudaFree(childdata));
			}
		}
		if (tid == 0) {
			leaf = true;
		}
		__syncthreads();
	}
	__syncthreads();
}

__device__ void tree::destroy() {

}
