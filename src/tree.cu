#include <gputiger/tree.hpp>

__device__
                       static tree** arena;

__device__
static int next_index;

__device__
static int arena_size;

__device__
void tree::initialize(void* data, size_t bytes) {
	int sztot = sizeof(tree) + sizeof(tree*);
	int N = bytes / sztot;
	next_index = 0;
	arena_size = N;
	tree* ptrs = (tree*) data;
	arena = (tree**) (data + sizeof(tree) * N);
	for (int i = 0; i < N; i++) {
		arena[i] = ptrs + i;
	}
	printf("Done allocating trees\n");
}

__device__ tree* tree::alloc() {
	int index = atomicAdd(&next_index, 1);
	if (index < arena_size) {
		return arena[index];
	} else {
		printf("Out of tree memory!\n");
		__trap();
	}
}

__device__ particle* sort_parts(int* lo, int* hi, particle* swap, particle* b, particle* e, pos_type xmid, int dim) {
	const int& tid = threadIdx.x;
	const int& block_size = blockDim.x;
	if (e == b) {
		return e;
	} else {
		if (tid == 0) {
			*lo = 0;
			*hi = (e - b) - 1;
		}
		__syncthreads();
		int index;
		for (particle* part = b + tid; part < e; part += block_size) {
			if (part->x[dim] > xmid) {
				index = atomicSub(hi, 1);
			} else {
				index = atomicAdd(lo, 1);
			}
			swap[index] = *part;
		}
		__syncthreads();
		for (index = tid; index < (e - b); index += block_size) {
			b[index] = swap[index];
		}
		__syncthreads();
		return b + *lo;
	}
}

struct tree_sort_type {
	array<tree*, NCHILD> tree_ptrs;
	array<range<pos_type>, NCHILD> boxes;
	array<particle*, NCHILD> begins;
	array<particle*, NCHILD> ends;
};

__global__
void root_tree_sort(tree* root, particle* swap_space, particle* pbegin, particle* pend, const range<pos_type> box) {
	__shared__ sort_workspace spaces[MAXDEPTH];
	root->sort(spaces, swap_space, pbegin, pend, box, 0);
}

__global__
void tree_sort(tree_sort_type* trees, particle* swap_space, int depth) {
	__shared__ sort_workspace spaces[MAXDEPTH];
	const int& bid = blockIdx.x;
	particle* base = trees->begins[0];
	particle* b = trees->begins[bid];
	particle* e = trees->ends[bid];
	particle* this_swap = swap_space + (b - base);
	range<pos_type> box = trees->boxes[bid];
	trees->tree_ptrs[bid]->sort(spaces, this_swap, b, e, box, depth);
	__syncthreads();
}

__device__ void tree::sort(sort_workspace* workspace, particle* swap_space, particle* pbegin, particle* pend,
		range<pos_type> box_, int depth_) {
	const int& tid = threadIdx.x;
	if (tid == 0) {
		part_begin = pbegin;
		part_end = pend;
		depth = depth_;
		leaf = false;
		box = box_;
	}
	pos_type midx;
	particle* mid;
	__syncthreads();
	if (tid == 0) {
		/*		for (auto* ptr = part_begin; ptr < part_end; ptr++) {
		 if (!box.in_range(ptr->x)) {
		 printf("Particle out of range at depth %i\n", depth);
		 for (int dim = 0; dim < NDIM; dim++) {
		 printf("%i %i %i\n", box.begin[dim], box.end[dim], ptr->x[dim]);
		 }
		 printf("\n");
		 __trap();
		 }
		 }*/
		printf("Sorting at depth %i\n", depth);
	}

	__syncthreads();
	if (pend - pbegin > opts.parts_per_bucket) {
//		auto tm = clock();
		midx = (box.begin[2] / pos_type(2) + box.end[2] / pos_type(2));
		mid = sort_parts(&workspace->lo, &workspace->hi, swap_space, pbegin, pend, midx, 2);
		if (tid == 0) {
			workspace->begin[0] = pbegin;
			workspace->end[0] = workspace->begin[4] = mid;
			workspace->end[4] = pend;
		}
		__syncthreads();
		for (int i = 0; i < 2; i++) {
			particle* b = workspace->begin[4 * i];
			particle* e = workspace->end[4 * i];
			midx = (box.begin[1] / pos_type(2) + box.end[1] / pos_type(2));
			mid = sort_parts(&workspace->lo, &workspace->hi, swap_space, b, e, midx, 1);
			if (tid == 0) {
				workspace->begin[4 * i] = b;
				workspace->end[4 * i] = workspace->begin[4 * i + 2] = mid;
				workspace->end[4 * i + 2] = e;
			}
			__syncthreads();
		}
		for (int i = 0; i < 4; i++) {
			particle* b = workspace->begin[2 * i];
			particle* e = workspace->end[2 * i];
			midx = (box.begin[0] / pos_type(2) + box.end[0] / pos_type(2));
			mid = sort_parts(&workspace->lo, &workspace->hi, swap_space, b, e, midx, 0);
			if (tid == 0) {
				workspace->begin[2 * i] = b;
				workspace->end[2 * i] = workspace->begin[2 * i + 1] = mid;
				workspace->end[2 * i + 1] = e;
			}
			__syncthreads();
		}
		__syncthreads();
		if (tid == 0) {
//			for( int l = 0; l < NCHILD; l++) {
//				printf( "%li %li\n", ((long)begin[l] - (long)part_begin),((long)end[l] -(long) part_begin));
//			}
//			__trap();
			for (int ci = 0; ci < NCHILD; ci++) {
				children[ci] = alloc();
			}
		}
		__syncthreads();
//		float sort_time = ((float) clock() - (float) tm) / opts.clock_rate;
//		if( tid == 0 && depth < 2 ) {
	//		printf( "Sort time = %f\n", sort_time );
//		}
		box.split(workspace->cranges);
		if (depth > opts.max_kernel_depth) {
			for (int ci = 0; ci < NCHILD; ci++) {
				particle* swap_base = swap_space + (workspace->begin[ci] - workspace->begin[0]);
				children[ci]->sort(workspace + 1, swap_base, workspace->begin[ci], workspace->end[ci],
						workspace->cranges[ci], depth + 1);
			}
		} else {
			__shared__ tree_sort_type* childdata;
			if (tid == 0) {
				CUDA_CHECK(cudaMalloc(&childdata, sizeof(tree_sort_type)));
			}
			__syncthreads();
			if (tid < NCHILD) {
				childdata->boxes[tid] = workspace->cranges[tid];
				childdata->tree_ptrs[tid] = children[tid];
				childdata->begins[tid] = workspace->begin[tid];
				childdata->ends[tid] = workspace->end[tid];
			}
			__syncthreads();
			if (tid == 0) {
				tree_sort<<<NCHILD,NCHILD>>>(childdata, swap_space, depth+1);
				CUDA_CHECK(cudaGetLastError());
			}
			__syncthreads();
			if (tid == 0) {
				CUDA_CHECK(cudaDeviceSynchronize());
				CUDA_CHECK(cudaFree(childdata));
			}
			__syncthreads();
		}
		__syncthreads();
	} else {
		if (tid == 0) {
			leaf = true;
//			printf("leaf node at depth %i with %li particles\n", depth, part_end - part_begin);
		}
	}
	__syncthreads();
}

__device__ void tree::destroy() {

}
