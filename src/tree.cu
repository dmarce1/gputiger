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
	arena = (tree**) (data + sizeof(tree)*N);
	for (int i = 0; i < N; i++) {
		arena[i] = ptrs + i;
	}
	printf("Done allocating trees\n");
}

__device__ tree* tree::alloc() {
	int index = atomicAdd(&next_index,1);
	if (index < arena_size) {
		return arena[index];
	} else {
		printf("Out of tree memory!\n");
		__trap();
	}
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
	array<particle*, NCHILD> begins;
	array<particle*, NCHILD> ends;
};

__global__
void root_tree_sort(tree* root, particle* pbegin, particle* pend, const range<pos_type> box) {
	__shared__ sort_workspace spaces[MAXDEPTH];
	root->sort(spaces, pbegin, pend, box, 0);
}

__global__
void tree_sort(tree_sort_type* trees, int depth) {
	__shared__ sort_workspace spaces[MAXDEPTH];
	const int& bid = blockIdx.x;
	particle* b = trees->begins[bid];
	particle* e = trees->ends[bid];
	range<pos_type> box = trees->boxes[bid];
	if( threadIdx.x == 0 ) {
		for( auto* p = b; p < e; p++) {
			if( !box.in_range(p->x)) {
				printf( "error\n");
				__trap();
			}
		}
	}
	__syncthreads();
	trees->tree_ptrs[bid]->sort(spaces, b, e, box, depth);
	__syncthreads();
}

__device__ void tree::sort(sort_workspace* workspace, particle* pbegin, particle* pend, range<pos_type> box_,
		int depth_) {
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
/*	if (tid == 0) {
		for (auto* ptr = part_begin; ptr < part_end; ptr++) {
			if (!box.in_range(ptr->x)) {
				printf("Particle out of range at depth %i\n", depth);
				for (int dim = 0; dim < NDIM; dim++) {
					printf("%i %i %i\n", box.begin[dim], box.end[dim], ptr->x[dim]);
				}
				printf("\n");
				__trap();
			}
		}
	}*/
	if( tid == 0 ) {
		printf( "Sorting at depth %i\n", depth);
	}
	__syncthreads();
	if (pend - pbegin > opts.parts_per_bucket) {
		if (tid == 0) {
			midx = (box.begin[2] / pos_type(2) + box.end[2] / pos_type(2));
			mid = sort_parts(pbegin, pend, midx, 2);
			workspace->begin[0] = pbegin;
			workspace->end[0] = workspace->begin[1] = mid;
			workspace->end[1] = pend;
		}
		__syncthreads();
		if (tid < 2) {
			particle* b = workspace->begin[tid];
			particle* e = workspace->end[tid];
			midx = (box.begin[1] / pos_type(2) + box.end[1] / pos_type(2));
			mid = sort_parts(b, e, midx, 1);
			workspace->begin[2 * tid] = b;
			workspace->end[2 * tid] = workspace->begin[2 * tid + 1] = mid;
			workspace->end[2 * tid + 1] = e;
		}
		__syncthreads();
		if (tid < 4) {
			particle* b = workspace->begin[tid];
			particle* e = workspace->end[tid];
			midx = (box.begin[0] / pos_type(2) + box.end[0] / pos_type(2));
			mid = sort_parts(b, e, midx, 0);
			workspace->begin[2 * tid] = b;
			workspace->end[2 * tid] = workspace->begin[2 * tid + 1] = mid;
			workspace->end[2 * tid + 1] = e;
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
		box.split(workspace->cranges);
		if (depth > opts.max_kernel_depth) {
			for (int ci = 0; ci < NCHILD; ci++) {
				children[ci]->sort(workspace + 1, workspace->begin[ci], workspace->end[ci], workspace->cranges[ci],
						depth + 1);
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
//				printf("Invoking kernel at depth %i\n", depth);
				tree_sort<<<NCHILD,NCHILD>>>(childdata, depth+1);
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
