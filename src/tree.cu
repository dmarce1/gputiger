#include <gputiger/tree.hpp>
#include <gputiger/math.hpp>

__device__
                               static tree** arena;

__device__
static int next_index;

__device__
static int arena_size;

__device__
static int* active_list;

__device__
static int active_count;

__device__
               static particle* part_base;

struct tree_sort_type {
	array<tree*, NCHILD> tree_ptrs;
	array<range<pos_type>, NCHILD> boxes;
	array<particle*, NCHILD> begins;
	array<particle*, NCHILD> ends;
	array<monopole, NCHILD> poles;
};

__device__
void tree::initialize(particle* parts, void* data, size_t bytes) {
	int sztot = sizeof(tree) + sizeof(tree*);
	int N = (bytes - opts.nparts * sizeof(int)) / sztot;
	next_index = 0;
	arena_size = N;
	tree* ptrs = (tree*) data;
	arena = (tree**) (data + sizeof(tree) * N);
	active_list = (int*) (data + (sizeof(tree) + sizeof(tree*)) * N);
	active_count = 0;
	for (int i = 0; i < N; i++) {
		arena[i] = ptrs + i;
	}
	::part_base = parts;
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

__global__
void root_tree_sort(tree* root, particle* swap_space, particle* pbegin, particle* pend, const range<pos_type> box,
		int rung) {
	__shared__ sort_workspace spaces[MAXDEPTH];
	root->sort(spaces, swap_space, pbegin, pend, box, 0, rung);
}

__global__
void tree_sort(tree_sort_type* trees, particle* swap_space, int depth, int rung) {
	__shared__ sort_workspace spaces[MAXDEPTH];
	const int& bid = blockIdx.x;
	particle* base = trees->begins[0];
	particle* b = trees->begins[bid];
	particle* e = trees->ends[bid];
	particle* this_swap = swap_space + (b - base);
	range<pos_type> box = trees->boxes[bid];
	trees->poles[bid] = trees->tree_ptrs[bid]->sort(spaces, this_swap, b, e, box, depth, rung);
	__syncthreads();
}

__device__ monopole tree::sort(sort_workspace* workspace, particle* swap_space, particle* pbegin, particle* pend,
		range<pos_type> box_, int depth_, int rung) {
	const int& tid = threadIdx.x;
	const int& block_size = blockDim.x;
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
	/*if (tid == 0) {
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
	 printf("Sorting at depth %i\n", depth);
	 }*/

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
			for (int ci = 0; ci < NCHILD; ci++) {
				children[ci] = alloc();
			}
		}
		__syncthreads();
		box.split(workspace->cranges);
		int64_t count = 0;
		if (depth > opts.max_kernel_depth) {
			for (int ci = 0; ci < NCHILD; ci++) {
				particle* swap_base = swap_space + (workspace->begin[ci] - workspace->begin[0]);
				monopole this_pole = children[ci]->sort(workspace + 1, swap_base, workspace->begin[ci],
						workspace->end[ci], workspace->cranges[ci], depth + 1, rung);
				if (tid == 0) {
					count += this_pole.mass;
				}
				__syncthreads();
				if (tid < NDIM) {
					workspace->poles[ci][tid] = this_pole.xcom[tid];
				}
				__syncthreads();
			}
			__syncthreads();
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
				int threadcnt;
				if (depth == opts.max_kernel_depth) {
					threadcnt = WARPSIZE;
				} else {
					threadcnt = min((int) ((part_end - part_begin) / NCHILD), (int) MAXTHREADCOUNT);
				}
				tree_sort<<<NCHILD,threadcnt>>>(childdata, swap_space, depth+1, rung);
				CUDA_CHECK(cudaGetLastError());
			}
			__syncthreads();
			if (tid == 0) {
				CUDA_CHECK(cudaDeviceSynchronize());
				CUDA_CHECK(cudaFree(childdata));
			}
			__syncthreads();
			if (tid < NCHILD) {
				for (int dim = 0; dim < NDIM; dim++) {
					workspace->poles[tid][dim] = childdata->poles[tid].xcom[dim];
				}
				if (tid == 0) {
					count += childdata->poles[tid].mass;
				}
			}
			__syncthreads();
		}
		for (int P = NCHILD / 2; P >= 1; P /= 2) {
			for (int dim = 0; dim < NDIM; dim++) {
				if (tid < P) {
					for (int dim = 0; dim < NDIM; dim++) {
						workspace->poles[tid][dim] += workspace->poles[tid + P][dim];
					}
				}
			}
			__syncthreads();
		}
		if (tid == 0) {
			pole.mass = count;
		}
		__syncthreads();
		if (tid < NDIM) {
			pole.xcom[tid] = workspace->poles[0][tid] / count;
		}
		__syncthreads();
	} else {
		if (tid == 0) {
			leaf = true;
		}
		if (tid < WARPSIZE) {
			auto& poles = workspace->poles[tid];
			auto& count = workspace->count[tid];
			for (int dim = 0; dim < NDIM; dim++) {
				poles[dim] = 0;
			}
			count = 0;
			for (particle* p = part_begin + tid; p < part_end; p += block_size) {
				for (int dim = 0; dim < NDIM; dim++) {
					poles[dim] += p->x[dim];
				}
				if (p->rung >= rung) {
					int index = atomicAdd(&active_count, 1);
					active_list[index] = (p - part_base);
				}
			}
			for (int P = block_size / 2; P >= 1; P /= 2) {
				if (tid < P) {
					for (int dim = 0; dim < NDIM; dim++) {
						poles[dim] += workspace->poles[tid + P][dim];
					}
				}
				__syncthreads();
			}
			if (tid == 0) {
				pole.mass = count;
			}
			__syncthreads();
			if (tid < NDIM) {
				pole.xcom[tid] = pos_type(poles[tid] / pole.mass);
			}
			__syncthreads();
		}
		__syncthreads();
	}
	__syncthreads();
	return pole;
}

#define KICKWARPSIZE 32
#define KICKBLOCKSIZE 128

__global__
void tree_kick(tree* root, int baseindex, int rung, float dt) {
	const int myindex = baseindex + blockIdx.x * blockDim.x + threadIdx.x;
	const float theta2 = POW(opts.opening_crit, 2);
	vector<tree*> directs;
	if (myindex < active_count) {
		int depth;
		particle& part = *(part_base + active_list[myindex]);
		tree* pointers[MAXDEPTH];
		int child_indexes[MAXDEPTH];
		for (int i = 0; i < MAXDEPTH; i++) {
			child_indexes[i] = 0;
		}
		pointers[0] = root;
		depth = 0;
		do {
			auto& other = *(pointers[depth]);
			array<pos_type, NDIM> other_x = other.pole.xcom;
			uint64_t w = other.box.end[0] - other.box.begin[0];
			uint64_t dist2 = 0;
			for (int dim = 0; dim < NDIM; dim++) {
				int64_t dist = part.x[dim] - other_x[dim];
				dist2 += dist * dist;
			}
			bool opened = float(w * w) > theta2 * float(dist2);
			if (opened && pointers[depth]->leaf) {
				directs.push_back(pointers[depth]);
			}
			if (!opened) {

				// Do indirect

			}

			if (!opened || pointers[depth]->leaf) {
				do {
					depth--;
					child_indexes[depth]++;
				} while (child_indexes[depth] == NCHILD && depth != 0);
			}
			depth++;
			child_indexes[depth] = 0;

			pointers[depth] = pointers[depth - 1]->children[child_indexes[depth - 1]];
		} while (depth);
	}
	__syncthreads();
	if (myindex < active_count) {
		int depth;
		particle& part = *(part_base + active_list[myindex]);
		tree* pointers[MAXDEPTH];
		int child_indexes[MAXDEPTH];
		for (int i = 0; i < MAXDEPTH; i++) {
			child_indexes[i] = 0;
		}
		pointers[0] = root;
		depth = 0;
		do {
			auto& other = *(pointers[depth]);
			array<pos_type, NDIM> other_x = other.pole.xcom;
			uint64_t w = other.box.end[0] - other.box.begin[0];
			uint64_t dist2 = 0;
			for (int dim = 0; dim < NDIM; dim++) {
				int64_t dist = part.x[dim] - other_x[dim];
				dist2 += dist * dist;
			}
			dist2 = max(dist2, int64_t(1) << int64_t(60));
			bool opened = float(w * w) > theta2 * float(dist2);
			if (opened && pointers[depth]->leaf) {
				printf("Error - ewald should not be direct\n");
				__trap();
			}
			if (!opened) {

				// Do indirect ewald

			}

			if (!opened || pointers[depth]->leaf) {
				do {
					depth--;
					child_indexes[depth]++;
				} while (child_indexes[depth] == NCHILD && depth != 0);
			}
			depth++;
			child_indexes[depth] = 0;

			pointers[depth] = pointers[depth - 1]->children[child_indexes[depth - 1]];
		} while (depth);
	}
	__syncthreads();
	if (myindex < active_count) {
		int part_index = 0;
		int tree_index = 0;
		particle* other = directs[tree_index]->part_begin;
		while (tree_index < directs.size()) {

			// Do direct

			part_index++;
			if (other + part_index == directs[tree_index]->part_end) {
				part_index = 0;
				tree_index++;
			}
		}
	}
	__syncthreads();

}

__device__
void tree::kick(tree* root, int rung, float dt) {
	for (int i = 0; i < active_count; i += KICKBLOCKSIZE * KICKWARPSIZE) {
		tree_kick<<<KICKBLOCKSIZE,KICKWARPSIZE>>>(root,i, rung,dt);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

}

