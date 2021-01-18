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

__device__
void tree::initialize(particle* parts, void* data, size_t bytes) {
	int sztot = sizeof(tree) + sizeof(tree*);
	int N = (bytes - opts.nparts * sizeof(int)) / sztot;
	printf("Allocating space for %i trees\n", N);
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

__device__ particle* sort_parts(int* lo, int* hi, particle* swap, particle* b, particle* e, float xmid, int dim) {
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
void root_tree_sort(tree* root, particle* swap_space, particle* pbegin, particle* pend, const range box, int rung) {
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
	range box = trees->boxes[bid];
	trees->poles[bid] = trees->tree_ptrs[bid]->sort(spaces, this_swap, b, e, box, depth, rung);
	__syncthreads();
}

__device__ monopole tree::sort(sort_workspace* workspace, particle* swap_space, particle* pbegin, particle* pend,
		range box_, int depth_, int rung) {
	const int& tid = threadIdx.x;
	const int& block_size = blockDim.x;
	if (tid == 0) {
		part_begin = pbegin;
		part_end = pend;
		depth = depth_;
		leaf = false;
		box = box_;
	}
	float midx;
	particle* mid;
	__syncthreads();
	/*	if (tid == 0) {
	 for (auto* ptr = part_begin; ptr < part_end; ptr++) {
	 if (!box.in_range(ptr->x)) {
	 printf("Particle out of range at depth %i\n", depth);
	 for (int dim = 0; dim < NDIM; dim++) {
	 printf("%e %e %e\n", box.begin[dim], box.end[dim], ptr->x[dim]);
	 }
	 printf("\n");
	 //				__trap();
	 }
	 }
	 }*/
//	printf("Sorting at depth %i\n", depth);
	__syncthreads();
	auto& poles = workspace->poles[tid];
	auto& count = workspace->count[tid];
	if (tid < WARPSIZE) {
		for (int dim = 0; dim < NDIM; dim++) {
			poles[dim] = 0;
		}
		count = float(0);
	}
	__syncthreads();
	if (pend - pbegin > opts.parts_per_bucket) {
//		auto tm = clock();
		midx = (box.begin[2] / float(2) + box.end[2] / float(2));
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
			midx = (box.begin[1] / float(2) + box.end[1] / float(2));
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
			midx = (box.begin[0] / float(2) + box.end[0] / float(2));
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
		__syncthreads();
		if (depth > opts.max_kernel_depth) {
			for (int ci = 0; ci < NCHILD; ci++) {
				particle* swap_base = swap_space + (workspace->begin[ci] - workspace->begin[0]);
				monopole this_pole = children[ci]->sort(workspace + 1, swap_base, workspace->begin[ci],
						workspace->end[ci], workspace->cranges[ci], depth + 1, rung);
				__syncthreads();
				if (tid == 0) {
					workspace->count[ci] = this_pole.mass;
				}
				if (tid < NDIM) {
					workspace->poles[ci][tid] = this_pole.mass * this_pole.xcom[tid];
				}
				__syncthreads();
			}
			__syncthreads();
		} else {
			tree_sort_type* &childdata = workspace->tree_sort;
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
					threadcnt = max(min((int) ((part_end - part_begin) / NCHILD), (int) MAXTHREADCOUNT), WARPSIZE);
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
				float mass = childdata->poles[tid].mass;
				for (int dim = 0; dim < NDIM; dim++) {
					poles[dim] = mass * childdata->poles[tid].xcom[dim];
				}
				count += childdata->poles[tid].mass;
			}
			__syncthreads();
		}
	} else {
		if (tid == 0) {
			leaf = true;
		}
		__syncthreads();
		if (tid < WARPSIZE) {
			for (particle* p = part_begin + tid; p < part_end; p += WARPSIZE) {
				for (int dim = 0; dim < NDIM; dim++) {
					poles[dim] += p->x[dim];
				}
				count += float(1);
				if (p->rung >= rung) {
					int index = atomicAdd(&active_count, 1);
					active_list[index] = (p - part_base);
				}
			}
		}
		__syncthreads();
	}
	__syncthreads();
	for (int P = WARPSIZE / 2; P >= 1; P /= 2) {
		if (tid < P) {
			for (int dim = 0; dim < NDIM; dim++) {
				poles[dim] += workspace->poles[tid + P][dim];
			}
			count += workspace->count[tid + P];
		}
		__syncthreads();
	}
	__syncthreads();
	if (tid == 0) {
		pole.mass = count;

	}
	__syncthreads();
	if (tid < NDIM) {
		if (pole.mass != float(0)) {
			pole.xcom[tid] = workspace->poles[0][tid] / pole.mass;
		} else {
			pole.xcom[tid] = float(0);
		}
	}
	__syncthreads();
	assert(abs(pole.xcom[0]) <= 1.0);
	assert(abs(pole.xcom[1]) <= 1.0);
	assert(abs(pole.xcom[2]) <= 1.0);
	assert(abs(pole.xcom[0]) >= 0.0);
	assert(abs(pole.xcom[1]) >= 0.0);
	assert(abs(pole.xcom[2]) >= 0.0);
	return pole;
}

#define KICKWARPSIZE 32

struct direct_t {
	particle* p;
	const tree* t;
};

__global__
void tree_kick(tree* root, int rung, float dt, double* flops) {
	const int& zi = threadIdx.x;
	const int& dz = blockDim.x;
	const int& yi = blockIdx.x;
	const int& dy = gridDim.y;
	const int& xi = blockIdx.y;
	const int myindex = dz * (dy * xi + yi) + zi;
	__shared__ direct_t* directs;
	__shared__ int next_direct_index;
	if (zi == 0) {
		CUDA_CHECK(cudaMalloc(&directs, 16 * 1024 * sizeof(direct_t)));
		next_direct_index = 0;
	}
	__shared__ int myflops[KICKWARPSIZE];
	myflops[zi] = float(0.0);
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
		bool done = false;
		do {
			const auto& other = *pointers[depth];
			array<float, NDIM> other_x = other.pole.xcom;
			const float w = other.box.end[0] - other.box.begin[0];
			float dist2 = float(0);
			float dist;
			for (int dim = 0; dim < NDIM; dim++) {
				dist = ewald_distance(part.x[dim] - other_x[dim]);
				dist2 += dist * dist;
			}
			myflops[zi] += 15;
			//		printf("%e %e %e\n", other_x[0], other_x[1], other_x[2]);
			assert(abs(other_x[0]) <= 1.0);
			assert(abs(other_x[1]) <= 1.0);
			assert(abs(other_x[2]) <= 1.0);
			assert(abs(other_x[0]) >= 0.0);
			assert(abs(other_x[1]) >= 0.0);
			assert(abs(other_x[2]) >= 0.0);
			dist = SQRT(dist2);
			bool opened = w > opts.opening_crit * dist;
			myflops[zi] += 10;
			assert(!(!opened && depth == 0));
			if (opened && other.leaf) {
				direct_t dir;
				dir.p = &part;
				dir.t = &other;
				int index = atomicAdd(&next_direct_index, 1);
				directs[index] = dir;
			}
			if (!opened || pointers[depth]->leaf) {
				child_indexes[depth] = 0;
				depth--;
				while (depth && child_indexes[depth] == NCHILD - 1) {
					child_indexes[depth] = 0;
					depth--;
				}
				child_indexes[depth]++;
			}
			if (!(child_indexes[0] == NCHILD)) {
				depth++;
				assert(child_indexes[depth - 1] < NCHILD);
				assert(child_indexes[depth - 1] >= 0);
				assert(depth < MAXDEPTH);
				assert(depth >= 0);
				pointers[depth] = pointers[depth - 1]->children[child_indexes[depth - 1]];
			} else {
				done = true;
			}
		} while (!done);

	}
	__syncthreads();
	const int& size = next_direct_index;
	int start = zi * size / KICKWARPSIZE;
	int stop = min((zi + 1) * size / KICKWARPSIZE, size);

	int jindex = start;
	int kindex = 0;
	while (jindex != stop) {
		const auto& t = *directs[jindex].t;
		const auto& sink = *directs[jindex].p;
		if (kindex >= t.part_end - t.part_begin) {
			jindex++;
			kindex = 0;
		}
		if (jindex != stop) {
			const auto& source = t.part_begin[kindex];
			float X[NDIM];
			float F[NDIM];
			float dist2 = float(0);
			for (int dim = 0; dim < NDIM; dim++) {
				float x = sink.x[dim] - source.x[dim];
				X[dim] = copysign(min(x, float(1) - x), x * (float(0.5) - x));
				dist2 += X[dim] * X[dim];
			}
			float dinv = rsqrtf(dist2);
			float dinv3 = dinv * dinv * dinv;
			myflops[zi] += 5;
			for (int dim = 0; dim < NDIM; dim++) {
				F[dim] += X[dim] * dinv3;
			}
			myflops[zi] += 28;
			kindex++;
		}
	}
	__syncthreads();
	for (int P = KICKWARPSIZE / 2; P >= 1; P /= 2) {
		if (zi > P) {
			myflops[zi] += myflops[zi + P];
		}
		__syncthreads();
	}
	atomicAdd(flops, (double) myflops[0]);
	if (zi == 0) {
		CUDA_CHECK(cudaFree(directs));
	}
}

__device__
void tree::kick(tree* root, int rung, float dt) {
	int blocks_needed = (active_count - 1) / KICKWARPSIZE + 1;
	int block_size = SQRT(float(blocks_needed -1 )) + 1;
	assert(block_size*block_size*KICKWARPSIZE >= active_count);
	dim3 dim;
	dim.x = dim.y = block_size;
	dim.z = 1;
	double* flops;
	CUDA_CHECK(cudaMalloc(&flops, sizeof(double)));
	*flops = 0.0;
	tree_kick<<<dim,KICKWARPSIZE>>>(root, rung,dt, flops);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	printf("FLOPS = %e\n", *flops);
	CUDA_CHECK(cudaFree(flops));

}

