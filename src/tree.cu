#include <gputiger/tree.hpp>
#include <gputiger/math.hpp>

__device__
                                                                                                static tree* tree_base;

__device__
static int next_index;

__device__
static int arena_size;

__device__
static int* leaf_list;

__device__
static int leaf_count;

__device__
void tree::initialize(particle* parts, void* data, size_t bytes) {
	int sztot = sizeof(tree) + sizeof(int);
	int N = bytes / sztot;
	printf("Allocating space for %i trees\n", N);
	next_index = 0;
	arena_size = N;
	tree_base = (tree*) data;
	leaf_list = (int*) (data + sizeof(tree) * N);
	leaf_count = 0;
	printf("Done allocating trees\n");
}

__device__ tree* tree::alloc() {
	int index = atomicAdd(&next_index, 1);
	if (index >= arena_size) {
		printf("Out of tree memory!\n");
		__trap();
	}
	return tree_base + index;
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
		if (depth_ >= MAXDEPTH) {
			printf("Maximum depth exceeded in sort\n");
			__trap();
		}
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
		float max_span = 0.0;
		int long_dim;
		for (int dim = 0; dim < NDIM; dim++) {
			if (box.end[dim] - box.begin[dim] > max_span) {
				max_span = box.end[dim] - box.begin[dim];
				long_dim = dim;
			}
		}
		midx = (box.begin[long_dim] / float(2) + box.end[long_dim] / float(2));
		mid = sort_parts(&workspace->lo, &workspace->hi, swap_space, pbegin, pend, midx, long_dim);
		if (tid == 0) {
			workspace->cranges[0] = box;
			workspace->cranges[1] = box;
			workspace->cranges[0].end[long_dim] = midx;
			workspace->cranges[1].begin[long_dim] = midx;
			workspace->begin[0] = pbegin;
			workspace->end[0] = workspace->begin[1] = mid;
			workspace->end[1] = pend;

		}
		__syncthreads();
		if (tid == 0) {
			for (int ci = 0; ci < NCHILD; ci++) {
				children[ci] = alloc();
			}
		}
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
			int index = atomicAdd(&leaf_count, 1);
			leaf_list[index] = this - tree_base;
		}
		__syncthreads();
		if (tid < WARPSIZE) {
			for (particle* p = part_begin + tid; p < part_end; p += WARPSIZE) {
				for (int dim = 0; dim < NDIM; dim++) {
					poles[dim] += p->x[dim];
				}
				count += float(1);
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
	const tree* self;
	const tree* other;
};

#define PARTMAX 128
#define OTHERSMAX (256)

__global__
void tree_kick(tree* root, int rung, float dt, double* flops) {
	const int& tid = threadIdx.x;
	const int& yi = blockIdx.x;
	const int& xi = blockIdx.y;
	const int myindex = (gridDim.x * xi + yi);
	__shared__ float h2;
	__shared__ int myflops[KICKWARPSIZE];
	myflops[tid] = 0.f;
	 bool opened;
	__shared__ array<float, NDIM> F[PARTMAX];
	__shared__ array<float, NDIM> others1[OTHERSMAX];
	__shared__ array<float, NDIM> others2[OTHERSMAX];
	__shared__ bool swtch;
	__shared__ array<float, NDIM>* others;
	__shared__ array<float, NDIM>* next_others;
	__shared__ int other_cnt;
	if (tid == 0) {
		swtch = false;
		others = others1;
		next_others = others2;
		other_cnt = 0;
		h2 = opts.hsoft*opts.hsoft;
	}
	__syncthreads();
	int ndirect = 0;
	int nindirect = 0;
	for (int i = tid; i < PARTMAX; i += KICKWARPSIZE) {
		for (int dim = 0; dim < NDIM; dim++) {
			F[i][dim] = float(0);
		}
	}
	__syncthreads();

	const auto accumulate = [&]() {
		__syncthreads();
		const tree& self = *(tree_base + leaf_list[myindex]);
		for (auto* sink = self.part_begin; sink < self.part_end; sink++) {
			const auto& sink_x = sink->x;
			const int count = min(other_cnt,OTHERSMAX);
			for (int oi = tid; oi < count; oi += KICKWARPSIZE) {
				array<float, NDIM> X;
				{
					const auto& source_x = others[oi];
					for (int dim = 0; dim < NDIM; dim++) {
						const float x = source_x[dim] - sink_x[dim];
						const float absx = fabs(x);  // 1
						X[dim] = copysignf(fminf(absx, 1.f - absx), x * (0.5f - absx)); // 5
					}
				}
				{
					float Xinv3 = rsqrtf(fmaxf(X[0] * X[0] + X[1] * X[1] + X[2] * X[2],h2));
					Xinv3 = Xinv3 * Xinv3 * Xinv3;
					const int index = sink - self.part_begin;
					for (int dim = 0; dim < NDIM; dim++) {
						F[index][dim] += X[dim] * Xinv3; // 2
					}
				}
				myflops[tid] += 42;
			}
		}
		__syncthreads();
		if (tid == 0) {
			other_cnt -= OTHERSMAX;
			swtch = !swtch;
			if( swtch ) {
				others = others2;
				next_others = others1;
			} else {
				others = others1;
				next_others = others2;
			}
		}
		__syncthreads();

	};
	if (myindex < leaf_count) {
		int depth;
		const tree& self = *(tree_base + leaf_list[myindex]);
		tree* pointers[MAXDEPTH];
		int child_indexes[MAXDEPTH];
		for (int i = 0; i < MAXDEPTH; i++) {
			child_indexes[i] = 0;
		}
		pointers[0] = root;
		depth = 0;
		bool done = false;
		float self_w = 0.0;
		for (int dim = 0; dim < NDIM; dim++) {
			self_w += POW(self.box.end[dim] - self.box.begin[dim], 2);
		}
		self_w = SQRT(self_w) / float(2);
		array<float, NDIM> self_x = self.pole.xcom;
		do {
			const auto& other = *pointers[depth];
			array<float, NDIM> other_x = other.pole.xcom;
			float other_w = 0.0f;
			float self_w = 0.0f;
			for (int dim = 0; dim < NDIM; dim++) {
				other_w += pow2(other.box.end[dim] - other.box.begin[dim]);
			}
			other_w = SQRT(other_w) / 2.f;
			float dist2 = 0.f;
			float dist;
			for (int dim = 0; dim < NDIM; dim++) {
				dist = ewald_distance(self_x[dim] - other_x[dim]);
				dist2 += dist * dist;
			}
			dist = SQRT(dist2);
			opened = (self_w + other_w) > opts.opening_crit * dist;
			assert(!(!opened && depth == 0));
			__syncthreads();
			if (opened && other.leaf) {
				ndirect++;
				for (auto* source = other.part_begin + tid; source < other.part_end; source += KICKWARPSIZE) {
					const auto source_x = source->x;
					int index = atomicAdd(&other_cnt, 1);
					(index < OTHERSMAX ? others[index] :next_others[index - OTHERSMAX]) = source_x;
				}
			} else {
				if (tid == 0) {
					others[other_cnt++] = other_x;
				}
				nindirect++;
			}
			__syncthreads();
			if( other_cnt >= OTHERSMAX) {
				accumulate();
			}
			__syncthreads();
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
			__syncthreads();
		} while (!done);
	}
	accumulate();
	__syncthreads();
	for (int P = KICKWARPSIZE / 2; P >= 1; P /= 2) {
		if (tid < P) {
			myflops[tid] += myflops[tid + P];
		}
		__syncthreads();
	}
	if (tid == 0) {
		atomicAdd(flops, myflops[0]);
	}
	__syncthreads();
//	printf("%i %i\n", ndirect, nindirect);
}

__device__
void tree::kick(tree* root, int rung, float dt) {
	int blocks_needed = (leaf_count - 1) + 1;
	int block_size = SQRT(float(blocks_needed -1 )) + 1;
	assert(block_size*block_size>= leaf_count);
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

