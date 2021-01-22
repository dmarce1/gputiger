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
static int* tree_list;

__device__
static int leaf_count;

__device__
     static ewald_table_t* etable;

__device__
static cudaTextureObject_t* tex_ewald;

__device__
void tree::initialize(particle* parts, void* data, size_t bytes, ewald_table_t* etable_) {
	etable = etable_;
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

__device__ particle* sort_parts(particle* swap, particle* b, particle* e, float xmid, int dim) {
	const int& tid = threadIdx.x;
	const int& block_size = blockDim.x;
	__shared__ int lo;
	__shared__ int hi;
	if (e == b) {
		return e;
	} else {
		if (tid == 0) {
			lo = 0;
			hi = (e - b) - 1;
		}
		__syncthreads();
		int index;
		for (particle* part = b + tid; part < e; part += block_size) {
			if (part->x[dim] > xmid) {
				index = atomicSub(&hi, 1);
			} else {
				index = atomicAdd(&lo, 1);
			}
			swap[index] = *part;
		}
		__syncthreads();
		for (index = tid; index < (e - b); index += block_size) {
			assert(index < (e - b));
			b[index] = swap[index];
		}
		__syncthreads();
		return b + lo;
	}
}

__global__
void root_tree_sort(tree* root, particle* swap_space, particle* pbegin, particle* pend, const range box, int rung) {
	root->sort(swap_space, pbegin, pend, box, 0, rung);
}

__global__
void tree_sort(tree** children, particle* swap_space, int depth, int rung) {
	const int& bid = blockIdx.x;
	particle* base = children[0]->part_begin;
	particle* b = children[bid]->part_begin;
	particle* e = children[bid]->part_end;
	particle* this_swap = swap_space + (b - base);
	range box = children[bid]->box;
	const auto tmp = children[bid]->sort(this_swap, b, e, box, depth, rung);
	if (threadIdx.x == 0) {
		children[bid]->pole = tmp;
	}
	__syncthreads();
}

__device__ monopole tree::sort(particle* swap_space, particle* pbegin, particle* pend, range box_, int depth_,
		int rung) {
	const int tid = threadIdx.x;
	//const int block_size = blockDim.x;
	__shared__ array<array<fixed64, NDIM>, WARPSIZE> poles;
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
	/*	for (auto* ptr = part_begin + tid; ptr < part_end; ptr += blockDim.x) {
	 if (!box.in_range(ptr->x.to_float())) {
	 printf("Particle out of range at depth %i\n", depth);
	 for (int dim = 0; dim < NDIM; dim++) {
	 printf("%e %e %e\n", box.begin[dim], box.end[dim], ptr->x[dim]);
	 }
	 printf("\n");
	 //				__trap();
	 }
	 }*/
	//	printf("Sorting at depth %i\n", depth);
	__syncthreads();

	if (pend - pbegin > opts.parts_per_bucket) {
		if (tid == 0) {
			for (int ci = 0; ci < NCHILD; ci++) {
				children[ci] = alloc();
			}
		}
		__syncthreads();
		float max_span = 0.0;
		int long_dim;
		for (int dim = 0; dim < NDIM; dim++) {
			if (box.end[dim] - box.begin[dim] > max_span) {
				max_span = box.end[dim] - box.begin[dim];
				long_dim = dim;
			}
		}
		midx = (box.begin[long_dim] / float(2) + box.end[long_dim] / float(2));
		mid = sort_parts(swap_space, pbegin, pend, midx, long_dim);
		if (tid == 0) {
			children[0]->box = box;
			children[1]->box = box;
			children[0]->box.end[long_dim] = midx;
			children[1]->box.begin[long_dim] = midx;
			children[0]->part_begin = part_begin;
			children[1]->part_end = part_end;
			children[0]->part_end = children[1]->part_begin = mid;
		}
		__syncthreads();
		if (depth >= opts.max_kernel_depth) {
			for (int ci = 0; ci < NCHILD; ci++) {
				particle* swap_base = swap_space + (children[ci]->part_begin - children[0]->part_begin);
				monopole this_pole = children[ci]->sort(swap_base, children[ci]->part_begin, children[ci]->part_end,
						children[ci]->box, depth + 1, rung);
			}
			__syncthreads();
		} else {
			__syncthreads();
			if (tid == 0) {
				int threadcnt;
				if (depth == opts.max_kernel_depth - 1) {
					threadcnt = WARPSIZE;
				} else {
					threadcnt = max(min((int) ((part_end - part_begin) / NCHILD), (int) MAXTHREADCOUNT), WARPSIZE);
				}
				tree_sort<<<NCHILD,threadcnt>>>(children.data(), swap_space, depth+1, rung);

				if (tid == 0) {
					CUDA_CHECK(cudaGetLastError());
					CUDA_CHECK(cudaDeviceSynchronize());
				}
			}
		}
		__syncthreads();
		if (tid == 0) {
			pole.count = children[0]->pole.count + children[1]->pole.count;
			for (int dim = 0; dim < NDIM; dim++) {
				auto p0 = (children[0]->pole.count) * (children[0]->pole.xcom[dim].to_float());
				auto p1 = (children[1]->pole.count) * (children[1]->pole.xcom[dim].to_float());
				p0 += p1;
				p0 /= pole.count;
				pole.xcom[dim] = p0;
//				printf( "%e\n", p0.to_float());
			}
		}
		__syncthreads();
	} else {
		__syncthreads();
		if (tid == 0) {
			leaf = true;
			int index = atomicAdd(&leaf_count, 1);
			leaf_list[index] = this - tree_base;
		}
		__syncthreads();
		if (tid < WARPSIZE) {
			for (int dim = 0; dim < NDIM; dim++) {
				poles[tid][dim] = 0.f;
			}
			for (particle* p = part_begin + tid; p < part_end; p += WARPSIZE) {
				for (int dim = 0; dim < NDIM; dim++) {
					poles[tid][dim] += fixed64(p->x[dim]);
				}
			}
		}
		__syncthreads();
		for (int P = WARPSIZE / 2; P >= 1; P /= 2) {
			if (tid < P) {
				for (int dim = 0; dim < NDIM; dim++) {
					poles[tid][dim] += poles[tid + P][dim];
				}
			}
			__syncthreads();
		}
		if (tid == 0) {
			pole.count = part_end - part_begin;
			if (pole.count) {
				for (int dim = 0; dim < NDIM; dim++) {
					poles[0][dim] /= fixed64(pole.count);
				}
				for (int dim = 0; dim < NDIM; dim++) {
					pole.xcom[dim] = fixed32(poles[0][dim]);
				}
			} else {
				for (int dim = 0; dim < NDIM; dim++) {
					pole.xcom[dim] = 0.f;
				}
			}
		}
		__syncthreads();
	}
//	if (tid == 0 && !leaf)
//		printf("%e\n", pole.xcom[0].to_float());
	return pole;
}

#define KICKWARPSIZE 32

#define KICKEWALDWARPSIZE 32

struct direct_t {
	const tree* self;
	const tree* other;
};

#define PARTMAX 64
#define OTHERSMAX 256

__global__
void tree_kick(tree* root, int rung, float dt, double* flops) {
	const int& tid = threadIdx.x;
	const int& yi = blockIdx.x;
	const int& xi = blockIdx.y;
	const int myindex = (gridDim.x * xi + yi);
	__shared__ float h2;
	__shared__ float dewald;
	bool opened;
	__shared__ array<array<float, NDIM>, PARTMAX> F;
	__shared__ array<array<fixed32, NDIM>, OTHERSMAX> others1;
	__shared__ array<array<fixed32, NDIM>, OTHERSMAX> others2;
	__shared__ bool swtch;
	__shared__ array<fixed32, NDIM>* others;
	__shared__ array<fixed32, NDIM>* next_others;
	__shared__ int other_cnt;
	if (tid == 0) {
		swtch = false;
		others = others1.data();
		next_others = others2.data();
		other_cnt = 0;
		h2 = opts.hsoft * opts.hsoft;
		dewald = EWALD_DIM - 1.0f;
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
	if (myindex < leaf_count) {
		int depth;
		const tree& self = *(tree_base + leaf_list[myindex]);
		tree* pointers[MAXDEPTH];
		int child_indexes[MAXDEPTH];
		float self_w = 0.f;
		bool done;
		for (int dim = 0; dim < NDIM; dim++) {
			self_w += POW(self.box.end[dim] - self.box.begin[dim], 2);
		}
		self_w = SQRT(self_w) / float(2);
		array<fixed32, NDIM> self_x = self.pole.xcom;

		done = false;
		for (int i = 0; i < MAXDEPTH; i++) {
			child_indexes[i] = 0;
		}
		pointers[0] = root;
		depth = 0;
		do {
			const auto& other = *pointers[depth];
			array<fixed32, NDIM> other_x = other.pole.xcom;
			float other_w = 0.0f;
			for (int dim = 0; dim < NDIM; dim++) {
				other_w += pow2(other.box.end[dim] - other.box.begin[dim]);
			}
			other_w = SQRT(other_w) / 2.f;
			float dist2 = 0.f;
			float dist;
			for (int dim = 0; dim < NDIM; dim++) {
				dist = self_x[dim].ewald_dif(other_x[dim]);
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
					if (index < OTHERSMAX) {
						others[index] = source_x;
					} else {
						next_others[index - OTHERSMAX] = source_x;
					}
				}
			} else {
				if (tid == 0) {
					others[other_cnt++] = other_x;
				}
				nindirect++;
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
			if (other_cnt >= OTHERSMAX || done) {
				__syncthreads();
				const tree& self = *(tree_base + leaf_list[myindex]);
				if (tid == 0) {
					atomicAdd(flops, double(42 * (self.part_end - self.part_begin) * min(other_cnt, OTHERSMAX)));
				}
				for (auto* sink = self.part_begin; sink < self.part_end; sink++) {
					const auto& sink_x = sink->x;
					const int count = min(other_cnt, OTHERSMAX);
					for (int oi = tid; oi < count; oi += KICKWARPSIZE) {
						array<float, NDIM> X;
						{
							const auto& source_x = others[oi];
							/*						for (int dim = 0; dim < NDIM; dim++) {
							 const float x = source_x[dim].to_float() - sink_x[dim].to_float();
							 const float absx = fabs(x);  // 1
							 X[dim] = copysignf(fminf(absx, 1.f - absx), x * (0.5f - absx));  // 5
							 }*/
							for (int dim = 0; dim < NDIM; dim++) {
								X[dim] = source_x[dim].ewald_dif(sink_x[dim]);
							}
						}
						{
							float Xinv3 = rsqrtf(fmaxf(X[0] * X[0] + X[1] * X[1] + X[2] * X[2], h2));
							Xinv3 = Xinv3 * Xinv3 * Xinv3;
							const int index = sink - self.part_begin;
							for (int dim = 0; dim < NDIM; dim++) {
								F[index][dim] += X[dim] * Xinv3; // 2
							}
						}
					}
				}
				__syncthreads();
				if (tid == 0) {
					other_cnt -= OTHERSMAX;
					swtch = !swtch;
					if (swtch) {
						others = others2.data();
						next_others = others1.data();
					} else {
						others = others1.data();
						next_others = others2.data();
					}
				}
				__syncthreads();
			}
		} while (!done);
	}
}


#define OTHERSMAX2 128

__global__
void tree_kick_ewald(tree* root, int rung, float dt, double* flops) {
	const int& tid = threadIdx.x;
	const int& yi = blockIdx.x;
	const int& xi = blockIdx.y;
	const int myindex = (gridDim.x * xi + yi);
	__shared__ float h2;
	__shared__ float dewald;
	bool opened;
	__shared__ array<array<float, NDIM>, PARTMAX> F;
	__shared__ array<array<fixed32, NDIM>, OTHERSMAX2> others1;
	__shared__ array<array<fixed32, NDIM>, OTHERSMAX2> others2;
	__shared__ bool swtch;
	__shared__ array<fixed32, NDIM>* others;
	__shared__ array<fixed32, NDIM>* next_others;
	__shared__ int other_cnt;
	if (tid == 0) {
		swtch = false;
		others = others1.data();
		next_others = others2.data();
		other_cnt = 0;
		h2 = opts.hsoft * opts.hsoft;
		dewald = EWALD_DIM - 1.0f;
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
	if (myindex < leaf_count) {
		int depth;
		const tree& self = *(tree_base + leaf_list[myindex]);
		tree* pointers[MAXDEPTH];
		int child_indexes[MAXDEPTH];
		float self_w = 0.f;
		bool done;
		for (int dim = 0; dim < NDIM; dim++) {
			self_w += POW(self.box.end[dim] - self.box.begin[dim], 2);
		}
		self_w = SQRT(self_w) / float(2);
		array<fixed32, NDIM> self_x = self.pole.xcom;

		done = false;
		for (int i = 0; i < MAXDEPTH; i++) {
			child_indexes[i] = 0;
		}
		pointers[0] = root;
		depth = 0;
		do {
			const auto& other = *pointers[depth];
			array<fixed32, NDIM> other_x = other.pole.xcom;
			float other_w = 0.0f;
			for (int dim = 0; dim < NDIM; dim++) {
				other_w += pow2(other.box.end[dim] - other.box.begin[dim]);
			}
			other_w = SQRT(other_w) / 2.f;
			float dist2 = 0.f;
			float dist;
			for (int dim = 0; dim < NDIM; dim++) {
				dist = max(0.25, self_x[dim].ewald_dif(other_x[dim]));
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
					if (index < OTHERSMAX2) {
						others[index] = source_x;
					} else {
						next_others[index - OTHERSMAX2] = source_x;
					}
				}
			} else {
				if (tid == 0) {
					others[other_cnt++] = other_x;
				}
				nindirect++;
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
			if (other_cnt >= OTHERSMAX2 || done) {
				__syncthreads();
				const tree& self = *(tree_base + leaf_list[myindex]);
				if (tid == 0) {
					atomicAdd(flops, double(128 * (self.part_end - self.part_begin) * min(other_cnt, OTHERSMAX2)));
				}
				for (auto* sink = self.part_begin; sink < self.part_end; sink++) {
					const auto& sink_x = sink->x;
					const int count = min(other_cnt, OTHERSMAX2);
					for (int oi = tid; oi < count; oi += KICKWARPSIZE) {
						array<float, NDIM> X;
						array<float, NDIM> absX;
						const auto& source_x = others[oi];
						for (int dim = 0; dim < NDIM; dim++) {
							X[dim] = source_x[dim].ewald_dif(sink_x[dim]);
							absX[dim] = abs(X[dim]);
							const int index = sink - self.part_begin;
							const float x0 = absX[0] * dewald + 0.5;
							const float y0 = absX[1] * dewald + 0.5;
							const float z0 = absX[2] * dewald + 0.5;
							for (int dim = 0; dim < NDIM; dim++) {
								float tmp = tex3D<float>(tex_ewald[dim], x0, y0, z0);
								F[index][dim] += tmp;
							}

						}
					}
				}
				__syncthreads();
				if (tid == 0) {
					other_cnt -= OTHERSMAX2;
					swtch = !swtch;
					if (swtch) {
						others = others2.data();
						next_others = others1.data();
					} else {
						others = others1.data();
						next_others = others2.data();
					}
				}
				__syncthreads();
			}
		} while (!done);
	}
}

__device__
void tree::kick(tree* root, int rung, float dt, cudaTextureObject_t* tex_ewald_ ) {
	tex_ewald = tex_ewald_;
	int blocks_needed = (leaf_count - 1) + 1;
	int block_size = SQRT(float(blocks_needed -1 )) + 1;
	assert(block_size * block_size >= leaf_count);
	dim3 dim;
	dim.x = dim.y = block_size;
	dim.z = 1;
	double* flops;
	CUDA_CHECK(cudaMalloc(&flops, sizeof(double)));
	*flops = 0.0;
	tree_kick<<<dim,KICKWARPSIZE>>>(root, rung,dt, flops);
	tree_kick_ewald<<<dim,KICKEWALDWARPSIZE>>>(root, rung,dt, flops);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	printf("FLOPS = %e\n", *flops);
	CUDA_CHECK(cudaFree(flops));

}

