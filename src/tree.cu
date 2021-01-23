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

__device__ cudaTextureObject_t* tex_ewald;

__device__
void tree::initialize(particle* parts, void* data, size_t bytes) {
	int sztot = sizeof(tree) + sizeof(int);
	int N = bytes / sztot;
	next_index = 0;
	arena_size = N;
	tree_base = (tree*) data;
	leaf_list = (int*) (data + sizeof(tree) * N);
	leaf_count = 0;
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

__device__ tree root;

__global__
void root_tree_sort(void* dataspace, int space_size, particle* swap_space, particle* parts, int* cnt) {
	if (threadIdx.x == 0) {
		tree::initialize(parts, dataspace, space_size);
		root.part_begin = parts;
		root.part_end = parts + opts.Ngrid * opts.Ngrid * opts.Ngrid;
		for (int dim = 0; dim < NDIM; dim++) {
			root.box.begin[dim] = 0.f;
			root.box.end[dim] = 1.f;
		}
	}
	__syncthreads();
	root.sort(swap_space, 0);
	*cnt = leaf_count;
}

__global__
void tree_sort(tree** children, particle* swap_space, int depth) {
	const int& bid = blockIdx.x;
	particle* base = children[0]->part_begin;
	particle* b = children[bid]->part_begin;
	particle* e = children[bid]->part_end;
	particle* this_swap = swap_space + (b - base);
	range box = children[bid]->box;
	const auto tmp = children[bid]->sort(this_swap, depth);
	if (threadIdx.x == 0) {
		children[bid]->pole = tmp;
	}
	__syncthreads();
}

__device__ monopole tree::sort(particle* swap_space, int depth_) {
	const int tid = threadIdx.x;
	//const int block_size = blockDim.x;
	__shared__ array<array<fixed64, NDIM>, WARPSIZE> poles;
	__shared__ array<float, WARPSIZE> radii;
	if (tid == 0) {
		if (depth_ >= MAXDEPTH) {
			printf("Maximum depth exceeded in sort\n");
			__trap();
		}
		depth = depth_;
		leaf = false;
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

	if (part_end - part_begin > opts.parts_per_bucket) {
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
		mid = sort_parts(swap_space, part_begin, part_end, midx, long_dim);
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
				monopole this_pole = children[ci]->sort(swap_base, depth + 1);
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
				tree_sort<<<NCHILD,threadcnt>>>(children.data(), swap_space, depth+1);

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
				auto p0 = (children[0]->pole.count) * (children[0]->pole.xcom[dim].to_double());
				auto p1 = (children[1]->pole.count) * (children[1]->pole.xcom[dim].to_double());
				p0 += p1;
				p0 /= pole.count;
				pole.xcom[dim] = p0;
			}
			double r0 = 0.0;
			for (int dim = 0; dim < NDIM; dim++) {
				r0 += pow2(children[0]->pole.xcom[dim].to_double() - pole.xcom[dim].to_double());
			}
			r0 = max(children[0]->pole.radius, children[1]->pole.radius) + sqrt(r0);
			double r1 = 0.0;
			double r2 = 0.0;
			array<double, NDIM> corner;
			corner[0] = box.begin[0];
			corner[1] = box.begin[1];
			corner[2] = box.begin[2];
			for (int dim = 0; dim < NDIM; dim++) {
				r1 += pow2(pole.xcom[dim].to_double() - corner[dim]);
			}
			r2 = max(r1, r2);
			r1 = 0.0;
			corner[0] = box.end[0];
			corner[1] = box.begin[1];
			corner[2] = box.begin[2];
			for (int dim = 0; dim < NDIM; dim++) {
				r1 += pow2(pole.xcom[dim].to_double() - corner[dim]);
			}
			r2 = max(r1, r2);
			r1 = 0.0;
			corner[0] = box.begin[0];
			corner[1] = box.end[1];
			corner[2] = box.begin[2];
			for (int dim = 0; dim < NDIM; dim++) {
				r1 += pow2(pole.xcom[dim].to_double() - corner[dim]);
			}
			r2 = max(r1, r2);
			r1 = 0.0;
			corner[0] = box.end[0];
			corner[1] = box.end[1];
			corner[2] = box.begin[2];
			for (int dim = 0; dim < NDIM; dim++) {
				r1 += pow2(pole.xcom[dim].to_double() - corner[dim]);
			}
			r2 = r1;
			r1 = 0.0;
			corner[0] = box.begin[0];
			corner[1] = box.begin[1];
			corner[2] = box.end[2];
			for (int dim = 0; dim < NDIM; dim++) {
				r1 += pow2(pole.xcom[dim].to_double() - corner[dim]);
			}
			r2 = max(r1, r2);
			r1 = 0.0;
			corner[0] = box.end[0];
			corner[1] = box.begin[1];
			corner[2] = box.end[2];
			for (int dim = 0; dim < NDIM; dim++) {
				r1 += pow2(pole.xcom[dim].to_double() - corner[dim]);
			}
			r2 = max(r1, r2);
			r1 = 0.0;
			corner[0] = box.begin[0];
			corner[1] = box.end[1];
			corner[2] = box.end[2];
			for (int dim = 0; dim < NDIM; dim++) {
				r1 += pow2(pole.xcom[dim].to_double() - corner[dim]);
			}
			r2 = max(r1, r2);
			r1 = 0.0;
			corner[0] = box.end[0];
			corner[1] = box.end[1];
			corner[2] = box.end[2];
			for (int dim = 0; dim < NDIM; dim++) {
				r1 += pow2(pole.xcom[dim].to_double() - corner[dim]);
			}
			r2 = max(r1, r2);
			r2 = sqrt(r2);
			r0 = min(r2, r0);
			pole.radius = r0;
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
		if (tid < WARPSIZE) {
			radii[tid] = 0.f;
			for (particle* p = part_begin + tid; p < part_end; p += WARPSIZE) {
				float r = 0.f;
				for (int dim = 0; dim < NDIM; dim++) {
					double d = p->x[dim].to_double() - pole.xcom[dim].to_double();
					r += pow2(d);
				}
				radii[tid] = fmaxf(radii[tid], r);
			}
		}
		__syncthreads();
		for (int P = WARPSIZE / 2; P >= 1; P /= 2) {
			if (tid < P) {
				radii[tid] = fmaxf(radii[tid], radii[tid + P]);
			}
			__syncthreads();
		}
		pole.radius = sqrtf(radii[0]);
//		printf( "%e\n", pole.radius);
		__syncthreads();
	}
//	if (tid == 0 && !leaf)
//		printf("%e\n", pole.xcom[0].to_float());
	return pole;
}

struct direct_t {
	const tree* self;
	const tree* other;
};

#define PARTMAX 64
#define OTHERSMAX 256

__global__
void tree_kick(int rung, float dt, float scale, int* nactive, int* maxrung) {
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
		const float sqrtah = sqrtf(opts.h * scale);
		const float log2inv = 1.f / logf(2.f);
		const float GM = opts.particle_mass;
		int depth;
		const tree& self = *(tree_base + leaf_list[myindex]);
		tree* pointers[MAXDEPTH];
		int child_indexes[MAXDEPTH];
		float self_r = self.pole.radius;
		bool done;
		array<fixed32, NDIM> self_x = self.pole.xcom;

		done = false;
		for (int i = 0; i < MAXDEPTH; i++) {
			child_indexes[i] = 0;
		}
		pointers[0] = &root;
		depth = 0;
		do {
			const auto& other = *pointers[depth];
			array<fixed32, NDIM> other_x = other.pole.xcom;
			float other_r = other.pole.radius;
			float dist2 = 0.f;
			float dist;
			for (int dim = 0; dim < NDIM; dim++) {
				dist = self_x[dim].ewald_dif(other_x[dim]);
				dist2 += dist * dist;
			}
			dist = SQRT(dist2);
			opened = (self_r + other_r) > opts.opening_crit * dist;
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
				for (auto* sink = self.part_begin; sink < self.part_end; sink++) {
					if (sink->rung >= rung) {
						const auto& sink_x = sink->x;
						const int count = min(other_cnt, OTHERSMAX);
						for (int oi = tid; oi < count; oi += KICKWARPSIZE) {
							array<float, NDIM> X;
							{
								const auto& source_x = others[oi];
								for (int dim = 0; dim < NDIM; dim++) {
									X[dim] = source_x[dim].ewald_dif(sink_x[dim]); // 9
								}
							}
							{
								float Xinv3 = rsqrtf(fmaxf(X[0] * X[0] + X[1] * X[1] + X[2] * X[2], h2)); // 7 + 3
								Xinv3 = Xinv3 * Xinv3 * Xinv3; //2
								const int index = sink - self.part_begin;
								for (int dim = 0; dim < NDIM; dim++) {
									F[index][dim] += X[dim] * Xinv3; // 6
								}
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
		other_cnt = 0;
		done = false;
		for (int i = 0; i < MAXDEPTH; i++) {
			child_indexes[i] = 0;
		}
		pointers[0] = &root;
		depth = 0;
		do {
			const auto& other = *pointers[depth];
			array<fixed32, NDIM> other_x = other.pole.xcom;
			float other_r = other.pole.radius;
			float dist2 = 0.f;
			float dist;
			for (int dim = 0; dim < NDIM; dim++) {
				dist = max(0.25, self_x[dim].ewald_dif(other_x[dim]));
				dist2 += dist * dist;
			}
			dist = SQRT(dist2);
			opened = (self_r + other_r) > opts.opening_crit * dist;
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
				for (auto* sink = self.part_begin; sink < self.part_end; sink++) {
					if (sink->rung >= rung) {
						const auto& sink_x = sink->x;
						const int count = min(other_cnt, OTHERSMAX);
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
									F[index][dim] += copysignf(tmp, X[dim]);
								}

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
		for (auto* p = self.part_begin + tid; p < self.part_end; p += KICKWARPSIZE) {
			if (p->rung >= rung) {
				const auto &f = F[p - self.part_begin];
				float fmag2 = 0.f;
				atomicAdd(nactive, 1);
				for (int dim = 0; dim < NDIM; dim++) {
					const float this_f = f[dim];
					p->v[dim] += GM * this_f * dt;
					fmag2 += this_f * this_f;
				}
				const float dt = sqrtah * powf(fmag2, -0.25);
				p->rung = max(-int(logf(dt) * log2inv + 1.0), min(p->rung - 1, rung));
				atomicMax(maxrung,p->rung);
			}
		}
	}
}

