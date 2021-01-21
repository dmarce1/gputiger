#include <gputiger/tree.hpp>
#include <gputiger/params.hpp>

#define BIGBLOCK 256
#define BIGDEPTH 6
#define SMALLBLOCK 128

__global__
void tree_sort_big_begin(particle* part_begin, particle* part_end, particle* swap, int* lo, int* hi, float xmid,
		int xdim);

__global__
void tree_sort_big_end(tree* tptr, particle* swap);

__device__
void tree_sort_big(tree* tptr, particle* swap, int xdim);

__global__
void tree_sort_small(int lev, particle* globalswap, int xdim);

__device__
                 extern tree* tree_base;

__device__
extern int* leaf_list;

__device__
int* tree_list;

__device__
int next_tree_list_index;

__device__ particle* particle_base;

__device__ array<int, MAXDEPTH + 2> levels;

__device__
void tree_sort(tree* root, particle* part_begin, particle* part_end, particle* swap) {
	int tid = threadIdx.x;
	int blocksize = blockIdx.x;
	if (tid == 0) {
		particle_base = part_begin;
		tree_list = leaf_list;
		root->part_begin = part_begin;
		root->part_end = part_end;
		root->leaf = true;
		for (int dim = 0; dim < NDIM; dim++) {
			root->box.begin[dim] = 0.f;
			root->box.end[dim] = 0.f;
		}

		next_tree_list_index = 0;
		levels[0] = 0;
		levels[1] = 1;
	}

	int xdim = 0;
	int lev;
	bool done = false;
	for (lev = 0; lev < BIGDEPTH; lev++) {
		for (int t = levels[lev]; t < levels[lev + 1] && !done; t += blocksize) {
			tree_sort_big(tree_base + tree_list[t], swap, xdim);
		}
		__syncthreads();
		if (tid == 0) {
			levels[lev + 2] = next_tree_list_index;
		}
		__syncthreads();
		done = levels[lev + 2] == levels[lev + 1];
		xdim = (xdim + 1) % NDIM;
	}
	for (lev = BIGDEPTH; lev < MAXDEPTH && !done; lev++) {
		int nblocks = levels[lev + 1] - levels[lev];
		__syncthreads();
		if (tid == 0) {
			tree_sort_small<<<nblocks,SMALLBLOCK>>>(lev,swap,xdim);
			CUDA_CHECK(cudaDeviceSynchronize());
			levels[lev + 2] = next_tree_list_index;
		}
		__syncthreads();
		done = levels[lev + 2] == levels[lev + 1];
		xdim = (xdim + 1) % NDIM;
	}

}

__global__
void tree_sort_small(int lev, particle* globalswap, int xdim) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int bsz = blockDim.x;
	__shared__ array<int, SMALLBLOCK> lo_index;
	__shared__ array<int, SMALLBLOCK> hi_index;
	__shared__ int lo;
	__shared__ int hi;
	tree* tptr = tree_base + tree_list[levels[lev] + bid];
	particle* swap = globalswap + (tptr->part_begin - particle_base);
	const int sz = tptr->part_end - tptr->part_begin;
	if (sz > opts.parts_per_bucket) {
		if (tid == 0) {
			lo = hi = 0;
		}
		__syncthreads();
		const float xmid = (tptr->box.begin[xdim] + tptr->box.end[xdim]) / 2.f;
		for (int i = 0; i < ((sz - 1) / bsz + 1) * bsz; i += bsz) {
			auto& part = tptr->part_begin[i];
			lo_index[tid] = hi_index[tid] = 0;
			int hilo = 0;
			if (i < bsz) {
				if (part.x[xdim] > xmid) {
					hi_index[tid] = 1;
					hilo = +1;
				} else {
					lo_index[tid] = 1;
					hilo = -1;
				}
			}
			__syncthreads();
			for (int P = 1; P < bsz; P *= 2) {
				if (tid - P >= 0) {
					hi_index[tid] += hi_index[tid - P];
					lo_index[tid] += lo_index[tid - P];
				}
				__syncthreads();
			}
			if (hilo > 0) {
				swap[sz - 1 - hi - hi_index[tid]] = part;
			} else if (hilo < 0) {
				swap[lo + lo_index[tid]] = part;
			}
			__syncthreads();
			if (tid == 0) {
				hi += hi_index[bsz - 1];
				lo += lo_index[bsz - 1];
			}
		}
		for (int i = 0; i < sz; i += bsz) {
			tptr->part_begin[i] = swap[i];
		}
		__syncthreads();
		if (tid == 0) {
			for (int ci = 0; ci < NCHILD; ci++) {
				tptr->children[ci] = tree::alloc();
				auto* cptr = tptr->children[ci];
				cptr->part_begin = tptr->part_begin;
				cptr->part_end = tptr->part_end;
				cptr->box = tptr->box;
				cptr->leaf = true;
				tree_list[atomicAdd(&next_tree_list_index, 1)] = cptr - tree_base;
			}
			tptr->leaf = false;
			tptr->children[0]->part_end = tptr->children[1]->part_begin = tptr->part_begin + lo;
			tptr->children[0]->box.end[xdim] = tptr->children[1]->box.begin[xdim] = xmid;
		}
	}
}

__device__ void tree_sort_big(tree* tptr, particle* swap, int xdim) {
	if (tptr->part_end - tptr->part_begin > opts.parts_per_bucket) {
		int nparts = tptr->part_end - tptr->part_begin;
		int nblocks = (nparts - 1) / BIGBLOCK + 1;
		dim3 dims;
		dims.x = nblocks;
		dims.y = 1;
		dims.z = 1;
		int* lo;
		int* hi;
		int mid;
		float xmid = (tptr->box.begin[xdim] + tptr->box.end[xdim]) / 2.f;
		CUDA_MALLOC(&lo, sizeof(int));
		CUDA_MALLOC(&hi, sizeof(int));
		*lo = *hi = 0;
		tree_sort_big_begin<<<dims,BIGBLOCK>>>(tptr->part_begin,tptr->part_end,swap,lo, hi, xmid,xdim);
		tree_sort_big_end<<<dims,BIGBLOCK>>>(tptr,swap);
		CUDA_CHECK(cudaDeviceSynchronize());
		mid = *lo;
		CUDA_CHECK(cudaFree(lo));
		CUDA_CHECK(cudaFree(hi));
		for (int ci = 0; ci < NCHILD; ci++) {
			tptr->children[ci] = tree::alloc();
			auto* cptr = tptr->children[ci];
			cptr->part_begin = tptr->part_begin;
			cptr->part_end = tptr->part_end;
			cptr->box = tptr->box;
			cptr->leaf = true;
			tree_list[atomicAdd(&next_tree_list_index, 1)] = cptr - tree_base;
		}
		tptr->leaf = false;
		tptr->children[0]->part_end = tptr->children[1]->part_begin = tptr->part_begin + mid;
		tptr->children[0]->box.end[xdim] = tptr->children[1]->box.begin[xdim] = xmid;
	}
}

__global__
void tree_sort_big_end(tree* tptr, particle* swap) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int bsz = blockDim.x;
	const int psz = tptr->part_end - tptr->part_begin;
	particle* const pb = tptr->part_begin + bid * psz / bsz;
	particle* const pe = tptr->part_begin + (bid + 1) * psz / bsz;
	particle* const p = pb + tid;
	if (p < tptr->part_end && p < pe) {
		*p = swap[p - tptr->part_begin];
	}
}

__global__
void tree_sort_big_begin(particle* part_begin, particle* part_end, particle* swap, int* lo, int* hi, float xmid,
		int xdim) {
	__shared__ int lo_index[BIGBLOCK];
	__shared__ int hi_index[BIGBLOCK];
	__shared__ int lo_base;
	__shared__ int hi_base;

	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int bsz = blockDim.x;
	const int psz = part_end - part_begin;
	particle* const pb = part_begin + (bid * psz / bsz);
	particle* const pe = part_begin + ((bid + 1) * psz / bsz);
	particle* const p = pb + tid;
	int hilo = 0;
	lo_index[tid] = hi_index[tid] = 0;
	if (p < part_end && p < pe) {
		if (p->x[xdim] > xmid) {
			hilo = +1;
			hi_index[tid] = 1;
		} else {
			hilo = -1;
			lo_index[tid] = 1;
		}
	}
	__syncthreads();
	for (int P = 1; P < bsz; P *= 2) {
		if (tid <= P) {
			hi_index[P] += hi_index[P - tid];
			lo_index[P] += lo_index[P - tid];
		}
		__syncthreads();
	}
	if (tid == 0) {
		lo_base = atomicAdd(lo, lo_index[bsz - 1]);
		hi_base = atomicAdd(hi, hi_index[bsz - 1]);
	}
	__syncthreads();
	if (hilo > 0) {
		swap[psz - 1 - hi_index[tid] - hi_base] = *p;
	} else {
		swap[lo_index[tid] + lo_base] = *p;
	}
}

