#include <gputiger/math.hpp>
#include <nvfunctional>
#include <gputiger/chemistry.hpp>
#include <gputiger/constants.hpp>
#include <gputiger/util.hpp>
#include <gputiger/zero_order.hpp>
#include <gputiger/boltzmann.hpp>
#include <gputiger/zeldovich.hpp>
#include <gputiger/particle.hpp>
#include <gputiger/tree.hpp>
#include <gputiger/ewald.hpp>

#define RANDOM_INIT
#define BLOCK_SIZE 256

__device__ zero_order_universe *zeroverse_ptr;

__device__ float test(float x) {
	return x * x;
}
__global__
void main_kernel(void* arena, particle* host_parts, options opts_, cudaTextureObject_t* ewald_ptr) {
	const int thread = threadIdx.x;
	if (thread == 0) {
		opts = opts_;
	}
	__syncthreads();
	const int block_size = blockDim.x;
	__shared__
	float *result_ptr;
	__shared__ sigma8_integrand *func_ptr;
	__shared__ interp_functor<float>* den_k;
	__shared__ interp_functor<float>* vel_k;
	__shared__ cos_state* states;
	__shared__ cmplx* basis;
	__shared__ ewald_table_t* etable;
	const int N = opts.Ngrid;
	const int N3 = N * N * N;
	cmplx* phi = (cmplx*) arena;
	cmplx* rands = ((cmplx*) arena) + N3;
	particle* parts = (particle*) arena;
	if (thread == 0) {
		zeroverse_ptr = new zero_order_universe;
		create_zero_order_universe(zeroverse_ptr, 1.0);
	}
	__syncthreads();
	const float taumax = zeroverse_ptr->scale_factor_to_conformal_time(zeroverse_ptr->amax);
	if (thread == 0) {
		printf("Universe is %e billion years old in conformal time\n", taumax * constants::seconds_to_years * 1e-9);
	}
	__syncthreads();

	float kmin = 2.0 * (float) M_PI / opts.box_size;
	float kmax = kmin * (float) (opts.Ngrid / 2);
	kmax *= SQRT(3);

	int Nk = opts.Ngrid * SQRT(3) + 2;
	if (thread == 0) {
		printf("\nNormalizing Einstein Boltzman solutions to a present day sigma8 of %e\n", opts.sigma8);
		result_ptr = new float;
		func_ptr = new sigma8_integrand;
		CUDA_CHECK(cudaMalloc(&states, sizeof(cos_state) * Nk));
		CUDA_CHECK(cudaMalloc(&basis, sizeof(cmplx) * opts.Ngrid / 2));
		den_k = new interp_functor<float>;
		vel_k = new interp_functor<float>;
		func_ptr->uni = zeroverse_ptr;
		func_ptr->littleh = opts.h;
//		integrate<sigma8_integrand, float> <<<1, BLOCK_SIZE>>>(func_ptr,
//				(float) LOG(0.25 / 32.0 * opts.h), (float) LOG(0.25 * 32.0 * opts.h), result_ptr, (float) 1.0e-6);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		*result_ptr = SQRT(opts.sigma8 * opts.sigma8 / *result_ptr);
		printf("The normalization value is %e\n", *result_ptr);
	}

	__syncthreads();
	if (thread == 0) {
		printf("Computing start time for non-linear evolution\n");
	}
//	float normalization = *result_ptr;
	float normalization = 6.221450e-09;
	if (thread == 0) {
		printf("wave number range %e to %e Mpc^-1 for %i^3 grid and box size of %e Mpc\n", kmin, kmax, opts.Ngrid,
				opts.box_size);
	}
	if (thread == 0) {
		printf("Computing FFT basis\n");
	}
	__syncthreads();
	fft_basis(basis, opts.Ngrid);
	if (thread == 0) {
		printf("Computing randoms\n");
	}
	__syncthreads();
	generate_random_normals(rands, N * N * N);
	__syncthreads();
	if (thread == 0) {
		printf("Initializing EB\n");
	}
	einstein_boltzmann_init_set(states, zeroverse_ptr, kmin, kmax, Nk, zeroverse_ptr->amin, normalization);
	//int iter = 0;
	float logamin = LOG(zeroverse_ptr->amin);
	float logamax = LOG(zeroverse_ptr->amax);
	float drho;
//	float last_drho;
//	float last_a;
//	float dtau = taumax / opts.nout;
	float taumin = 1.0 / (zeroverse_ptr->amin * zeroverse_ptr->hubble(zeroverse_ptr->amin));
	float a = zeroverse_ptr->amin;
	/*float tau;
	 for (tau = taumin; tau < taumax - dtau; tau += dtau) {
	 last_a = a;
	 a = zeroverse_ptr->conformal_time_to_scale_factor(tau + dtau);
	 if (thread == 0) {
	 printf("Advancing Einstein Boltzmann solutions from redshift %.1f to %.1f\n", 1 / last_a - 1, 1 / a - 1);
	 }
	 einstein_boltzmann_interpolation_function(den_k, vel_k, states, zeroverse_ptr, kmin, kmax, Nk, last_a, a);
	 __syncthreads();
	 last_drho = drho;
	 drho = zeldovich_overdensity(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid);
	 __syncthreads();
	 if (thread == 0) {
	 printf("Maximum over/under density is %e\n", drho);
	 }
	 __syncthreads();
	 if (iter > 0) {
	 if (drho > opts.max_overden) {
	 drho = last_drho;
	 break;
	 }
	 }
	 __syncthreads();
	 iter++;
	 }
	 a = zeroverse_ptr->conformal_time_to_scale_factor(tau);
	 */
	a = 0.025;
#ifndef RANDOM_INIT
	if (thread == 0) {
		printf(
				"Computing initial conditions for non-linear evolution to begin at redshift %e with a maximum over/under density of %e\n",
				1 / a - 1, drho);
	}
	einstein_boltzmann_init_set(states, zeroverse_ptr, kmin, kmax, Nk, zeroverse_ptr->amin, normalization);
	__syncthreads();
	einstein_boltzmann_interpolation_function(den_k, vel_k, states, zeroverse_ptr, kmin, kmax, Nk, zeroverse_ptr->amin,
			a);
#endif
	__syncthreads();
	for (int dim = 0; dim < NDIM; dim++) {
#ifndef RANDOM_INIT
		if (thread == 0) {
			printf("Computing %c velocities\n", 'x' + dim);
		}
		float vmax = zeldovich_velocities(phi, basis, rands, *vel_k, opts.box_size, opts.Ngrid, 0);
		__syncthreads();
		for (int ij = thread; ij < N * N; ij += block_size) {
			int i = ij / N;
			int j = ij % N;
			for (int k = 0; k < N; k++) {
				const int l = N * (N * i + j) + k;
				float v = phi[l].real();
				host_parts[l].v[0] = v * a / opts.box_size;
			}
		}
#endif
		__syncthreads();
		if (thread == 0) {
			printf("Computing %c positions\n", 'x' + dim);
		}
#ifndef RANDOM_INIT
		float xdisp;
		xdisp = zeldovich_displacements(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid, 0);
#endif
		__syncthreads();
		for (int ij = thread; ij < N * N; ij += block_size) {
			int i = ij / N;
			int j = ij % N;
			for (int k = 0; k < N; k++) {
				const int l = N * (N * i + j) + k;
				const int I[NDIM] = { i, j, k };
				float x = (((float) I[dim] + 0.5f) / (float) N);
#ifdef RANDOM_INIT
				x += rands[l].real();
#else
				x += phi[l].real();
#endif
				while (x > 1.0) {
					x -= 1.0;
				}
				while (x < 0.0) {
					x += 1.0;
				}
				host_parts[l].x[dim] = x;
			}
		}
		__syncthreads();
	}
	for (int ij = thread; ij < N * N; ij += block_size) {
		int i = ij / N;
		int j = ij % N;
		for (int k = 0; k < N; k++) {
			const int l = N * (N * i + j) + k;
			host_parts[l].rung = 0;
		}
	}
	__syncthreads();
	for (int i = thread; i < N3; i += block_size) {
		parts[i] = host_parts[i];
		for (int dim = 0; dim < NDIM; dim++) {
			parts[i].v[dim] *= a;
		}
	}

	__shared__ tree* root;
	__syncthreads();
	if (thread == 0) {
		printf("Begining non-linear evolution\n");
		tree::initialize(parts, arena + N3 * sizeof(float) * 8, N3 * sizeof(float) * TREESPACE, etable);
		CUDA_CHECK(cudaMalloc(&root, sizeof(tree)));
	}
	__syncthreads();
	__shared__ range root_range;
	if (thread < 3) {
		root_range.begin[thread] = float(0.0);
		root_range.end[thread] = float(1.0);
	}
	__syncthreads();
	if (thread == 0) {
		printf("Sorting\n");
		root_tree_sort<<<1,MAXTHREADCOUNT>>>(root, host_parts, parts, parts+N3, root_range);
		CUDA_CHECK(cudaGetLastError());
	}
	__syncthreads();
	if (thread == 0) {
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	__syncthreads();
	if (thread == 0) {
		printf("Kicking\n");
		root->kick(root, 0, 0.1, ewald_ptr);
		CUDA_CHECK(cudaGetLastError());
	}
	__syncthreads();
	if (thread == 0) {
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	__syncthreads();

	if (thread == 0) {
		delete zeroverse_ptr;
		delete result_ptr;
		delete func_ptr;
		CUDA_CHECK(cudaFree(basis));
		CUDA_CHECK(cudaFree(states));
		CUDA_CHECK(cudaFree(root));
		CUDA_CHECK(cudaFree(etable));
		delete den_k;
		delete vel_k;
	}
}

cudaTextureObject_t* host_ewald;

#define KERNEL_DEPTH 13
int main() {

	options opts;

	size_t stack_size;
	size_t desired_stack_size = 4 * 1024;
	size_t rlimit = KERNEL_DEPTH+1;
	size_t heapsize = 4 * 1024 * 1024;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, rlimit));
	CUDA_CHECK(cudaDeviceGetLimit(&rlimit, cudaLimitDevRuntimeSyncDepth));
	printf("CUDA recursion limit = %li\n", rlimit);
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, desired_stack_size));
	CUDA_CHECK(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapsize));
	CUDA_CHECK(cudaDeviceGetLimit(&heapsize, cudaLimitMallocHeapSize));
	CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	printf("heapsize = %li\n", heapsize / 1024 / 1024);
//	CUDA_CHECK(cudaThreadSetCacheConfig(cudaFuncCachePreferShared));
	particle* parts_ptr;
	printf("Stack Size = %li\n", stack_size);

	struct cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	opts.clock_rate = prop.clockRate * pow(1024 / 1000, 3) * 1000;
	printf("Clock rate = %e\n", opts.clock_rate);
	opts.h = 0.697;
	opts.Neff = 3.84;
	opts.Y = 0.24;
	opts.omega_b = 0.0240 / (opts.h * opts.h);
	opts.omega_c = 0.1146 / (opts.h * opts.h);
	opts.Theta = 1.0;
	opts.Ngrid = 256;
	opts.sigma8 = 0.8367;
	opts.max_overden = 1.0;
	opts.box_size = 1000;
	//	opts.box_size = 613.0 / 2160.0 * opts.Ngrid;
	opts.nout = 64;
	opts.max_kernel_depth = KERNEL_DEPTH;
	opts.parts_per_bucket = 64;
	opts.opening_crit = 0.7;
	opts.nparts = opts.Ngrid * opts.Ngrid * opts.Ngrid;
	opts.hsoft = opts.Ngrid / 50.0;
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + opts.Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * opts.Theta, 4) * std::pow(opts.h, -2);
	opts.omega_nu = omega_r * opts.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + opts.Neff);
	opts.omega_gam = omega_r - opts.omega_nu;
	void* arena;
	const int N = opts.Ngrid;
	const int N3 = N * N * N;

	ewald_table_t* etable;
	printf("Computing ewald tables\n");
	CUDA_CHECK(cudaMallocManaged(&etable, sizeof(ewald_table_t)));
	CUDA_CHECK(cudaMallocManaged(&host_ewald, sizeof(cudaTextureObject_t)));
	compute_ewald_table<<<EWALD_DIM*EWALD_DIM, EWALD_DIM>>>(etable);
	CUDA_CHECK(cudaDeviceSynchronize());

	for (int i = 0; i < NDIM + 1; i++) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaArray *d_cuArr;
		CUDA_CHECK(
				cudaMalloc3DArray(&d_cuArr, &channelDesc,
						make_cudaExtent(opts.Ngrid * sizeof(float), opts.Ngrid, opts.Ngrid), 0));
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr((*etable)[i].data(), opts.Ngrid * sizeof(float), opts.Ngrid,
				opts.Ngrid);
		copyParams.dstArray = d_cuArr;
		copyParams.extent = make_cudaExtent(opts.Ngrid, opts.Ngrid, opts.Ngrid);
		copyParams.kind = cudaMemcpyDeviceToDevice;
		CUDA_CHECK(cudaMemcpy3D(&copyParams));
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = d_cuArr;
		cudaTextureDesc texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = false;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		CUDA_CHECK(cudaCreateTextureObject(&host_ewald[i], &texRes, &texDescr, NULL));
	}

	CUDA_CHECK(cudaMallocManaged(&parts_ptr, sizeof(particle) * N3));
	size_t arena_size = (8 + TREESPACE) * sizeof(float) * N3;
	printf("Allocating arena of %li Mbytes\n", (arena_size / 1024 / 1024));
	CUDA_CHECK(cudaMallocManaged(&arena, arena_size));
	if (arena == nullptr) {
		printf("Not enough memory\n");
		abort();
	}
	main_kernel<<<1, BLOCK_SIZE>>>(arena, parts_ptr, opts, host_ewald);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaFree(arena));
	CUDA_CHECK(cudaFree(parts_ptr));
	CUDA_CHECK(cudaFree(etable));
	CUDA_CHECK(cudaFree(host_ewald));

}
