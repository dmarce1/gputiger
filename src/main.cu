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

#define BLOCK_SIZE 256

__device__ zero_order_universe *zeroverse_ptr;

__device__ float test(float x) {
	return x * x;
}
__global__
void main_kernel(void* arena, particle* host_parts, options opts_) {
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
	int Nk = opts.Ngrid * SQRT(3) + 1;
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
		integrate<sigma8_integrand, float> <<<1, BLOCK_SIZE>>>(func_ptr,
				(float) LOG(0.25 / 32.0 * opts.h), (float) LOG(0.25 * 32.0 * opts.h), result_ptr, (float) 1.0e-6);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		*result_ptr = SQRT(opts.sigma8 * opts.sigma8 / *result_ptr);
		printf("The normalization value is %e\n", *result_ptr);
	}
	__syncthreads();
	if (thread == 0) {
		printf("Computing start time for non-linear evolution\n");
	}
	float normalization = *result_ptr;
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
	int iter = 0;
	float logamin = LOG(zeroverse_ptr->amin);
	float logamax = LOG(zeroverse_ptr->amax);
	float drho;
	float last_drho;
	float last_a;
	float dtau = taumax / opts.nout;
	float taumin = 1.0 / (zeroverse_ptr->amin * zeroverse_ptr->hubble(zeroverse_ptr->amin));
	float a = zeroverse_ptr->amin;
	float tau;
	/*for (tau = taumin; tau < taumax - dtau; tau += dtau) {
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
	if (thread == 0) {
		printf(
				"Computing initial conditions for non-linear evolution to begin at redshift %e with a maximum over/under density of %e\n",
				1 / a - 1, drho);
	}
	einstein_boltzmann_init_set(states, zeroverse_ptr, kmin, kmax, Nk, zeroverse_ptr->amin, normalization);
	__syncthreads();
	einstein_boltzmann_interpolation_function(den_k, vel_k, states, zeroverse_ptr, kmin, kmax, Nk, zeroverse_ptr->amin,
			a);
	__syncthreads();
	for (int dim = 0; dim < NDIM; dim++) {
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
				host_parts[l].v[0] = v;
			}
		}
		__syncthreads();
		if (thread == 0) {
			printf("Computing %c positions\n", 'x' + dim);
		}
		float xdisp = zeldovich_displacements(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid, 0);
		__syncthreads();
		for (int ij = thread; ij < N * N; ij += block_size) {
			int i = ij / N;
			int j = ij % N;
			for (int k = 0; k < N; k++) {
				const int l = N * (N * i + j) + k;
				const int I[NDIM] = { i, j, k };
				float x = opts.box_size * (((float) I[dim] + 0.5f) / (float) N - 0.5f);
				x += phi[l].real();
				float test1 = host_parts[l].v[0] / phi[l].real();
				host_parts[l].x[dim] = float_to_pos(x);
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
		tree::initialize(parts, arena + N3 * sizeof(float) * 8, N3 * sizeof(float) * 2);
		CUDA_CHECK(cudaMalloc(&root, sizeof(tree)));
	}
	__syncthreads();
	__shared__ range<pos_type> root_range;
	if (thread < 3) {
		root_range.begin[thread] = 0x80000000;
		root_range.end[thread] = 0x7FFFFFFF;
	}
	__syncthreads();
	if (thread == 0) {
		printf("Sorting\n");
		root_tree_sort<<<1,1024>>>(root, host_parts, parts, parts+N3, root_range, 0);
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
		delete den_k;
		delete vel_k;
	}
}

int main() {

	options params;

	size_t stack_size;
	size_t desired_stack_size = 4 * 1024;
	size_t rlimit = 5;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, rlimit));
	CUDA_CHECK(cudaDeviceGetLimit(&rlimit, cudaLimitDevRuntimeSyncDepth));
	printf("CUDA recursion limit = %li\n", rlimit);
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, desired_stack_size));
	CUDA_CHECK(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
//	CUDA_CHECK(cudaThreadSetCacheConfig(cudaFuncCachePreferShared));
	particle* parts_ptr;
	printf("Stack Size = %li\n", stack_size);

	struct cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	params.clock_rate = prop.clockRate * pow(1024/1000,3) * 1000;
	printf( "Clock rate = %e\n", params.clock_rate);
	params.h = 0.697;
	params.Neff = 3.84;
	params.Y = 0.24;
	params.omega_b = 0.0240 / (params.h * params.h);
	params.omega_c = 0.1146 / (params.h * params.h);
	params.Theta = 1.0;
	params.Ngrid = 256;
	params.sigma8 = 0.8367;
	params.max_overden = 1.0;
	params.box_size = 1000;
	//	params.box_size = 613.0 / 2160.0 * params.Ngrid;
	params.nout = 64;
	params.max_kernel_depth = 3;
	params.parts_per_bucket = 64;
	params.nparts = params.Ngrid*params.Ngrid*params.Ngrid;
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + params.Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * params.Theta, 4) * std::pow(params.h, -2);
	params.omega_nu = omega_r * params.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + params.Neff);
	params.omega_gam = omega_r - params.omega_nu;
	void* arena;
	const int N = params.Ngrid;
	const int N3 = N * N * N;
	CUDA_CHECK(cudaMallocManaged(&parts_ptr, sizeof(particle) * N3));
	size_t arena_size = 10 * sizeof(float) * N3;
	printf("Allocating arena of %li Mbytes\n", (arena_size / 1024 / 1024));
	CUDA_CHECK(cudaMalloc(&arena, arena_size));
	main_kernel<<<1, BLOCK_SIZE>>>(arena, parts_ptr, params);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaFree(arena));
	CUDA_CHECK(cudaFree(parts_ptr));
}
