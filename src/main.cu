#include <nvfunctional>
#include <gputiger/chemistry.hpp>
#include <gputiger/options.hpp>
#include <gputiger/constants.hpp>
#include <gputiger/util.hpp>
#include <gputiger/zero_order.hpp>

#define BLOCK_SIZE 128

__device__ vector<boltz_real> *karray_ptr;

__device__ zero_order_universe *zeroverse_ptr;

using eb_results_type = vector<array<nvstd::function<boltz_real(boltz_real)>, NFIELD>>;

__device__ eb_results_type *eb_results_ptr;

__global__
void compute_einstein_boltzmann_solutions(zero_order_universe *uni) {
	const int thread = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__ vector<cos_state> *states_ptr;

	if (thread == 0) {
		states_ptr = new vector<cos_state>(block_size);
	}
	__syncthreads();
	assert(uni->hubble);
	boltz_real result;
	einstein_boltzmann(&result, uni, (*karray_ptr)[thread]);

	__syncthreads();
	if (thread == 0) {
		delete states_ptr;
	}
}

__global__
void main_kernel(cosmic_parameters opts) {
	const int thread = threadIdx.x;
	const int block_size = blockDim.x;
	double *result_ptr;
	sigma8_integrand *func_ptr;
	if (thread == 0) {
//		printf("Executing main kernel\n");
		zeroverse_ptr = new zero_order_universe;
		create_zero_order_universe(zeroverse_ptr, opts, 1.0);
//		assert(zeroverse_ptr->hubble);
	//	karray_ptr = new vector<boltz_real>;
	//	eb_results_ptr = new eb_results_type;
	//	karray_ptr->resize(opts.Ngrid / 2);
	//	eb_results_ptr->resize(opts.Ngrid / 2);
	}
	__syncthreads();

	if (thread == 0) {
		printf("Normalizing\n");
		result_ptr = new double;
		func_ptr = new sigma8_integrand;
		func_ptr->uni = zeroverse_ptr;
		integrate<sigma8_integrand, double> <<<1, BLOCK_SIZE>>>(func_ptr, log(1e-4), log(2.5),
				result_ptr, 1.0e-7);
		CUDA_CHECK(cudaDeviceSynchronize());
		*result_ptr = sqrt(*result_ptr);
		printf("Normalized to %e\n", *result_ptr);
	}
	__syncthreads();
//	int N = opts.Ngrid / 2;
//	for (int i = thread; i < N; i += block_size) {
//		if (i < N) {
//			karray[i] = (boltz_real) 2 * (boltz_real) M_PI
//					/ (boltz_real) (i + 1) / (boltz_real) opts.box_size;
//		}
//	}
//	__syncthreads();
//	if (thread == 0) {
//		printf("Computing linear Einstein-Boltzmann evolution\n");
//		printf("N = %i kmin = %e Mpc^-1 kmax = %e Mpc^-1\n", N, karray[0],
//				karray[N - 1]);
//		assert(zeroverse_ptr->hubble);
//		compute_einstein_boltzmann_solutions<<<1, N>>>(zeroverse_ptr);
//	}
//	__syncthreads();
//	if (thread == 0) {
//		CUDA_CHECK(cudaDeviceSynchronize());
//		printf("Done computing Einstein Boltzmann solution\n");
//	}
	__syncthreads();
	if (thread == 0) {
//		delete karray_ptr;
//		delete zeroverse_ptr;
//		delete eb_results_ptr;
//		delete result_ptr;
//		delete func_ptr;
	}
}

int main() {

	cosmic_parameters params;

//	size_t stack_size;
//	size_t desired_stack_size = 4 * 1024;
//	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, desired_stack_size));
//	CUDA_CHECK(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
//	CUDA_CHECK(cudaThreadSetCacheConfig(cudaFuncCachePreferShared));
//	printf("Stack Size = %li\n", stack_size);
	params.h = 0.697;
	params.Neff = 3.84;
	params.Y = 0.299;
	params.omega_b = 0.02256 / (params.h * params.h);
	params.omega_c = 0.1142 / (params.h * params.h);
	params.Theta = 2.72548 / 2.73;
	params.Ngrid = 32;
	params.box_size = 100.0;
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + params.Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.))
			* std::pow(constants::H0, -2) * std::pow(constants::c, -3)
			* std::pow(2.73 * params.Theta, 4) * std::pow(params.h, -2);
	params.omega_nu = omega_r * params.Neff
			/ (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + params.Neff);
	params.omega_gam = omega_r - params.omega_nu;

	main_kernel<<<1, BLOCK_SIZE>>>(params);

	CUDA_CHECK(cudaDeviceSynchronize());
}
