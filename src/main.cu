#include <nvfunctional>
#include <gputiger/chemistry.hpp>
#include <gputiger/options.hpp>
#include <gputiger/constants.hpp>
#include <gputiger/util.hpp>
#include <gputiger/zero_order.hpp>

#define BLOCK_SIZE 256

__device__ vector<float> *karray_ptr;

__device__ zero_order_universe *zeroverse_ptr;

using eb_results_type = vector<array<nvstd::function<float(float)>, NFIELD>>;

__device__ eb_results_type *eb_results_ptr;

__global__
void main_kernel(cosmic_parameters opts) {
	const int thread = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	double *result_ptr;
	sigma8_integrand *func_ptr;
	if (thread == 0) {
		zeroverse_ptr = new zero_order_universe;
		create_zero_order_universe(zeroverse_ptr, opts, 1.0);
	}
	__syncthreads();

	if (thread == 0) {
		printf("\nNormalizing Einstein Boltzman solutsions to a present day sigma8 of %e\n", opts.sigma8);
		result_ptr = new double;
		func_ptr = new sigma8_integrand;
		func_ptr->uni = zeroverse_ptr;
		integrate<sigma8_integrand, double> <<<1, BLOCK_SIZE>>>(func_ptr,
				LOG(0.25 / 16), LOG(0.25 * 16), result_ptr, 1.0e-6);
		CUDA_CHECK(cudaDeviceSynchronize());
		*result_ptr = SQRT(1.0 / *result_ptr);
		printf("The normalization value is %e\n", *result_ptr * pow(opts.sigma8, 2));
	}
	__syncthreads();
	if (thread == 0) {
		printf("Computing start time for non-linear evolution\n");
	}
	float normalization = *result_ptr;
	float kmax = (float) 2.0 * (float) M_PI / opts.box_size;
	float kmin = kmax / (float) (opts.Ngrid / 2);
	if (thread == 0) {
		printf("wave number range %e to %e Mpc^-1 for %i^3 grid and box size of %e Mpc\n", kmin, kmax, opts.Ngrid,
				opts.box_size);
	}

	float time = find_nonlinear_time(zeroverse_ptr, kmin, kmax, opts.box_size / opts.Ngrid, normalization);
	if (thread == 0) {
		printf("Non-linear evolution starts at redshift = %e\n", (float) 1 / time - (float) 1);
	}

	__syncthreads();
	if (thread == 0) {
		delete zeroverse_ptr;
		delete result_ptr;
		delete func_ptr;
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
	params.h = 0.7;
	params.Neff = 3.086;
	params.Y = 0.24;
	params.omega_b = 0.05;
	params.omega_c = 0.25;
	params.Theta = 1.0;
	params.Ngrid = 256;
	params.sigma8 = 0.761 / 0.8;
	params.box_size = 64.0;
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + params.Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * params.Theta, 4) * std::pow(params.h, -2);
	params.omega_nu = omega_r * params.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + params.Neff);
	params.omega_gam = omega_r - params.omega_nu;

	main_kernel<<<1, BLOCK_SIZE>>>(params);

	CUDA_CHECK(cudaDeviceSynchronize());
}
