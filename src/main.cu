
class cmplx {
	float x, y;
public:
	__device__
	cmplx() = default;
	__device__
	cmplx(float a) {
		x = a;
		y = 0.f;
	}
	__device__
	cmplx(float a, float b) {
		x = a;
		y = b;
	}
	__device__
	cmplx& operator+=(cmplx other) {
		x += other.x;
		y += other.y;
		return *this;
	}
	__device__
	cmplx& operator-=(cmplx other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}
	__device__
	cmplx operator*(cmplx other) const {
		cmplx a;
		a.x = x * other.x - y * other.y;
		a.y = x * other.y + y * other.x;
		return a;
	}
	__device__
	cmplx operator/(cmplx other) const {
		return *this * other.conj() / other.norm();
	}
	__device__
	cmplx operator/(float other) const {
		cmplx b;
		b.x = x / other;
		b.y = y / other;
		return b;
	}
	__device__
	cmplx operator*(float other) const {
		cmplx b;
		b.x = x * other;
		b.y = y * other;
		return b;
	}
	__device__
	cmplx operator+(cmplx other) const {
		cmplx a;
		a.x = x + other.x;
		a.y = y + other.y;
		return a;
	}
	__device__
	cmplx operator-(cmplx other) const {
		cmplx a;
		a.x = x - other.x;
		a.y = y - other.y;
		return a;
	}
	__device__
	cmplx conj() const {
		cmplx a;
		a.x = x;
		a.y = -y;
		return a;
	}
	__device__
	float real() const {
		return x;
	}
	__device__
	float imag() const {
		return y;
	}
	__device__
	float norm() const {
		return ((*this)*conj()).real();
	}
	__device__
	float abs() const {
		return sqrtf(norm());
	}
	__device__
	cmplx operator-() const {
		cmplx a;
		a.x = -x;
		a.y = -y;
		return a;
	}
};

__device__
cmplx operator*(float a, cmplx b)  {
	return b * a;
}


struct arena_t {
	cmplx* cmplx_space;
};

__device__ arena_t arena;

#include <nvfunctional>
#include <gputiger/chemistry.hpp>
#include <gputiger/options.hpp>
#include <gputiger/constants.hpp>
#include <gputiger/util.hpp>
#include <gputiger/zero_order.hpp>
#include <gputiger/boltzmann.hpp>
#include <gputiger/zeldovich.hpp>

#define BLOCK_SIZE 256

__device__ zero_order_universe *zeroverse_ptr;

__device__ interp_functor<float>* power_interp_ptr;


__device__ float test(float x) {
	return x*x;
}
__global__
void main_kernel(arena_t arena_, cosmic_parameters opts) {
	const int thread = threadIdx.x;
	if (thread == 0) {
		arena = arena_;
	}
	__syncthreads();
	__shared__
	double *result_ptr;
	__shared__
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
		func_ptr->littleh= opts.h;
		integrate<sigma8_integrand, double> <<<1, BLOCK_SIZE>>>(func_ptr,
				LOG(0.25 / 32.0 * opts.h), LOG(0.25 * 32.0 * opts.h), result_ptr, 1.0e-6);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		*result_ptr = SQRT(opts.sigma8*opts.sigma8 / *result_ptr);
		printf("The normalization value is %e\n", *result_ptr);
	}
	__syncthreads();
	if (thread == 0) {
		printf("Computing start time for non-linear evolution\n");
	}
	float normalization = *result_ptr;
	float kmin = 2.0 * (float) M_PI / opts.box_size;
	float kmax = kmin * (float) (opts.Ngrid / 2);
	kmax *= SQRT(3);
	if (thread == 0) {
		printf("wave number range %e to %e Mpc^-1 for %i^3 grid and box size of %e Mpc\n", kmin, kmax, opts.Ngrid,
				opts.box_size);
	}
//	normalization *= ;
	float time = find_nonlinear_time(zeroverse_ptr, kmin, kmax, opts.box_size, normalization);
	if (thread == 0) {
		float z = (float) 1 / time - (float) 1;
		printf("Non-linear evolution starts at redshift = %.1f, \n", z);
		print_time(zeroverse_ptr->redshift_to_time(z));
		printf("\t after the Big Bang.\n");
	}
	__syncthreads();
	auto tmp = compute_einstein_boltzmann_interpolation_function(zeroverse_ptr, kmin, kmax, normalization, time);
	if (thread == 0) {
		power_interp_ptr = new interp_functor<float>;
		*power_interp_ptr = tmp;
	}
	__syncthreads();
	auto& pfunc = *power_interp_ptr;
	if (thread == 0) {
		printf("Normalized spectrum in range for simulation\n");
		for (double k = kmin; k < kmax; k *= POW(kmax / kmin, 1.0 / 25.0)) {
			printf("%e %e\n", k, pfunc(k));
		}
		printf("\n");
	}
	__syncthreads();
	if (thread == 0) {
		printf("Generating density Fourier transform\n");
	}
	zeldovich(pfunc, opts.box_size, opts.Ngrid);
	__syncthreads();
	if (thread == 0) {
		delete zeroverse_ptr;
		delete result_ptr;
		delete func_ptr;
		delete power_interp_ptr;
	}
}

int main() {

	cosmic_parameters params;

	size_t stack_size;
	size_t desired_stack_size = 4 * 1024;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, desired_stack_size));
	CUDA_CHECK(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
//	CUDA_CHECK(cudaThreadSetCacheConfig(cudaFuncCachePreferShared));
	printf("Stack Size = %li\n", stack_size);
	params.h = 0.7;
	params.Neff = 3.086;
	params.Y = 0.24;
	params.omega_b = 0.05;
	params.omega_c = 0.25;
	params.Theta = 1.0;
	params.Ngrid = 256;
	params.sigma8 = 0.8367;
	params.box_size = 100.0;
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + params.Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * params.Theta, 4) * std::pow(params.h, -2);
	params.omega_nu = omega_r * params.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + params.Neff);
	params.omega_gam = omega_r - params.omega_nu;
	arena_t arena;
	const int N = params.Ngrid;
	CUDA_CHECK(cudaMalloc(&arena.cmplx_space, 5*sizeof(cmplx) * N * N * N));

	main_kernel<<<1, BLOCK_SIZE>>>(arena, params);
	CUDA_CHECK(cudaGetLastError());


	CUDA_CHECK(cudaDeviceSynchronize());
}
