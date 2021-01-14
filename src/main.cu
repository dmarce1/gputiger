class cmplx {
	float x, y;
public:
	__device__ cmplx() = default;
	__device__ cmplx(float a) {
		x = a;
		y = 0.f;
	}
	__device__ cmplx(float a, float b) {
		x = a;
		y = b;
	}
	__device__ cmplx& operator+=(cmplx other) {
		x += other.x;
		y += other.y;
		return *this;
	}
	__device__ cmplx& operator-=(cmplx other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}
	__device__ cmplx operator*(cmplx other) const {
		cmplx a;
		a.x = x * other.x - y * other.y;
		a.y = x * other.y + y * other.x;
		return a;
	}
	__device__ cmplx operator/(cmplx other) const {
		return *this * other.conj() / other.norm();
	}
	__device__ cmplx operator/(float other) const {
		cmplx b;
		b.x = x / other;
		b.y = y / other;
		return b;
	}
	__device__ cmplx operator*(float other) const {
		cmplx b;
		b.x = x * other;
		b.y = y * other;
		return b;
	}
	__device__ cmplx operator+(cmplx other) const {
		cmplx a;
		a.x = x + other.x;
		a.y = y + other.y;
		return a;
	}
	__device__ cmplx operator-(cmplx other) const {
		cmplx a;
		a.x = x - other.x;
		a.y = y - other.y;
		return a;
	}
	__device__ cmplx conj() const {
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
		return ((*this) * conj()).real();
	}
	__device__
	float abs() const {
		return sqrtf(norm());
	}
	__device__ cmplx operator-() const {
		cmplx a;
		a.x = -x;
		a.y = -y;
		return a;
	}
};

__device__ cmplx operator*(float a, cmplx b) {
	return b * a;
}

#include <nvfunctional>
#include <gputiger/chemistry.hpp>
#include <gputiger/options.hpp>
#include <gputiger/constants.hpp>
#include <gputiger/util.hpp>
#include <gputiger/zero_order.hpp>
#include <gputiger/boltzmann.hpp>
#include <gputiger/zeldovich.hpp>
#include <gputiger/particle.hpp>

#define BLOCK_SIZE 256

__device__ zero_order_universe *zeroverse_ptr;

__device__ float test(float x) {
	return x * x;
}
__global__
void main_kernel(void* arena, particle* host_parts, cosmic_parameters opts) {
	const int thread = threadIdx.x;
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
	if (thread == 0) {
		zeroverse_ptr = new zero_order_universe;
		create_zero_order_universe(zeroverse_ptr, opts, 1.0);
	}
	__syncthreads();

	float kmin = 2.0 * (float) M_PI / opts.box_size;
	float kmax = kmin * (float) (opts.Ngrid / 2);
	kmax *= SQRT(3);
	int Nk = opts.Ngrid * SQRT(3) + 1;
	if (thread == 0) {
		printf("\nNormalizing Einstein Boltzman solutsions to a present day sigma8 of %e\n", opts.sigma8);
		result_ptr = new float;
		func_ptr = new sigma8_integrand;
		CUDA_CHECK(cudaMalloc(&states, sizeof(cos_state) * Nk));
		den_k = new interp_functor<float>;
		vel_k = new interp_functor<float>;
		basis = new cmplx[opts.Ngrid / 2];
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
	fft_basis(basis, opts.Ngrid);
	generate_random_normals(rands, N * N * N);
	__syncthreads();
	einstein_boltzmann_init_set(states, zeroverse_ptr, kmin, kmax, Nk, zeroverse_ptr->amin, normalization);
	int iter = 0;
	float dloga = 2.f;
	float logamin = LOG(zeroverse_ptr->amin);
	float logamax = LOG(zeroverse_ptr->amax);
	float loga = logamin;
	float drho;
	float a;
	for (; loga < LOG(zeroverse_ptr->amax) - dloga; loga += dloga) {
		if (iter > 1) {
			dloga = 2*(opts.max_overden - drho) / opts.max_overden;
		}
		float amin = EXP(loga);
		float amax = EXP(loga + dloga);
		if (thread == 0) {
			printf("Advancing Einstein Boltzmann solutions from redshift %.1f to %.1f\n", 1 / amin - 1, 1 / amax - 1);
		}
		einstein_boltzmann_interpolation_function(den_k, vel_k, states, zeroverse_ptr, kmin, kmax, Nk, amin, amax);
		__syncthreads();
		drho = zeldovich_overdensity(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid);
		__syncthreads();
		if (thread == 0) {
			printf("Maximum over/under density is %e\n", drho);
		}
		__syncthreads();
		if (iter > 0) {
			if (drho > opts.max_overden) {
				a = amax;
				break;
			}
		}
		__syncthreads();
		iter++;
	}
	__syncthreads();
	generate_random_normals(rands, N * N * N);
	__syncthreads();
	for (int dim = 0; dim < NDIM; dim++) {
		float vmax = zeldovich_velocities(phi, basis, rands, *vel_k, opts.box_size, opts.Ngrid, 0);
		__syncthreads();
		for (int ij = thread; ij < N*N; ij += block_size) {
			int i = ij / N;
			int j = ij % N;
			for (int k = 0; k < N; k++) {
				const int l = N * (N * i + j) + k;
				float v = phi[l].real();
				host_parts[l].v[0] = v;
			}
		}
		__syncthreads();
		float xdisp = zeldovich_displacements(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid, 0);
		__syncthreads();
		for (int ij = thread; ij < N*N; ij += block_size) {
			int i = ij / N;
			int j = ij % N;
			for (int k = 0; k < N; k++) {
				const int l = N * (N * i + j) + k;
				const int I[NDIM] = { i, j, k };
				float x = ((float) I[dim] + 0.5f) / (float) N - 0.5f;
				x += phi[l].real();
				float test1 = host_parts[l].v[0] / phi[l].real();
				printf("%e\n", test1);
				host_parts[l].x[dim] = float_to_pos(x);
			}
		}
		__syncthreads();
	}
	__syncthreads();
	for (int ij = thread; ij < N*N; ij += block_size) {
		int i = ij / N;
		int j = ij % N;
		for (int k = 0; k < N; k++) {
			const int l = N * (N * i + j) + k;
			host_parts[l].rung = 0;
		}
	}
	__syncthreads();
	if (thread == 0) {
		delete zeroverse_ptr;
		delete result_ptr;
		delete func_ptr;
		CUDA_CHECK(cudaFree(states));
		delete den_k;
		delete vel_k;
		delete[] basis;
	}
}

int main() {

	cosmic_parameters params;

	size_t stack_size;
	size_t desired_stack_size = 4 * 1024;
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, desired_stack_size));
	CUDA_CHECK(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
//	CUDA_CHECK(cudaThreadSetCacheConfig(cudaFuncCachePreferShared));
	particle* parts_ptr;
	printf("Stack Size = %li\n", stack_size);
	params.h = 0.7;
	params.Neff = 3.086;
	params.Y = 0.24;
	params.omega_b = 0.05;
	params.omega_c = 0.25;
	params.Theta = 1.0;
	params.Ngrid = 256;
	params.sigma8 = 0.8367;
	params.max_overden = 0.9;
	params.box_size = 100.0;
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + params.Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * params.Theta, 4) * std::pow(params.h, -2);
	params.omega_nu = omega_r * params.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + params.Neff);
	params.omega_gam = omega_r - params.omega_nu;
	void* arena;
	const int N = params.Ngrid;
	const int N3 = N * N * N;
	particle* parts_device;
	CUDA_CHECK(cudaHostAlloc(&parts_ptr, sizeof(particle) * N3, cudaHostAllocMapped));
	CUDA_CHECK(cudaHostGetDevicePointer(&parts_device, parts_ptr, 0));
	CUDA_CHECK(cudaMalloc(&arena, 2 * sizeof(cmplx) * N3));

	main_kernel<<<1, BLOCK_SIZE>>>(arena, parts_device, params);
	CUDA_CHECK(cudaGetLastError());

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaFree(arena));
	CUDA_CHECK(cudaFreeHost(parts_ptr));
}
