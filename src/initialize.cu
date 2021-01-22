#include <gputiger/initialize.hpp>
#include <gputiger/boltzmann.hpp>
#include <gputiger/zero_order.hpp>
#include <gputiger/particle.hpp>
#include <gputiger/fourier.hpp>
#include <gputiger/zeldovich.hpp>

#define SIGMA8SIZE 1024
#define EBSIZE 1024
#define FFTSIZE 1024
#define RANDSIZE 1024

__global__
void initialize(void* arena, particle* host_parts, options opts_, cudaTextureObject_t* ewald_ptr) {
	const int tid = threadIdx.x;
	if (tid == 0) {
		opts = opts_;

		const int N = opts.Ngrid;
		const int N3 = N * N * N;
		cmplx* phi = (cmplx*) arena;
		cmplx* rands = ((cmplx*) arena) + N3;
		particle* parts = (particle*) arena;
		float kmin = 2.0 * (float) M_PI / opts.box_size;
		float kmax = sqrtf(3) * (kmin * (float) (opts.Ngrid / 2));
		int Nk = 2 * opts.Ngrid * SQRT(3);

		float normalization;
		float *result_ptr;
		zero_order_universe *zeroverse_ptr;
		sigma8_integrand *func_ptr;
		interp_functor<float>* den_k;
		interp_functor<float>* vel_k;
		cos_state* states;
		cmplx* basis;

		CUDA_MALLOC(&zeroverse_ptr, sizeof(zero_order_universe));
		CUDA_MALLOC(&result_ptr, sizeof(float));
		CUDA_MALLOC(&func_ptr, sizeof(sigma8_integrand));
		CUDA_MALLOC(&states, sizeof(cos_state) * Nk);
		CUDA_MALLOC(&basis, sizeof(cmplx) * opts.Ngrid / 2);
		CUDA_MALLOC(&den_k, sizeof(interp_functor<float> ));
		CUDA_MALLOC(&vel_k, sizeof(interp_functor<float> ));
		printf("\tComputing zero order Universe\n");
		create_zero_order_universe(zeroverse_ptr, 1.0);

		printf("\tNormalizing Eintein-Boltzmann solutions\n");
		func_ptr->uni = zeroverse_ptr;
		func_ptr->littleh = opts.h;
		integrate<sigma8_integrand, float> <<<1, SIGMA8SIZE>>>(func_ptr,
				(float) LOG(0.25 / sqrtf(1000) * opts.h), (float) LOG(0.25 * sqrtf(1000) * opts.h), result_ptr, (float) 1.0e-6);
		CUDA_CHECK(cudaDeviceSynchronize());
		*result_ptr = SQRT(opts.sigma8 * opts.sigma8 / *result_ptr);
		normalization = *result_ptr;
		printf("\t\t\The normalization value is %e\n", *result_ptr);

		const float ainit = 1.0f / (opts.redshift + 1.0f);
		printf("\tComputing Einstain-Boltzmann interpolation solutions for wave numbers %e to %e Mpc^-1\n", kmin, kmax,
				opts.Ngrid, opts.box_size);
		int block_size = max(EBSIZE, Nk);
		einstein_boltzmann_interpolation_function<<<1, block_size>>>(den_k, vel_k, states, zeroverse_ptr, kmin, kmax, normalization, Nk, zeroverse_ptr->amin, ainit);
		CUDA_CHECK(cudaDeviceSynchronize());

		printf("\tComputing FFT basis\n");
		fft_basis<<<1,FFTSIZE>>>(basis, opts.Ngrid);
		CUDA_CHECK(cudaDeviceSynchronize());

		printf("\tComputing random number set\n");
		generate_random_normals<<<1,RANDSIZE>>>(rands, N * N * N);
		CUDA_CHECK(cudaDeviceSynchronize());

		printf("\tComputing over/under density\n");
		const float drho = zeldovich_overdensity(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid);
		CUDA_CHECK(cudaDeviceSynchronize());
		printf("\t\tOver/under density is %e\n", drho);

		/*
		 for (int dim = 0; dim < NDIM; dim++) {
		 if (tid == 0) {
		 printf("\tComputing %c velocities\n", 'x' + dim);
		 }
		 float vmax = zeldovich_velocities(phi, basis, rands, *vel_k, opts.box_size, opts.Ngrid, 0);
		 for (int ij = tid; ij < N * N; ij += WARPSIZE) {
		 int i = ij / N;
		 int j = ij % N;
		 for (int k = 0; k < N; k++) {
		 const int l = N * (N * i + j) + k;
		 float v = phi[l].real();
		 host_parts[l].v[0] = v * a / opts.box_size;
		 }
		 }
		 if (tid == 0) {
		 printf("\tComputing %c positions\n", 'x' + dim);
		 }
		 float xdisp = zeldovich_displacements(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid, 0);
		 for (int ij = tid; ij < N * N; ij += WARPSIZE) {
		 int i = ij / N;
		 int j = ij % N;
		 for (int k = 0; k < N; k++) {
		 const int l = N * (N * i + j) + k;
		 const int I[NDIM] = { i, j, k };
		 float x = (((float) I[dim] + 0.5f) / (float) N);
		 x += phi[l].real() / opts.box_size;
		 while (x > 1.0) {
		 x -= 1.0;
		 }
		 while (x < 0.0) {
		 x += 1.0;
		 }
		 host_parts[l].x[dim] = x;
		 }
		 }
		 }


		 /*
		 printf("\tComputing randoms\n");
		 printf("\tInitializing EB\n");
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
		 for (tau = taumin; tau < taumax - dtau; tau += dtau) {
		 last_a = a;
		 a = zeroverse_ptr->conformal_time_to_scale_factor(tau + dtau);
		 if (tid == 0) {
		 printf("\tAdvancing Einstein Boltzmann solutions from redshift %.1f to %.1f\n", 1 / last_a - 1,
		 1 / a - 1);
		 }
		 last_drho = drho;
		 drho = zeldovich_overdensity(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid);
		 if (tid == 0) {
		 printf("\tMaximum over/under density is %e\n", drho);
		 }
		 if (iter > 0) {
		 if (drho > opts.max_overden) {
		 drho = last_drho;
		 break;
		 }
		 }
		 iter++;
		 }
		 a = zeroverse_ptr->conformal_time_to_scale_factor(tau);

		 a = 0.025;
		 if (tid == 0) {
		 printf(
		 "Computing initial conditions for non-linear evolution to begin at redshift %e with a maximum over/under density of %e\n",
		 1 / a - 1, drho);
		 }
		 einstein_boltzmann_init_set(states, zeroverse_ptr, kmin, kmax, Nk, zeroverse_ptr->amin, normalization);
		 einstein_boltzmann_interpolation_function(den_k, vel_k, states, zeroverse_ptr, kmin, kmax, Nk,
		 zeroverse_ptr->amin, a);
		 for (int dim = 0; dim < NDIM; dim++) {
		 if (tid == 0) {
		 printf("\tComputing %c velocities\n", 'x' + dim);
		 }
		 float vmax = zeldovich_velocities(phi, basis, rands, *vel_k, opts.box_size, opts.Ngrid, 0);
		 for (int ij = tid; ij < N * N; ij += WARPSIZE) {
		 int i = ij / N;
		 int j = ij % N;
		 for (int k = 0; k < N; k++) {
		 const int l = N * (N * i + j) + k;
		 float v = phi[l].real();
		 host_parts[l].v[0] = v * a / opts.box_size;
		 }
		 }
		 if (tid == 0) {
		 printf("\tComputing %c positions\n", 'x' + dim);
		 }
		 float xdisp = zeldovich_displacements(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid, 0);
		 for (int ij = tid; ij < N * N; ij += WARPSIZE) {
		 int i = ij / N;
		 int j = ij % N;
		 for (int k = 0; k < N; k++) {
		 const int l = N * (N * i + j) + k;
		 const int I[NDIM] = { i, j, k };
		 float x = (((float) I[dim] + 0.5f) / (float) N);
		 x += phi[l].real() / opts.box_size;
		 while (x > 1.0) {
		 x -= 1.0;
		 }
		 while (x < 0.0) {
		 x += 1.0;
		 }
		 host_parts[l].x[dim] = x;
		 }
		 }
		 }
		 for (int ij = tid; ij < N * N; ij += WARPSIZE) {
		 int i = ij / N;
		 int j = ij % N;
		 for (int k = 0; k < N; k++) {
		 const int l = N * (N * i + j) + k;
		 host_parts[l].rung = 0;
		 }
		 }
		 for (int i = tid; i < N3; i += WARPSIZE) {
		 parts[i] = host_parts[i];
		 for (int dim = 0; dim < NDIM; dim++) {
		 parts[i].v[dim] *= a;
		 }
		 }*/
		CUDA_CHECK(cudaFree(zeroverse_ptr));
		CUDA_CHECK(cudaFree(&result_ptr));
		CUDA_CHECK(cudaFree(&func_ptr));
		CUDA_CHECK(cudaFree(&states));
		CUDA_CHECK(cudaFree(&basis));
		CUDA_CHECK(cudaFree(&den_k));
		CUDA_CHECK(cudaFree(&vel_k));
	}

	/*	__shared__ tree* root;
	 __syncthreads();
	 if (tid == 0) {
	 printf("\tBegining non-linear evolution\n");
	 tree::initialize(parts, arena + N3 * sizeof(float) * 8, N3 * sizeof(float) * TREESPACE, etable);
	 CUDA_CHECK(cudaMalloc(&root, sizeof(tree)));
	 }
	 __syncthreads();
	 __shared__ range root_range;
	 if (tid < 3) {
	 root_range.begin[tid] = float(0.0);
	 root_range.end[tid] = float(1.0);
	 }
	 __syncthreads();
	 if (tid == 0) {
	 printf("\tSorting\n");
	 root_tree_sort<<<1,MAXTHREADCOUNT>>>(root, host_parts, parts, parts+N3, root_range);
	 CUDA_CHECK(cudaGetLastError());
	 }
	 __syncthreads();
	 if (tid == 0) {
	 CUDA_CHECK(cudaDeviceSynchronize());
	 }
	 __syncthreads();
	 if (tid == 0) {
	 printf("\tKicking\n");
	 root->kick(root, 0, 0.1, ewald_ptr);
	 CUDA_CHECK(cudaGetLastError());
	 }
	 __syncthreads();
	 if (tid == 0) {
	 CUDA_CHECK(cudaDeviceSynchronize());
	 }
	 __syncthreads();

	 if (tid == 0) {
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
	 }*/

}
