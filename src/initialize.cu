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
#define TRANSFERSIZE 1024

__global__
void transfer_data(particle* parts, particle* host_parts) {
	const int tid = threadIdx.x;
	const float a = 1.f / (opts.redshift + 1.f);
	const int N = opts.Ngrid;
	const int N3 = N * N * N;
	for (int i = tid; i < N3; i += TRANSFERSIZE) {
		parts[i] = host_parts[i];
		parts[i].rung = 0;
		for (int dim = 0; dim < NDIM; dim++) {
			parts[i].v[dim] *= a / opts.box_size;
			parts[i].x[dim] /= opts.box_size;
		}
	}
}

__global__
void velocities_to_particles(cmplx* phi, particle* parts, float ainit, int dim) {
	const int tid = threadIdx.x;
	const int N = opts.Ngrid;
	for (int ij = tid; ij < N * N; ij += TRANSFERSIZE) {
		int i = ij / N;
		int j = ij % N;
		for (int k = 0; k < N; k++) {
			const int l = N * (N * i + j) + k;
			float v = phi[l].real();
			parts[l].v[dim] = v * ainit * ainit / opts.box_size;
		}
	}
}

__global__ void displacements_to_particles(cmplx* phi, particle* parts, int dim) {
	const int tid = threadIdx.x;
	const int N = opts.Ngrid;
	for (int ij = tid; ij < N * N; ij += TRANSFERSIZE) {
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
			parts[l].x[dim] = x;
		}
	}

}

__device__
            extern cudaTextureObject_t* tex_ewald;

__global__
void initialize(void* arena, particle* host_parts, options opts_, cudaTextureObject_t* ewald_ptr, float* matterpow, float* velpow) {
	const int tid = threadIdx.x;
	if (tid == 0) {
		tex_ewald = ewald_ptr;
		opts = opts_;

		const int N = opts.Ngrid;
		const int N3 = N * N * N;
		cmplx* phi = (cmplx*) arena;
		cmplx* rands = ((cmplx*) arena) + N3;
		particle* parts = (particle*) arena;
		float kmin;
		float kmax;
		float dk;
		float kmin0;
		float kmax0;
		float vmax, xdisp;
		float normalization;
		float *result_ptr;
		zero_order_universe *zeroverse_ptr;
		sigma8_integrand *func_ptr;
		interp_functor<float>* den_k;
		interp_functor<float>* vel_k;
		cos_state* states;
		cmplx* basis;

		kmin = 2.0 * (float) M_PI / opts.box_size;
		kmax = sqrtf(3) * (kmin * (float) (opts.Ngrid));
		kmin0 = (1e-4 * opts.h);
		kmax0 = (5 * opts.h);
		dk = min(logf(kmax/kmin), 5/(1e-4)) / opts.Ngrid;
		kmin = min(kmin0,kmin);
		kmax = max(kmax0,kmax);
		int Nk = fmaxf(logf(kmax/kmin) / dk + 1, (float) opts.Nmp);


		CUDA_MALLOC(&zeroverse_ptr, sizeof(zero_order_universe));
		CUDA_MALLOC(&result_ptr, sizeof(float));
		CUDA_MALLOC(&func_ptr, sizeof(sigma8_integrand));
		CUDA_MALLOC(&states, sizeof(cos_state) * Nk);
		CUDA_MALLOC(&basis, sizeof(cmplx) * opts.Ngrid / 2);
		CUDA_MALLOC(&den_k, sizeof(interp_functor<float> ));
		CUDA_MALLOC(&vel_k, sizeof(interp_functor<float> ));
		printf("\tComputing zero order Universe\n");
		create_zero_order_universe(zeroverse_ptr, 1.0);

		printf("\tNormalizing Einstein-Boltzmann solutions\n");
		func_ptr->uni = zeroverse_ptr;
		func_ptr->littleh = opts.h;
		integrate<sigma8_integrand, float> <<<1, SIGMA8SIZE>>>(func_ptr,
				(float) LOG(0.25 / 128 * opts.h), (float) LOG(0.25 * 65 * opts.h), result_ptr, (float) 1.0e-6);
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

		Nk = opts.Nmp;
		dk = logf(kmax0/kmin0) / (Nk-1);
		for( int i = 0; i < Nk; i++) {
			float logk = log(kmin0) + i * dk;
			matterpow[i] = (*den_k)(expf(logk));
			velpow[i] = (*vel_k)(expf(logk));
		}

		printf("\tComputing FFT basis\n");
		fft_basis<<<1,FFTSIZE>>>(basis, opts.Ngrid);
		CUDA_CHECK(cudaDeviceSynchronize());

		printf("\tComputing random number set\n");
		generate_random_normals<<<1,RANDSIZE>>>(rands, N * N * N);
		CUDA_CHECK(cudaDeviceSynchronize());

		printf("\tComputing over/under density\n");
		zeldovich<<<1,ZELDOSIZE>>>(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid, 0, DENSITY, result_ptr);
		CUDA_CHECK(cudaDeviceSynchronize());
		const float drho = *result_ptr;
		CUDA_CHECK(cudaDeviceSynchronize());
		printf("\t\tOver/under density is %e\n", drho);

		for (int dim = 0; dim < NDIM; dim++) {
			printf("\t\tComputing %c velocities\n", 'x' + dim);
			zeldovich<<<1,ZELDOSIZE>>>(phi, basis, rands, *vel_k, opts.box_size, opts.Ngrid, dim, VELOCITY, result_ptr);
			velocities_to_particles<<<1,TRANSFERSIZE>>>(phi, host_parts, ainit, dim);
			CUDA_CHECK(cudaDeviceSynchronize());
			vmax = fmaxf(*result_ptr, vmax);
			printf("\t\tComputing %c positions\n", 'x' + dim);
			zeldovich<<<1,ZELDOSIZE>>>(phi, basis, rands, *den_k, opts.box_size, opts.Ngrid, dim, DISPLACEMENT, result_ptr);
			displacements_to_particles<<<1,TRANSFERSIZE>>>(phi, host_parts, dim);
			CUDA_CHECK(cudaDeviceSynchronize());
			xdisp = fmaxf(*result_ptr, xdisp);
		}
		xdisp /= opts.box_size / opts.Ngrid;
		printf("\t\tMaximum displacement is %e\n", xdisp);
		printf("\t\tMaximum velocity is %e\n", vmax);

		printf("\tTransferring data back from host\n");
		transfer_data<<<1,TRANSFERSIZE>>>(parts,host_parts);
		CUDA_CHECK(cudaDeviceSynchronize());
		double rho = zeroverse_ptr->redshift_to_density(opts.redshift);
		double vol = pow(opts.box_size * constants::mpc_to_cm, 3);
		double mpart = rho * vol / N3;
		opts.code_to_g = constants::G * pow2(constants::c) * opts.box_size * constants::mpc_to_cm;
		opts.code_to_cm = opts.box_size * constants::mpc_to_cm;
		opts.code_to_s = pow(opts.code_to_cm,1.5)*pow(opts.code_to_g,-0.5);
		printf("Initialization complete\n");
		printf("\tParticle mass = %e g/cm^3\n", mpart);
		printf("\t\tCode Units\n");
		printf( "\t\t\tmass   = %e g\n", opts.code_to_g);
		printf( "\t\t\tlength = %e cm\n", opts.code_to_cm);
		printf( "\t\t\ttime   = %e s\n", opts.code_to_s);
		CUDA_CHECK(cudaFree(zeroverse_ptr));
		CUDA_CHECK(cudaFree(&result_ptr));
		CUDA_CHECK(cudaFree(&func_ptr));
		CUDA_CHECK(cudaFree(&states));
		CUDA_CHECK(cudaFree(&basis));
		CUDA_CHECK(cudaFree(&den_k));
		CUDA_CHECK(cudaFree(&vel_k));
	}

}

