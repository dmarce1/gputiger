/*
 * zeldovich.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#ifndef ZELDOVICH_HPP_
#define ZELDOVICH_HPP_

#include <gputiger/params.hpp>
#include <gputiger/vector.hpp>
#include <gputiger/interp.hpp>
#include <gputiger/math.hpp>
#include <gputiger/fourier.hpp>

__device__ float zeldovich_overdensity(const cmplx* basis, const cmplx* rands, const interp_functor<float>& P,
		float box_size, int N) {
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	cmplx* den;
	den = arena.cmplx_space;
	__shared__ float* maxoverden;
	if (thread == 0) {
		maxoverden = new float[block_size];
	}
	__syncthreads();
	maxoverden[thread] = 0.f;
	__syncthreads();
	for (int dim = 0; dim < NDIM; dim++) {
		for (int i = thread; i < N * N * N; i += block_size) {
			den[i] = cmplx(0, 0);
		}
		__syncthreads();
		int rand_index = thread;
		for (int ij = thread; ij <= N * N; ij += block_size) {
			int i = ij / N;
			int j = ij % N;
			int i0 = i < N / 2 ? i : i - N;
			int j0 = j < N / 2 ? j : j - N;
			float kx = 2.f * (float) M_PI / box_size * float(i0);
			float ky = 2.f * (float) M_PI / box_size * float(j0);
			for (int l = 0; l < N / 2; l++) {
				int l0 = l < N / 2 ? l : l - N;
				int index0 = N * (N * i + j) + l;
				int index1 = N * (N * ((N - i) % N) + ((N - j) % N)) + ((N - l) % N);
				int i2 = i0 * i0 + j0 * j0 + l0 * l0;
				if (i2 > 0 && i2 < N * N / 4) {
					float kz = 2.f * (float) M_PI / box_size * float(l0);
					float k = sqrt(kx * kx + ky * ky + kz * kz);
					den[index0] = rands[rand_index] * SQRT(P(k)) * POW(box_size, -1.5);
					den[index1] = den[index0].conj();
					rand_index += block_size;
				}
			}
		}
	}
	__syncthreads();
	if (thread == 0) {
		fft3d<<<1,256>>>(den,basis, N);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	__syncthreads();
	for (int i = thread; i < N * N * N; i += block_size) {
		maxoverden[thread] = max(maxoverden[thread], abs(den[i].real()));
	}
	__syncthreads();
	for (int P = block_size / 2; P >= 1; P /= 2) {
		if (thread < P) {
			maxoverden[thread] = max(maxoverden[thread], maxoverden[thread + P]);
		}
		__syncthreads();
	}
	float maxover = maxoverden[0];
	if (thread == 0) {
		delete[] maxoverden;
	}
	return maxover;
}

__device__ float zeldovich_displacements(cmplx* phi, const cmplx* basis, const cmplx* rands,
		const interp_functor<float>& P, float box_size, int N, int dim) {
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	__shared__ float* maxdisp;
	if (thread == 0) {
		maxdisp = new float[block_size];
	}
	__syncthreads();
	maxdisp[thread] = 0.f;
	__syncthreads();

	for (int i = thread; i < N * N * N; i += block_size) {
		phi[i] = cmplx(0, 0);
	}
	__syncthreads();
	int rand_index = thread;
	for (int ij = thread; ij <= N * N; ij += block_size) {
		int i = ij / N;
		int j = ij % N;
		int i0 = i < N / 2 ? i : i - N;
		int j0 = j < N / 2 ? j : j - N;
		float kx = 2.f * (float) M_PI / box_size * float(i0);
		float ky = 2.f * (float) M_PI / box_size * float(j0);
		for (int l = 0; l < N; l++) {
			int l0 = l < N / 2 ? l : l - N;
			int index0 = N * (N * i + j) + l;
			int index1 = N * (N * ((N - i) % N) + ((N - j) % N)) + ((N - l) % N);
			int i2 = i0 * i0 + j0 * j0 + l0 * l0;
			if (i2 > 0 && i2 < N * N / 4) {
				float kz = 2.f * (float) M_PI / box_size * float(l0);
				float K[NDIM] = { kx, ky, kz };
				float k = sqrt(kx * kx + ky * ky + kz * kz);
				phi[index0] = -cmplx(0, 1) * (rands[rand_index] * SQRT(P(k))) * K[dim]
						/ (k * k)* POW(box_size,-1.5);
				phi[index1] = phi[index0].conj();
				rand_index += block_size;
			}
		}
	}
	__syncthreads();
	if (thread == 0) {
		fft3d<<<1,256>>>(phi,basis, N);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	__syncthreads();
	for (int i = thread; i < N * N * N; i += block_size) {
		maxdisp[thread] = max(maxdisp[thread], abs(phi[i].real()));
	}
	__syncthreads();
	for (int P = block_size / 2; P >= 1; P /= 2) {
		if (thread < P) {
			maxdisp[thread] = max(maxdisp[thread], maxdisp[thread + P]);
		}
		__syncthreads();
	}
	float this_max = maxdisp[0];
	if (thread == 0) {
		delete[] maxdisp;
	}
	return this_max;
}

/*
 __device__ void zeldovich(const interp_functor<float>& P, float box_size, int N) {
 const int& thread = threadIdx.x;
 const int& block_size = blockDim.x;
 cmplx* phi[NDIM];
 cmplx* den;
 phi[0] = arena.cmplx_space;
 phi[1] = arena.cmplx_space + N * N * N;
 phi[2] = arena.cmplx_space + 2 * N * N * N;
 den = arena.cmplx_space + 3 * N * N * N;
 cmplx* rands = arena.cmplx_space + 4 * N * N * N;
 __shared__ cmplx* basis;
 __shared__ float* maxx;
 __shared__ float* maxoverden;
 if (thread == 0) {
 basis = new cmplx[N / 2];
 maxx = new float[block_size];
 maxoverden = new float[block_size];
 }
 __syncthreads();
 fft_basis(basis, N);
 generate_random_normals(rands, N * N * N);
 maxx[thread] = 0.f;
 maxoverden[thread] = 0.f;
 __syncthreads();
 for (int dim = 0; dim < NDIM; dim++) {
 for (int i = thread; i < N * N * N; i += block_size) {
 den[i] = cmplx(0, 0);
 }
 __syncthreads();
 int rand_index = thread;
 for (int ij = thread; ij <= N * N; ij += block_size) {
 int i = ij / N;
 int j = ij % N;
 int i0 = i < N / 2 ? i : i - N;
 int j0 = j < N / 2 ? j : j - N;
 float kx = 2.f * (float) M_PI / box_size * float(i0);
 float ky = 2.f * (float) M_PI / box_size * float(j0);
 for (int l = 0; l < N/2; l++) {
 int l0 = l < N / 2 ? l : l - N;
 int index0 = N * (N * i + j) + l;
 int index1 = N * (N * ((N - i) % N) + ((N - j) % N)) + ((N - l) % N);
 int i2 = i0 * i0 + j0 * j0 + l0 * l0;
 if (i2 > 0 && i2 < N * N / 4) {
 float kz = 2.f * (float) M_PI / box_size * float(l0);
 float k = sqrt(kx * kx + ky * ky + kz * kz);
 den[index0] = rands[rand_index] * SQRT(P(k)) * POW(box_size,-1.5);
 den[index1] = den[index0].conj();
 rand_index+=block_size;
 }
 }
 }
 }
 for (int dim = 0; dim < NDIM; dim++) {
 for (int i = thread; i < N * N * N; i += block_size) {
 phi[dim][i] = cmplx(0, 0);
 }
 __syncthreads();
 int rand_index = thread;
 for (int ij = thread; ij <= N * N; ij += block_size) {
 int i = ij / N;
 int j = ij % N;
 int i0 = i < N / 2 ? i : i - N;
 int j0 = j < N / 2 ? j : j - N;
 float kx = 2.f * (float) M_PI / box_size * float(i0);
 float ky = 2.f * (float) M_PI / box_size * float(j0);
 for (int l = 0; l < N/2; l++) {
 int l0 = l < N / 2 ? l : l - N;
 int index0 = N * (N * i + j) + l;
 int index1 = N * (N * ((N - i) % N) + ((N - j) % N)) + ((N - l) % N);
 int i2 = i0 * i0 + j0 * j0 + l0 * l0;
 if (i2 > 0 && i2 < N * N / 4) {
 float kz = 2.f * (float) M_PI / box_size * float(l0);
 float K[NDIM] = { kx, ky, kz };
 float k = sqrt(kx * kx + ky * ky + kz * kz);
 phi[dim][index0] = -cmplx(0, 1) * (rands[rand_index] * SQRT(P(k))) * K[dim] / (k * k) * POW(box_size,-1.5);
 phi[dim][index1] = phi[dim][index0].conj();
 rand_index+=block_size;
 }
 }
 }
 }
 __syncthreads();
 if (thread == 0) {
 fft3d<<<NDIM+1,256>>>(phi[0],basis, N);
 CUDA_CHECK(cudaGetLastError());
 CUDA_CHECK(cudaDeviceSynchronize());
 }
 __syncthreads();
 for (int i = thread; i < N * N * N; i += block_size) {
 for (int dim = 0; dim < NDIM; dim++) {
 maxx[thread] = max(maxx[thread], (float) N * phi[dim][i].real() / box_size);
 }
 maxoverden[thread] = max(maxoverden[thread], abs(den[i].real()));
 }
 __syncthreads();
 for (int P = block_size / 2; P >= 1; P /= 2) {
 if (thread < P) {
 maxx[thread] = max(maxx[thread], maxx[thread + P]);
 maxoverden[thread] = max(maxoverden[thread], maxoverden[thread + P]);
 }
 __syncthreads();
 }
 if (thread == 0) {
 printf("maximum over density          = %e\n", maxoverden[0]);
 printf("maximum interparticle spacing = %e\n", maxx[0]);
 delete[] basis;
 delete[] maxx;
 delete[] maxoverden;
 }
 }
 */
#endif /* ZELDOVICH_HPP_ */
