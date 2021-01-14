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

__device__ float zeldovich_overdensity(cmplx* den, const cmplx* basis, const cmplx* rands,
		const interp_functor<float>& P, float box_size, int N) {
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	__shared__ float* maxoverden;
	if (thread == 0) {
		maxoverden = new float[block_size];
	}
	__syncthreads();
	maxoverden[thread] = 0.f;
	__syncthreads();
	for (int i = thread; i < N * N * N; i += block_size) {
		den[i] = cmplx(0, 0);
	}
	__syncthreads();
	for (int ij = thread; ij < N * N; ij += block_size) {
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
			if (!((l == 0) && (j == 0) && (i >= N / 2)) && !((l == 0) && (j >= N / 2))) {
				if (i2 > 0 && i2 < N * N / 4) {
					float kz = 2.f * (float) M_PI / box_size * float(l0);
					float k = sqrt(kx * kx + ky * ky + kz * kz);
					den[index0] = rands[index0] * SQRT(P(k)) * POW(box_size, -1.5);
					den[index1] = den[index0].conj();
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
	for (int ij = thread; ij < N * N; ij += block_size) {
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
			if (!((l == 0) && (j == 0) && (i >= N / 2)) && !((l == 0) && (j >= N / 2))) {
				if (i2 > 0 && i2 < N * N / 4) {
					float kz = 2.f * (float) M_PI / box_size * float(l0);
					float K[NDIM] = { kx, ky, kz };
					float k = sqrt(kx * kx + ky * ky + kz * kz);
					phi[index0] = -cmplx(0, 1) * (rands[index0] * SQRT(P(k))) * K[dim] / (k * k) * POW(box_size, -1.5);
					phi[index1] = phi[index0].conj();
				}
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
	this_max *= (float) N / box_size;
	return this_max;
}

__device__ float zeldovich_velocities(cmplx* vel_k, const cmplx* basis, const cmplx* rands,
		const interp_functor<float>& P, float box_size, int N, int dim) {
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	__shared__ float* maxvel;
	if (thread == 0) {
		maxvel = new float[block_size];
	}
	__syncthreads();
	maxvel[thread] = 0.f;
	__syncthreads();

	for (int i = thread; i < N * N * N; i += block_size) {
		vel_k[i] = cmplx(0, 0);
	}
	__syncthreads();
	for (int ij = thread; ij < N * N; ij += block_size) {
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
			if (!((l == 0) && (j == 0) && (i >= N / 2)) && !((l == 0) && (j >= N / 2))) {
				if (i2 > 0 && i2 < N * N / 4) {
					float kz = 2.f * (float) M_PI / box_size * float(l0);
					float K[NDIM] = { kx, ky, kz };
					float k = sqrt(kx * kx + ky * ky + kz * kz);
					vel_k[index0] = -cmplx(0, 1) * (rands[index0] * SQRT(P(k))) * K[dim] / k * POW(box_size, -1.5);
					vel_k[index1] = vel_k[index0].conj();
				}
			}
		}
	}
	__syncthreads();
	if (thread == 0) {
		fft3d<<<1,256>>>(vel_k,basis, N);
		CUDA_CHECK(cudaGetLastError());

		CUDA_CHECK(cudaDeviceSynchronize());
	}
	__syncthreads();
	for (int i = thread; i < N * N * N; i += block_size) {
		maxvel[thread] = max(maxvel[thread], abs(vel_k[i].real()));
	}
	__syncthreads();
	for (int P = block_size / 2; P >= 1; P /= 2) {
		if (thread < P) {
			maxvel[thread] = max(maxvel[thread], maxvel[thread + P]);
		}
		__syncthreads();
	}
	float this_max = maxvel[0];
	if (thread == 0) {
		delete[] maxvel;
	}
	this_max *= (float) N / box_size;
	return this_max;
}

#endif /* ZELDOVICH_HPP_ */
