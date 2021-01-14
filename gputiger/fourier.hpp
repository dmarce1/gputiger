/*
 * fourier.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#ifndef FOURIER_HPP_
#define FOURIER_HPP_

#include <gputiger/params.hpp>
#include <gputiger/math.hpp>

__device__ void fft_basis(cmplx* X, int N) {
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	for (int i = thread; i < N / 2; i += block_size) {
		float omega = 2.0f * (float) M_PI * (float) i / (float) N;
		X[i] = expc(-cmplx(0, 1) * omega);
	}
	__syncthreads();
}

__global__
void fft3d(cmplx* Y, const cmplx* expi, int N) {
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	int level = 0;
	for (int i = N; i > 1; i >>= 1) {
		level++;
	}
	for (int dim = 0; dim < NDIM; dim++) {
		for (int xy = thread; xy < N * N; xy += block_size) {
			int xi = xy / N;
			int yi = xy % N;
			int offset = N * (N * xi + yi);
			cmplx* y = Y + offset;

			for (int i = 0; i < N; i++) {
				int j = 0;
				int l = i;
				for (int k = 0; k < level; k++) {
					j = (j << 1) | (l & 1);
					l >>= 1;
				}
				if (j > i) {
					auto tmp = y[i];
					y[i] = y[j];
					y[j] = tmp;
				}
			}

			for (int P = 2; P <= N; P *= 2) {
				const int s = N / P;
				for (int i = 0; i < N; i += P) {
					int k = 0;
					for (int j = i; j < i + P / 2; j++) {
						const auto t = y[j + P / 2] * expi[k];
						y[j + P / 2] = y[j] - t;
						y[j] += t;
						k += s;
					}
				}
			}
		}
		__syncthreads();
		if (dim == 0) {
			for (int xy = thread; xy < N * N; xy += block_size) {
				int xi = xy / N;
				int yi = xy % N;
				for (int zi = 0; zi < yi; zi++) {
					const int i1 = N * (N * xi + yi) + zi;
					const int i2 = N * (N * xi + zi) + yi;
					cmplx tmp = Y[i1];
					Y[i1] = Y[i2];
					Y[i2] = tmp;
				}
			}
			__syncthreads();
		} else if (dim == 1) {
			for (int xy = thread; xy < N * N; xy += block_size) {
				int xi = xy / N;
				int yi = xy % N;
				for (int zi = 0; zi < xi; zi++) {
					const int i1 = N * (N * xi + yi) + zi;
					const int i2 = N * (N * zi + yi) + xi;
					cmplx tmp = Y[i1];
					Y[i1] = Y[i2];
					Y[i2] = tmp;
				}
			}
			__syncthreads();
		} else {
			for (int xy = thread; xy < N * N; xy += block_size) {
				int xi = xy / N;
				int yi = xy % N;
				for (int zi = 0; zi < yi; zi++) {
					const int i1 = N * (N * xi + yi) + zi;
					const int i2 = N * (N * xi + zi) + yi;
					cmplx tmp = Y[i1];
					Y[i1] = Y[i2];
					Y[i2] = tmp;
				}
			}
			__syncthreads();
			for (int xy = thread; xy < N * N; xy += block_size) {
				int xi = xy / N;
				int yi = xy % N;
				if (xi < yi) {
					for (int zi = 0; zi < N; zi++) {
						const int i1 = N * (N * xi + yi) + zi;
						const int i2 = N * (N * yi + xi) + zi;
						cmplx tmp = Y[i1];
						Y[i1] = Y[i2];
						Y[i2] = tmp;
					}
				}
			}
			__syncthreads();
		}
	}
	__syncthreads();

}

#endif /* FOURIER_HPP_ */
