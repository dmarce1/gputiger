/*
 * math.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_MATH_HPP_
#define GPUTIGER_MATH_HPP_

#include <nvfunctional>
#include <cstdio>
#include <cstdint>
#include <gputiger/vector.hpp>
#include <gputiger/params.hpp>

#define POW(a,b) powf(a,b)
#define LOG(a) logf(a)
#define EXP(a) expf(a)
#define SQRT(a) sqrtf(a)
#define COS(a) cosf(a)
#define SIN(a) sinf(a)
#define SINCOS(a,b,c) sincosf(a,b,c)

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

__device__  inline cmplx operator*(float a, cmplx b) {
	return b * a;
}

__device__  inline cmplx expc(cmplx z) {
	float x, y;
	float t = EXP(z.real());
	SINCOS(z.imag(), &y, &x);
	x *= t;
	y *= t;
	return cmplx(x, y);
}

__device__
double find_root(nvstd::function<double(double)> f);

template<class FUNC, class REAL>
__global__
void integrate(FUNC *fptr, REAL a, REAL b, REAL* result, REAL toler) {
	const int block_size = blockDim.x;
	int thread = threadIdx.x;
	const auto& f = *fptr;
	__syncthreads();
	__shared__ REAL* sums1;
	__shared__ REAL* sums2;
	__shared__ REAL* error_ptr;
	if (thread == 0) {
		sums1 = new REAL[1024];
		sums2 = new REAL[1024];
		error_ptr = new REAL;
	}
	__syncthreads();
	REAL& err = *error_ptr;
	int N = 6 * ((block_size - 1) / 6) + 1;
	REAL sum1, sum2;
	do {
		sum1 = REAL(0);
		sum2 = REAL(0);
		REAL dx = (b - a) / REAL(N - 1);
		sum1 = sum2 = 0.0;
		const REAL wt1[3] = { 6.0 / 8.0, 9.0 / 8.0, 9.0 / 8.0 };
		const REAL wt2[6] = { 82.0 / 140.0, 216.0 / 140.0, 27.0 / 140.0, 272.0 / 140.0, 27.0 / 140.0, 216.0 / 140.0 };
		for (int i = thread; i < N; i += block_size) {
			REAL x = a + REAL(i) * dx;
			REAL this_f = f(x);
			sum2 += this_f * dx * wt2[i % 6] * (i == 0 || i == N - 1 ? REAL(0.5) : REAL(1));
			sum1 += this_f * dx * wt1[i % 3] * (i == 0 || i == N - 1 ? REAL(0.5) : REAL(1));
		}
		sums1[thread] = sum1;
		sums2[thread] = sum2;
		__syncthreads();
		for (int M = block_size / 2; M >= 1; M /= 2) {
			if (thread < M) {
				sums1[thread] += sums1[thread + M];
				sums2[thread] += sums2[thread + M];
			}
			__syncthreads();
		}
		if (thread == 0) {
			sum1 = sums1[0];
			sum2 = sums2[0];
			if (sum2 != REAL(0)) {
				err = abs((sum2 - sum1) / sum2);
			} else if (sum1 != REAL(0.0)) {
				err = abs((sum1 - sum2) / sum1);
			} else {
				err = REAL(0.0);
			}
			*result = sum2;
		}
		N = 2 * (N - 1) + 1;
		__syncthreads();
	} while (err > toler);
	__syncthreads();
	if (thread == 0) {
		delete[] sums1;
		delete[] sums2;
		delete error_ptr;
	}
	__syncthreads();

}

__device__ inline float pow2(float r) {
	return r * r;
}

__device__
void generate_random_normals(cmplx* nums, int N);

#endif /* GPUTIGER_MATH_HPP_ */
