/*
 * math.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_MATH_HPP_
#define GPUTIGER_MATH_HPP_

#include <gputiger/vector.hpp>
#include <gputiger/params.hpp>
#include <nvfunctional>

#include <cstdio>

__device__
static double find_root(nvstd::function<double(double)> f) {
	double x = 0.5;
	double err;
	int iters = 0;
	do {
		double dx0 = x * 1.0e-6;
		if (abs(dx0) == 0.0) {
			dx0 = 1.0e-10;
		}
		double fx = f(x);
		double dfdx = (f(x + dx0) - fx) / dx0;
		double dx = -fx / dfdx;
		err = abs(dx / max(1.0, abs(x)));
		x += 0.5 * dx;
		iters++;
		if (iters > 100000) {
			printf("Finished early with error = %e\n", err);
			break;
		}
	} while (err > 1.0e-14);
	return x;
}

template<class FUNC, class REAL>
__global__
void integrate(FUNC *fptr, REAL a, REAL b, REAL* result, REAL toler) {
	const int block_size = blockDim.x;
	int thread = threadIdx.x;
	const auto& f = *fptr;
	__syncthreads();
	__shared__
	REAL* sums1;
	__shared__
	REAL* sums2;
	__shared__
	REAL* error_ptr;
	if( thread == 0 ) {
		sums1 = new REAL[1024];
		sums2 = new REAL[1024];
		error_ptr = new REAL;
	}
	__syncthreads();
	REAL& err = *error_ptr;
	int N = 33;
	REAL sum1, sum2;
	do {
		sum1 = REAL(0);
		sum2 = REAL(0);
		REAL dx = (b - a) / REAL(N-1);
		REAL fa, fb;
		if( thread == 0 ) {
			fa = f(a);
			fb = f(b);
			sum1 = (fa + fb) * dx / REAL(3.0);
			sum2 = (fa + fb) * dx * REAL(2*7)/REAL(45);
		} else {
			sum1 = sum2 = 0.0;
		}
		__syncthreads();
		const REAL wt1[4] = {REAL(2)/REAL(3),REAL(4)/REAL(3),REAL(2)/REAL(3),REAL(4)/REAL(3)};
		const REAL wt2[4] = {REAL(4*7)/REAL(45),REAL(64)/REAL(45),REAL(24)/REAL(45),REAL(64)/REAL(45)};
			for (int i = 1 + thread; i < N-1; i += block_size) {
			REAL x = a + REAL(i) * dx;
			REAL this_f = f(x);
			sum2 += wt2[i%4] * this_f * dx;
			sum1 += wt1[i%4] * this_f * dx;
		}
		sums1[thread] = sum1;
		sums2[thread] = sum2;
			__syncthreads();
		for( int M = block_size / 2; M >= 1; M/=2) {
			if( thread < M) {
				sums1[thread] += sums1[thread+M];
				sums2[thread] += sums2[thread+M];
			}
			__syncthreads();
		}
		if( thread == 0 ) {
			sum1 = sums1[0];
			sum2 = sums2[0];
			if( sum2 != REAL(0)) {
				err = abs((sum2 - sum1)/sum2);
			} else if( sum1 != REAL(0.0)) {
				err = abs((sum1 - sum2)/sum1);
			} else {
				err = REAL(0.0);
			}
			*result = sum2;
			printf( "%i %e %e\n", N, sum1, sum2);
		}
		N = 2 * (N-1)+1;
		__syncthreads();
	}while( err > toler);
	__syncthreads();
	if( thread == 0 ) {
		delete [] sums1;
		delete [] sums2;
		delete error_ptr;
	}
	__syncthreads();

}

#endif /* GPUTIGER_MATH_HPP_ */
