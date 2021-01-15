
#include <gputiger/math.hpp>


__device__
double find_root(nvstd::function<double(double)> f) {
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
	} while (err > 1.0e-12);
	return x;
}

__device__
void generate_random_normals(cmplx* nums, int N) {
	uint64_t mod = 1LL << 31LL;
	uint64_t a1 = 1664525LL;
	uint64_t a2 = 1103515245LL;
	uint64_t c1 = 1013904223LL;
	uint64_t c2 = 12345LL;
	const int& thread = threadIdx.x;
	const int& block_size = blockDim.x;
	uint32_t int1 = a1;
	uint32_t int2 = a2;
	for( int i = 0; i < 2*(block_size-thread); i++) {
		int1 *= a1;
		int2 *= a2;
		int1 >>= 8;
		int2 >>= 8;
	}
	int1 = int1 % mod;
	int2 = int2 % mod;
	for (int i = thread; i < N; i += block_size) {
		int1 = (a1 * (uint64_t) int1 + c1) % mod;
		int2 = (a2 * (uint64_t) int2 + c2) % mod;
		float x1 = ((float) int1 + 0.5f) / (float) uint64_t(mod + uint64_t(1));
		float y1 = ((float) int2 + 0.5f) / (float) uint64_t(mod + uint64_t(1));
		float x = x1;
		float y = 2.f * (float) M_PI * y1;
		nums[i] = SQRT(-LOG(abs(x))) * expc(cmplx(0, 1) * y);
	}
	__syncthreads();
}
