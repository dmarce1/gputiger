#include <gputiger/math.hpp>
#include <nvfunctional>
#include <gputiger/chemistry.hpp>
#include <gputiger/constants.hpp>
#include <gputiger/util.hpp>
#include <gputiger/zero_order.hpp>
#include <gputiger/boltzmann.hpp>
#include <gputiger/zeldovich.hpp>
#include <gputiger/particle.hpp>
#include <gputiger/tree.hpp>
#include <gputiger/ewald.hpp>
#include <gputiger/initialize.hpp>
#include <cuda.h>
#include <curand.h>
#include <cmath>

cudaTextureObject_t* host_ewald;

#define STACK_SIZE (4*1024)
#define KERNEL_DEPTH 13
#define HEAP_SIZE (4*1024*1024)

int main() {
	size_t stack_size = STACK_SIZE;
	size_t recur_limit = KERNEL_DEPTH;
	size_t heap_size = HEAP_SIZE;
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop,0 ));
	//printf( "Maximum pitch %lli\n", prop.memPitch);
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, recur_limit));
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
	CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
	CUDA_CHECK(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
	CUDA_CHECK(cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize));
	CUDA_CHECK(cudaDeviceGetLimit(&recur_limit, cudaLimitDevRuntimeSyncDepth));
	bool success = true;
	if (stack_size != STACK_SIZE) {
		printf("Unable to allocate stack size of %li\n", STACK_SIZE);
		success = false;
	}
	if (heap_size != HEAP_SIZE) {
		printf("Unable to allocate heap size of %li\n", HEAP_SIZE);
		success = false;
	}
	if (recur_limit != KERNEL_DEPTH) {
		printf("Unable to set kernel recursion depth to %i\n", KERNEL_DEPTH);
		success = false;
	}
	if (!success) {
		return -1;
	}

	options opts;
	opts.redshift = 20.0;
	opts.h = 0.697;
	opts.Neff = 3.84;
	opts.Y = 0.24;
	opts.omega_b = 0.0240 / (opts.h * opts.h);
	opts.omega_c = 0.1146 / (opts.h * opts.h);
	opts.Theta = 1.0;
	opts.Ngrid = 256;
	opts.sigma8 = 0.8367;
	opts.max_overden = 1.0;
	opts.box_size = 1000;//613.0 *opts.Ngrid/2160.0;
	//	opts.box_size = 613.0 / 2160.0 * opts.Ngrid;
	opts.nout = 64;
	opts.max_kernel_depth = KERNEL_DEPTH - 1;
	opts.parts_per_bucket = 64;
	opts.opening_crit = 0.7;
	opts.particle_mass = 1.0;
	opts.G = 1.0;
	opts.nparts = opts.Ngrid * opts.Ngrid * opts.Ngrid;
	opts.hsoft = opts.Ngrid / 50.0;
	double omega_r = 32.0 * M_PI / 3.0 * constants::G * constants::sigma
			* (1 + opts.Neff * (7. / 8.0) * std::pow(4. / 11., 4. / 3.)) * std::pow(constants::H0, -2)
			* std::pow(constants::c, -3) * std::pow(2.73 * opts.Theta, 4) * std::pow(opts.h, -2);
	opts.omega_nu = omega_r * opts.Neff / (8.0 / 7.0 * std::pow(11.0 / 4.0, 4.0 / 3.0) + opts.Neff);
	opts.omega_gam = omega_r - opts.omega_nu;

	printf("Computing Ewald tables\n");
	ewald_table_t* etable;
	CUDA_MALLOC_MANAGED(&etable, sizeof(ewald_table_t));
	CUDA_MALLOC_MANAGED(&host_ewald, sizeof(cudaTextureObject_t));
	compute_ewald_table<<<EWALD_DIM*EWALD_DIM, EWALD_DIM>>>(etable);
	CUDA_CHECK(cudaDeviceSynchronize());
	for (int i = 0; i < NDIM + 1; i++) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaArray *d_cuArr;
		CUDA_CHECK(cudaMalloc3DArray(&d_cuArr, &channelDesc, make_cudaExtent(EWALD_DIM, EWALD_DIM, EWALD_DIM), 0));
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPos = make_cudaPos(0, 0, 0);
		copyParams.dstPos = make_cudaPos(0, 0, 0);
		copyParams.srcPtr = make_cudaPitchedPtr((*etable)[i].data(), EWALD_DIM * sizeof(float), EWALD_DIM,
				EWALD_DIM);
		copyParams.dstArray = d_cuArr;
		copyParams.extent = make_cudaExtent(EWALD_DIM, EWALD_DIM, EWALD_DIM);
		copyParams.kind = cudaMemcpyDeviceToDevice;
		CUDA_CHECK(cudaMemcpy3D(&copyParams));
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = d_cuArr;
		cudaTextureDesc texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = false;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeElementType;
		CUDA_CHECK(cudaCreateTextureObject(&host_ewald[i], &texRes, &texDescr, NULL));
	}

	printf("Setting up initial conditions for a red shift of %f\n", opts.redshift);
	particle* parts_ptr;
	void* arena;
	const int N = opts.Ngrid;
	const int N3 = N * N * N;
	size_t arena_size = (8 + TREESPACE) * sizeof(float) * N3;
	CUDA_MALLOC_MANAGED(&parts_ptr, sizeof(particle) * N3);
	CUDA_MALLOC_MANAGED(&arena, arena_size);
	initialize<<<1, WARPSIZE>>>(arena, parts_ptr, opts, host_ewald);
	CUDA_CHECK(cudaDeviceSynchronize());


	/*	 size_t stack_size;
	 size_t desired_stack_size = 4 * 1024;
	 size_t rlimit = KERNEL_DEPTH + 1;
	 size_t heapsize = 4 * 1024 * 1024;
	 CUDA_CHECK(cudaDeviceGetLimit(&rlimit, cudaLimitDevRuntimeSyncDepth));
	 printf("CUDA recursion limit = %li\n", rlimit);
	 CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	 printf("heapsize = %li\n", heapsize / 1024 / 1024);
	 //	CUDA_CHECK(cudaThreadSetCacheConfig(cudaFuncCachePreferShared));
	 printf("Stack Size = %li\n", stack_size);

	 struct cudaDeviceProp prop;
	 CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	 opts.clock_rate = prop.clockRate * pow(1024 / 1000, 3) * 1000;
	 printf("Clock rate = %e\n", opts.clock_rate);


	 CUDA_CHECK(cudaMallocManaged(&parts_ptr, sizeof(particle) * N3));
	 size_t arena_size = (8 + TREESPACE) * sizeof(float) * N3;
	 printf("Allocating arena of %li Mbytes\n", (arena_size / 1024 / 1024));
	 CUDA_CHECK(cudaMallocManaged(&arena, arena_size));
	 if (arena == nullptr) {
	 printf("Not enough memory\n");
	 abort();
	 }
	 main_kernel<<<1, BLOCK_SIZE>>>(arena, parts_ptr, opts, host_ewald);
	 CUDA_CHECK(cudaGetLastError());

	 CUDA_CHECK(cudaDeviceSynchronize());
	 CUDA_CHECK(cudaFree(arena));
	 CUDA_CHECK(cudaFree(parts_ptr));
	 CUDA_CHECK(cudaFree(etable));
	 CUDA_CHECK(cudaFree(host_ewald));
	 */
}
