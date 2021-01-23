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

#define DRIFTSIZE 32
#define TREESORTSIZE 32

int main() {
	timer time;
	timer kick_time;
	timer sort_time;
	size_t stack_size = STACK_SIZE;
	size_t recur_limit = KERNEL_DEPTH;
	size_t heap_size = HEAP_SIZE;
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
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
	opts.Nmp = 100;
	opts.redshift = 75.0;
	opts.h = 0.7;
	opts.Neff = 3.046;
	opts.Y = 0.24;
	opts.omega_b = 0.05;
	opts.omega_c = 0.25;
	opts.Theta = 1.0;
	opts.Ngrid = 256;
	opts.sigma8 = 0.85;
	opts.max_overden = 1.0;
	opts.box_size = 613.0 * opts.Ngrid / 2160.0;
	//	opts.box_size = 613.0 / 2160.0 * opts.Ngrid;
	opts.nout = 64;
	opts.max_kernel_depth = KERNEL_DEPTH - 1;
	opts.parts_per_bucket = 64;
	opts.opening_crit = 0.7;
	opts.particle_mass = 1.0;
	opts.nparts = opts.Ngrid * opts.Ngrid * opts.Ngrid;
	opts.hsoft = 1.0f / (opts.Ngrid * 50.0);
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
	float* matter_power;
	float* vel_power;
	const int N = opts.Ngrid;
	const int N3 = N * N * N;
	size_t arena_size = (8 + TREESPACE) * sizeof(float) * N3;
	CUDA_MALLOC_MANAGED(&parts_ptr, sizeof(particle) * N3);
	CUDA_MALLOC_MANAGED(&arena, arena_size);
	CUDA_MALLOC_MANAGED(&matter_power, sizeof(opts.Nmp));
	CUDA_MALLOC_MANAGED(&vel_power, sizeof(opts.Nmp));
	initialize<<<1, 1>>>(arena, parts_ptr, opts, host_ewald, matter_power, vel_power);
	CUDA_CHECK(cudaDeviceSynchronize());
	double kmin = 1e-4;
	double kmax = 5;
	double dlogk = logf(kmax / kmin) / (100 - 1);
	FILE* fp = fopen("power.dat", "wt");
	if (fp == nullptr) {
		printf("Unable to open power.dat for writing\n");
		return -2;
	}
	for (int i = 0; i < 100; i++) {
		double k = expf(logf(1e-4 * opts.h) + dlogk * i);
		fprintf(fp, "%10.4e %10.4e %10.4e\n", k, matter_power[i], vel_power[i]);
	}
	fclose(fp);

	int* parts_processed;
	double a = 1.f / (opts.redshift + 1.f);
	int* leaf_count;
	int rung = 0;
	int* maxrung;
	CUDA_MALLOC_MANAGED(&parts_processed, sizeof(int));
	CUDA_MALLOC_MANAGED(&leaf_count, sizeof(int));
	CUDA_MALLOC_MANAGED(&maxrung, sizeof(int));
	return 1;
	do {

		sort_time.start();
		printf("\tSorting\n");
		root_tree_sort<<<1,TREESORTSIZE>>>(arena + 8*sizeof(float)*N3, TREESPACE*sizeof(float)*N3, parts_ptr, (particle*) arena, leaf_count);
		CUDA_CHECK(cudaDeviceSynchronize());
		printf("\t\tSort took %e seconds\n", sort_time.stop());

		printf("\t\tKicking\n");
		kick_time.start();
		int blocks_needed = (*leaf_count - 1) + 1;
		int block_size = sqrtf(double(blocks_needed - 1)) + 1;
		dim3 dim;
		dim.x = dim.y = block_size;
		dim.z = 1;
		*parts_processed = 0.0;
		*maxrung = 0;
		tree_kick<<<dim,KICKWARPSIZE>>>(rung,dtau, a,parts_processed,maxrung);
		CUDA_CHECK(cudaDeviceSynchronize());
		printf("\t\tKick took %e seconds\n", kick_time.stop());
		printf("\tScience Rate = %e pps\n", *parts_processed / (kick_time.result() + sort_time.result()));

		time.start();
		printf("\t\tDrifting\n");
		int nblocks = (N3 - 1) / (opts.parts_per_bucket * DRIFTSIZE) + 1;
		tree_drift<<<nblocks,DRIFTSIZE>>>((particle*)arena,1.0/C.scale(),dtau);
		CUDA_CHECK(cudaDeviceSynchronize());
		printf("\t\tDrift took %e seconds\n", time.stop());

	} while (1);
	return 0;
}
