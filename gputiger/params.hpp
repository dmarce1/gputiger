/*
 * params.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_PARAMS_HPP_
#define GPUTIGER_PARAMS_HPP_

#include <nvfunctional>
#include <cstdio>


#define TREESPACE 3
#define NDIM 3
#define NCHILD 2
#define MAXDEPTH 30
#define MAXTHREADCOUNT 512
#define WARPSIZE 32

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))

#define CHECK_PTR( a ) if( a == nullptr) printf( "Not able to allocate memory near line %i in file %s\n", __LINE__, __FILE__);

#define CUDA_MALLOC( a, b ) \
{ \
	auto rc = cudaMalloc(a,b); \
	if( rc != cudaSuccess ) { \
		printf( "cudaMalloc failed on %i in %s with \"%s\"\n", __LINE__, __FILE__, cudaGetErrorString(rc)); \
		__trap(); \
	} else { \
		if( a == nullptr ) { \
			printf( "Failed to allocate memory on %i in %s\n", __LINE__, __FILE__); \
			__trap(); \
		} \
	} \
}

#ifdef __CUDA_ARCH__

#define CUDA_MALLOC_MANAGED( a, b ) \
{ \
	auto rc = cudaMallocManaged(a,b); \
	if( rc != cudaSuccess ) { \
		printf( "cudaMalloc failed on %i in %s with \"%s\"\n", __LINE__, __FILE__, cudaGetErrorString(rc)); \
		__trap(); \
	} else { \
		if( a == nullptr ) { \
			printf( "Failed to allocate memory on %i in %s\n", __LINE__, __FILE__); \
			__trap(); \
		} \
	} \
}
#else

#define CUDA_MALLOC_MANAGED( a, b ) \
{ \
	auto rc = cudaMallocManaged(a,b); \
	if( rc != cudaSuccess ) { \
		printf( "cudaMalloc failed on %i in %s with \"%s\"\n", __LINE__, __FILE__, cudaGetErrorString(rc)); \
		abort(); \
	} else { \
		if( a == nullptr ) { \
			printf( "Failed to allocate memory on %i in %s\n", __LINE__, __FILE__); \
			abort(); \
		} \
	} \
}
#endif


struct options {
	float Nmp;
	double code_to_cm;
	double code_to_g;
	double code_to_s;
	float redshift;
	float particle_mass;
	float hsoft;
	float max_overden;
	float h;
	float Y;
	float opening_crit;
	float Theta;
	float omega_b;
	float omega_nu;
	float omega_gam;
	float omega_c;
	float Neff;
	float sigma8;
	float clock_rate;
	int nout;
	int Ngrid;
	int nparts;
	float box_size;
	float parts_per_bucket;
	int max_kernel_depth;
};

#ifndef PARAMSCU
__device__ extern options opts;
#endif


#endif /* GPUTIGER_PARAMS_HPP_ */
