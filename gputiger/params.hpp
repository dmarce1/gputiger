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


#define TREESPACE 4
#define NDIM 3
#define NCHILD (2)
#define MAXDEPTH 30
#define MAXTHREADCOUNT 512
#define WARPSIZE 32

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))


struct options {
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
