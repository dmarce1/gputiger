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


#define NDIM 3
#define NCHILD (1<<NDIM)
#define MAXDEPTH 20

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))


struct options {
	float max_overden;
	float h;
	float Y;
	float Theta;
	float omega_b;
	float omega_nu;
	float omega_gam;
	float omega_c;
	float Neff;
	float sigma8;
	int nout;
	int Ngrid;
	float box_size;
	float parts_per_bucket;
	int max_kernel_depth;
};

#ifndef PARAMSCU
__device__ extern options opts;
#endif


#endif /* GPUTIGER_PARAMS_HPP_ */
