/*
 * params.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_PARAMS_HPP_
#define GPUTIGER_PARAMS_HPP_

#include <nvfunctional>

#define NDIM 3

#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))


struct cosmic_parameters {
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
};



#endif /* GPUTIGER_PARAMS_HPP_ */
