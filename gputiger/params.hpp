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

#define POW(a,b) powf(a,b)
#define LOG(a) logf(a)
#define EXP(a) expf(a)
#define SQRT(a) sqrtf(a)
#define COS(a) cosf(a)
#define SIN(a) sinf(a)
#define SINCOS(a,b,c) sincosf(a,b,c)


#define CUDA_CHECK( a ) if( a != cudaSuccess ) printf( "CUDA error on line %i of %s : %s\n", __LINE__, __FILE__, cudaGetErrorString(a))


struct cosmic_parameters {
	double h;
	double Y;
	double Theta;
	double omega_b;
	double omega_nu;
	double omega_gam;
	double omega_c;
	double Neff;
	double sigma8;
	int Ngrid;
	double box_size;
};



#endif /* GPUTIGER_PARAMS_HPP_ */
