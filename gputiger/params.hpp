/*
 * params.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_PARAMS_HPP_
#define GPUTIGER_PARAMS_HPP_

#include <nvfunctional>
#include <gputiger/interp.hpp>

using boltz_real = float;

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
	int Ngrid;
	double box_size;
};


struct zero_order_universe {
	cosmic_parameters params;
	double amin;
	double amax;
	nvstd::function<boltz_real(boltz_real)> hubble;
	interp_functor<boltz_real> sigma_T;
	interp_functor<boltz_real> cs2;
};




#endif /* GPUTIGER_PARAMS_HPP_ */
