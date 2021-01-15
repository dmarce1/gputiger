/*
 * boltzmann.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_BOLTZMANN_HPP_
#define GPUTIGER_BOLTZMANN_HPP_

#include <gputiger/vector.hpp>
#include <gputiger/zero_order.hpp>
#include <gputiger/interp.hpp>
#include <nvfunctional>

#define LMAX 32

#define hdoti 0
#define etai 1
#define taui 2
#define deltaci 3
#define deltabi 4
#define thetabi 5
#define FLi 6
#define GLi (6+LMAX)
#define NLi (6+2*LMAX)

#define deltagami (FLi+0)
#define thetagami (FLi+1)
#define F2i (FLi+2)
#define deltanui (NLi+0)
#define thetanui (NLi+1)
#define N2i (NLi+2)
#define G0i (GLi+0)
#define G1i (GLi+1)
#define G2i (GLi+2)

#define NFIELD (6+(3*LMAX))

#include <gputiger/array.hpp>
#include <gputiger/zero_order.hpp>

using cos_state = array<float,NFIELD>;

__device__ void einstein_boltzmann_init(cos_state* uptr, const zero_order_universe* uni_ptr, float k,
		float normalization, float a);
__device__
void einstein_boltzmann(cos_state* uptr, const zero_order_universe *uni_ptr, float k, float amin, float amax);

struct sigma8_integrand {
	const zero_order_universe* uni;
	float littleh;
	__device__ float operator()(float x) const;
};

__device__ void einstein_boltzmann_init_set(cos_state* U, zero_order_universe* uni, float kmin, float kmax, int N,
		float amin, float normalization);

__device__ void einstein_boltzmann_interpolation_function(interp_functor<float>* den_k_func,
		interp_functor<float>* vel_k_func, cos_state* U, zero_order_universe* uni, float kmin, float kmax, int N,
		float astart, float astop);

#endif /* GPUTIGER_BOLTZMANN_HPP_ */
