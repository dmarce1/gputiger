/*
 * zeldovich.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#ifndef ZELDOVICH_HPP_
#define ZELDOVICH_HPP_


#include <gputiger/params.hpp>
#include <gputiger/vector.hpp>
#include <gputiger/interp.hpp>
#include <gputiger/math.hpp>
#include <gputiger/fourier.hpp>

enum zeldovich_t{ DENSITY, DISPLACEMENT, VELOCITY };

#define ZELDOSIZE 1024

__global__ void zeldovich(cmplx* den, const cmplx* basis, const cmplx* rands,
		const interp_functor<float>& P, float box_size, int N, int dim, zeldovich_t, float* res );

#endif /* ZELDOVICH_HPP_ */
