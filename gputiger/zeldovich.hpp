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

__device__ float zeldovich_overdensity(cmplx* den, const cmplx* basis, const cmplx* rands,
		const interp_functor<float>& P, float box_size, int N);

__device__ float zeldovich_displacements(cmplx* phi, const cmplx* basis, const cmplx* rands,
		const interp_functor<float>& P, float box_size, int N, int dim);

__device__ float zeldovich_velocities(cmplx* vel_k, const cmplx* basis, const cmplx* rands,
		const interp_functor<float>& P, float box_size, int N, int dim);
#endif /* ZELDOVICH_HPP_ */
