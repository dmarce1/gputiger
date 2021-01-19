/*
 * particle.hpp
 *
 *  Created on: Jan 14, 2021
 *      Author: dmarce1
 */

#ifndef PARTICLE_HPP_
#define PARTICLE_HPP_

#include <cstdint>
#include <gputiger/params.hpp>
#include <gputiger/array.hpp>

__device__ inline float ewald_distance(float x) {
	x = abs(x);
	x = min(x, 1.f - x);
	return x;
}
struct particle {
	array<float, NDIM> x;
	array<float, NDIM> v;
	int8_t rung;
};

#endif /* PARTICLE_HPP_ */
