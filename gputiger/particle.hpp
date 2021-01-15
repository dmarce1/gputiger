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

using pos_type = int32_t;

__device__ inline float pos_to_float(pos_type pos) {
	return (float) pos / ((float) (1LL << 32LL));
}

__device__ inline  pos_type float_to_pos(float flt) {
	while (flt > 0.5) {
		flt -= 1.0;
	}
	while (flt < -0.5) {
		flt += 1.0;
	}
	return (pos_type) std::floor(flt * ((float) (1LL << 32LL)));
}

struct particle {
	array<pos_type, NDIM> x;
	array<float, NDIM> v;
	int32_t rung;
	int32_t padding;
};

#endif /* PARTICLE_HPP_ */
