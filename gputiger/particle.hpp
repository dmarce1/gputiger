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
#include <gputiger/vector.hpp>

__device__ float pos_to_float(int32_t pos) {
	return (float) pos / ((float) (1LL << 32LL));
}

__device__ int32_t float_to_pos(float flt) {
	while (flt > 0.5) {
		flt -= 1.0;
	}
	while (flt < -0.5) {
		flt += 1.0;
	}
	return (int32_t) std::floor(flt * ((float) (1LL << 32LL)));
}

struct particle {
	array<int32_t, NDIM> x;
	array<float, NDIM> v;
	int32_t rung;
	int32_t padding;
};

#endif /* PARTICLE_HPP_ */
