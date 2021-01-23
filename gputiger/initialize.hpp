/*
 * initialize.hpp
 *
 *  Created on: Jan 22, 2021
 *      Author: dmarce1
 */

#ifndef INITIALIZE_HPP_
#define INITIALIZE_HPP_

#include <gputiger/particle.hpp>

__global__
void initialize(void* arena, particle* host_parts, options opts_, cudaTextureObject_t* ewald_ptr, float* matterpow, float*);




#endif /* INITIALIZE_HPP_ */
