/*
 * fourier.hpp
 *
 *  Created on: Jan 12, 2021
 *      Author: dmarce1
 */

#ifndef FOURIER_HPP_
#define FOURIER_HPP_

#include <gputiger/params.hpp>
#include <gputiger/math.hpp>


__global__
void fft_basis(cmplx* X, int N);


__device__
void fft3d(cmplx* Y, const cmplx* expi, int N);


#endif /* FOURIER_HPP_ */
