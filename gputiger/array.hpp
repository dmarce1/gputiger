/*
 * array.hpp
 *
 *  Created on: Jan 14, 2021
 *      Author: dmarce1
 */

#ifndef ARRAY_HPP_
#define ARRAY_HPP_


template<class T, int N>
class array {
	T A[N];
public:
	__device__ T& operator[](int i) {
		assert(i >= 0);
		assert(i < N);
		return A[i];
	}
	__device__ T operator[](int i) const {
		assert(i >= 0);
		assert(i < N);
		return A[i];
	}
	__device__ array<T, N>& operator=(const array<T, N> &other) {
		for (int i = 0; i < N; i++) {
			A[i] = other[i];
		}
		return *this;
	}
	__device__ array<T, N>(const array<T, N> &other) {
		*this = other;
	}
	__device__ array<T, N>& operator=(array<T, N> &&other) {
		for (int i = 0; i < N; i++) {
			A[i] = other[i];
		}
		return *this;
	}
	__device__ array<T, N>(array<T, N> &&other) {
		*this = other;
	}
	__device__ constexpr array<T, N>() {
	}
};



#endif /* ARRAY_HPP_ */
