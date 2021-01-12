/*
 * vector.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_VECTOR_HPP_
#define GPUTIGER_VECTOR_HPP_

#include <gputiger/params.hpp>
#include <cassert>

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

template<class T>
class vector {
	T *A;
	size_t sz;
public:
	__device__
	void resize(size_t new_size) {
		T* new_ptr;
		CUDA_CHECK(cudaMalloc(&new_ptr, sizeof(T) * new_size));
		if( A ) {
			for (int i = 0; i < min(new_size, sz); i++) {
				new_ptr[i] = A[i];
			}
			cudaFree(A);
		}
		A = new_ptr;
		sz = new_size;
	}
	__device__ vector() {
		sz = 0;
		A = nullptr;
	}
	__device__ vector(int size_) {
		sz = size_;
		CUDA_CHECK(cudaMalloc(&A, sizeof(T) * sz));
	}
	__device__ size_t size() const {
		return sz;
	}
	__device__ T& operator[](int i) {
		assert(i >= 0);
		assert(i < sz);
		return A[i];
	}
	__device__ T operator[](int i) const {
		assert(i >= 0);
		assert(i < sz);
		return A[i];
	}
	__device__ ~vector() {
		if (A) {
			cudaFree(A);
		}
	}
	__device__ vector<T>& operator=(const vector<T> &other) {
		resize(other.size());
		for (int i = 0; i < other.size(); i++) {
			A[i] = other.A[i];
		}
		return *this;
	}
	__device__ vector<T>& operator=(vector<T> &&other) {
		if (A) {
			delete[] A;
		}
		A = other.A;
		sz = other.sz;
		other.A = nullptr;
		other.sz = 0;
		return *this;
	}
	__device__ vector<T>(const vector<T> &other) {
		*this = other;
	}
	__device__ vector<T>(vector<T> &&other) {
		*this = std::move(other);
	}
	__device__ T* data() {
		return A;
	}
};

template<class T>
class vector3d {
	vector<T> A;
	int nx, ny, nz;
	__device__
	int index(int i, int j, int k) const {
		return nx * (ny * i + j) + k;
	}
public:
	__device__ vector3d() {
		nx = ny = nz = 0;
	}
	__device__ vector3d(int x, int y, int z) {
		nx = x;
		ny = y;
		nz = z;
		A.resize(nx * ny * nz);
	}
	__device__ T* data() {
		return A.data();
	}
	__device__ T& operator()(int i, int j, int k) {
		return A[index(i, j, k)];
	}
	__device__ T operator()(int i, int j, int k) const {
		return A[index(i, j, k)];
	}
};

#endif /* GPUTIGER_VECT++OR_HPP_ */
