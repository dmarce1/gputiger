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

template<class T>
class vector {
	static constexpr size_t block_size = 1024 / sizeof(T);
	T *A;
	size_t sz;
	size_t capacity;
public:
	__device__
	void resize(size_t new_size) {
		if (new_size > capacity) {
			size_t new_cap = block_size;
			while (new_cap < new_size) {
				new_cap *= 2;
			}
			T* new_ptr;
			CUDA_CHECK(cudaMalloc(&new_ptr, sizeof(T) * new_cap));
			if (A) {
				for (int i = 0; i < min(new_size, sz); i++) {
					new_ptr[i] = A[i];
				}
				CUDA_CHECK(cudaFree(A));
			}
			A = new_ptr;
			capacity = new_cap;
		}
		sz = new_size;
	}
	__device__ vector( T* ptr, size_t sz_) {
		A = ptr;
		sz = 0;
		capacity = sz_;
	}
	__device__ vector() {
		sz = 0;
		capacity = 0;
		A = nullptr;
	}
	__device__ vector(int size_) {
		sz = size_;
		capacity = block_size;
		while (capacity < size_) {
			capacity *= 2;
		}
		CUDA_CHECK(cudaMalloc(&A, sizeof(T) * capacity));
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
			cudaFree(A);
		}
		A = other.A;
		sz = other.sz;
		capacity = other.capacity;
		other.A = nullptr;
		other.sz = 0;
		other.capacity = 0;
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
	__device__ T& front() {
		return A[0];
	}
	__device__ T& back() {
		return A[sz - 1];
	}
	__device__
	   const T& front() const {
		return A[0];
	}
	__device__
	   const T& back() const {
		return A[sz - 1];
	}
	__device__
	void push_back(T data) {
		resize(sz + 1);
		back() = data;
	}
};

#endif /* GPUTIGER_VECT++OR_HPP_ */
