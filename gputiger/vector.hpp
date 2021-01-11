/*
 * vector.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_VECTOR_HPP_
#define GPUTIGER_VECTOR_HPP_

#include <cassert>

template<class T, int N>
class array {
	T data[N];
public:__device__
	T& operator[](int i) {
		assert(i >= 0);
		assert(i < N);
		return data[i];
	}
	__device__
	T operator[](int i) const {
		assert(i >= 0);
		assert(i < N);
		return data[i];
	}
	__device__
	array<T, N>& operator=(const array<T, N> &other) {
		for (int i = 0; i < N; i++) {
			data[i] = other[i];
		}
		return *this;
	}
	__device__ array<T, N>(const array<T, N> &other) {
		*this = other;
	}
	__device__
	array<T, N>& operator=(array<T, N> &&other) {
		for (int i = 0; i < N; i++) {
			data[i] = other[i];
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
	T *data;
	size_t sz;
public:__device__
	void resize(size_t new_size) {
		T *new_ptr = new T[new_size];
		for (int i = 0; i < min(new_size, sz); i++) {
			new_ptr[i] = data[i];
		}
		delete[] data;
		data = new_ptr;
		sz = new_size;
	}
	__device__ vector() {
		sz = 1;
		data = new T[sz];
	}
	__device__ vector(int size_) {
		sz = size_;
		data = new T[sz];
	}
	__device__
	size_t size() const {
		return sz;
	}
	__device__
	T& operator[](int i) {
		assert(i >= 0);
		assert(i < sz);
		return data[i];
	}
	__device__
	T operator[](int i) const {
		assert(i >= 0);
		assert(i < sz);
		return data[i];
	}
	__device__ ~vector() {
		delete[] data;
	}
	__device__
	vector<T>& operator=(const vector<T> &other) {
		resize(other.size());
		for (int i = 0; i < other.size(); i++) {
			data[i] = other.data[i];
		}
		return *this;
	}
	__device__
	vector<T>& operator=(vector<T> &&other) {
		delete[] data;
		data = other.data;
		sz = other.sz;
		other.data = new T[1];
		other.sz = 0;
		return *this;
	}
	__device__ vector<T>(const vector<T> &other) {
		*this = other;
	}
	__device__ vector<T>(vector<T> &&other) {
		*this = std::move(other);
	}
};

#endif /* GPUTIGER_VECT++OR_HPP_ */
