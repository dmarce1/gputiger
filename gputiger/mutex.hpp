/*
 * mutex.hpp
 *
 *  Created on: Jan 14, 2021
 *      Author: dmarce1
 */

#ifndef MUTEX_HPP_
#define MUTEX_HPP_

class mutex {
	int bit;
public:
	__device__
	constexpr mutex() :
			bit(0) {
	}
	__device__ mutex(const mutex&) = delete;
	__device__ mutex(mutex&&) = delete;
	__device__ mutex& operator=(const mutex&) = delete;
	__device__ mutex& operator=(mutex&&) = delete;
	__device__
	void lock() {
		while (atomicCAS(&bit, 0, 1) != 0)
			;
	}
	__device__
	bool try_lock() {
		return atomicCAS(&bit, 0, 1) == 0;
	}
	__device__
	void unlock() {
		bit = 0;
	}
};

template<class T>
class lock_guard {
	T& mtx;
public:
	__device__ lock_guard(T& mtx_) :
			mtx(mtx_) {
		mtx.lock();
	}
	__device__ ~lock_guard() {
		mtx.unlock();
	}
};

#endif /* MUTEX_HPP_ */
