/*
 * stack.hpp
 *
 *  Created on: Jan 14, 2021
 *      Author: dmarce1
 */

#ifndef STACK_HPP_
#define STACK_HPP_

#include <gputiger/vector.hpp>

template<class T>
class stack {
	vector<T> data;
public:
	__device__
	void push( T item ) {
		data.push_back(item);
	}
	__device__
	void pop() {
		data.resize(data.size()-1);
	}
	__device__
	size_t size() const {
		return data.size();
	}
	__device__
	T& top() {
		return data.back();
	}
	__device__
	const T& top() const {
		return data.top();
	}
};


#endif /* STACK_HPP_ */
