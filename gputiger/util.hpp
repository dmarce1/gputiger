/*
 * util.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_UTIL_HPP_
#define GPUTIGER_UTIL_HPP_

#include <cstdio>
#include <ctime>

__device__  void print_time(double tm);


class timer {
	double tm;
public:
	void start() {
		tm = clock();
	}
	double stop() {
		tm = (clock() - tm) / CLOCKS_PER_SEC;
		return tm;
	}
	double result() {
		return tm;
	}

};

#endif /* GPUTIGER_UTIL_HPP_ */
