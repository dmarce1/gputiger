/*
 * util.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_UTIL_HPP_
#define GPUTIGER_UTIL_HPP_

#include <cstdio>

__device__ static void print_time(double tm) {
	const double speryear = 365.24 * 24 * 3600.0;
	if (tm < 60.0) {
		printf("%.1f seconds", tm);
	} else if (tm < 3600.0) {
		printf("%.1f minutes", tm / 60);
	} else if (tm < 24 * 3600.0) {
		printf("%.1f hours", tm / 3600);
	} else if (tm < 365.24 * 24 * 3600.0) {
		printf("%.1f days", tm / 3600 / 24);
	} else if (tm < speryear * 1e3) {
		printf("%.1f years", tm / speryear / 1e0);
	} else if (tm < speryear * 1e6) {
		printf("%.1f thousand years", tm / speryear / 1e3);
	} else if (tm < speryear * 1e9) {
		printf("%.1f million years", tm / speryear / 1e6);
	} else {
		printf("%.1f billion years", tm / speryear / 1e9);
	}
}




#endif /* GPUTIGER_UTIL_HPP_ */
