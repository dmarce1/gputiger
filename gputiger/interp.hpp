/*
 * interp.hpp
 *
 *  Created on: Jan 11, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_INTERP_HPP_
#define GPUTIGER_INTERP_HPP_

#include <gputiger/vector.hpp>

template<class T>
struct interp_functor {
	vector<T> values;
	T amin;
	T amax;
	T minloga;
	T maxloga;
	int N;
	T dloga;
	__device__
	T operator()(T a) const {
		T loga = logf(a);
		if (loga < minloga || loga > maxloga) {
			printf("Error in interpolation_function out of range %e %e %e\n", a, amin, amax);
		}
		int i1 = min(max(1, int((loga - minloga) / (dloga))), N - 3);
		int i0 = i1 - 1;
		int i2 = i1 + 1;
		int i3 = i2 + 1;
		const T c0 = values[i1];
		const T c1 = -values[i0] / (T) 3.0 - (T) 0.5 * values[i1] + values[i2] - values[i3] / (T) 6.0;
		const T c2 = (T) 0.5 * values[i0] - values[i1] + (T) 0.5 * values[i2];
		const T c3 = -values[i0] / (T) 6.0 + (T) 0.5 * values[i1] - (T) 0.5 * values[i2] + values[i3] / (T) 6.0;
		T x = (loga - i1 * dloga - minloga) / dloga;
		return c0 + c1 * x + c2 * x * x + c3 * x * x * x;
	}
};



template<class T>
__device__
void build_interpolation_function(interp_functor<T>* f, const vector<T>& values, T amin, T amax) {
	T minloga = log(amin);
	T maxloga = log(amax);
	int N = values.size() - 1;
	T dloga = (maxloga - minloga) / N;
	interp_functor<T> functor;
	functor.values = std::move(values);
	functor.maxloga = maxloga;
	functor.minloga = minloga;
	functor.dloga = dloga;
	functor.amin = amin;
	functor.amax = amax;
	functor.N = N;
	*f = functor;
}



template<class T>
__device__
void build_interpolation_function(interp_functor<T>* f, T* values, T amin, T amax, int N) {
	T minloga = log(amin);
	T maxloga = log(amax);
	T dloga = (maxloga - minloga) / N;
	interp_functor<T> functor;
	functor.values.resize(N);
	for( int i = 0; i < N; i++) {
		functor.values[i] = values[i];
	}
	functor.maxloga = maxloga;
	functor.minloga = minloga;
	functor.dloga = dloga;
	functor.amin = amin;
	functor.amax = amax;
	functor.N = N;
	*f = functor;
}

#include <gputiger/params.hpp>


#endif /* GPUTIGER_INTERP_HPP_ */
