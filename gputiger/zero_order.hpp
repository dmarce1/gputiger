/*
 * zero_order.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */
#pragma once

#include <gputiger/chemistry.hpp>
#include <gputiger/util.hpp>
#include <gputiger/params.hpp>
#include <gputiger/interp.hpp>

struct zero_order_universe {
	double amin;
	double amax;
	nvstd::function<float(float)> hubble;
	interp_functor<float> sigma_T;
	interp_functor<float> cs2;
	__device__
	void compute_matter_fractions(float& Oc, float& Ob, float a) const;
	__device__
	void compute_radiation_fractions(float& Ogam, float& Onu, float a) const;
	__device__
	float conformal_time_to_scale_factor(float taumax);
	__device__
	float scale_factor_to_conformal_time(float a);
	__device__
	float redshift_to_time(float z) const;
};

__device__
void create_zero_order_universe(zero_order_universe* uni_ptr, double amax);

