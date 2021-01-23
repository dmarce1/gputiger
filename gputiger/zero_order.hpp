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
	__device__
	double redshift_to_density(double z) const;
};

__device__
void create_zero_order_universe(zero_order_universe* uni_ptr, double amax);

class cosmos {
	double a;
	double t;
	double tau;
	double omega_m;
	double omega_r;
	double omega_lam;
	double H;
public:
	cosmos(double omega_c, double omega_b, double omega_gam, double omega_nu, double h_) {
		H = constants::H0 * h_;
		omega_m = omega_c + omega_b;
		omega_r = omega_gam + omega_nu;
		omega_lam = 1.0 - omega_r - omega_m;
		a = 1.0e-8;
		t = tau = 0.0;
	}
	double advance(double dtau0) {
		const double beta[3] = { 1.0, 0.25, 2.0 / 3.0 };

		double tau1 = tau + dtau0;
		while (tau < tau1) {
			double a0 = a;
			double t0 = t;
			double dtau = min(abs(1.0e-2/a / (H * sqrt(omega_m / (a * a * a) + omega_r / (a * a * a * a) + omega_lam))) * 0.01, tau1 - tau);
			for (int rk = 0; rk < 3; rk++) {
				double da = (H * sqrt(omega_m / (a * a * a) + omega_r / (a * a * a * a) + omega_lam))* a *a  * dtau;
				double dt = a * dtau;
				a = (1.0 - beta[rk]) * a0 + beta[rk] * (a + da);
				t = (1.0 - beta[rk]) * t0 + beta[rk] * (t + dt);
			}
			tau += dtau;
		}
		return t;
	}
	double scale() const {
		return a;
	}
	double time() const {
		return t;
	}

};

