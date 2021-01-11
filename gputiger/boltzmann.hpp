/*
 * boltzmann.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_BOLTZMANN_HPP_
#define GPUTIGER_BOLTZMANN_HPP_

#include <gputiger/vector.hpp>
#include <gputiger/zero_order.hpp>
#include <nvfunctional>

#define LMAX 32

#define hdoti 0
#define deltaci 1
#define deltabi 2
#define thetabi 3
#define FLi 4
#define GLi (4+LMAX)
#define NLi (4+2*LMAX)

#define deltagami (FLi+0)
#define thetagami (FLi+1)
#define F2i (FLi+2)
#define deltanui (NLi+0)
#define thetanui (NLi+1)
#define N2i (NLi+2)
#define G0i (GLi+0)
#define G1i (GLi+1)
#define G2i (GLi+2)

#define NFIELD (4+(3*LMAX))

using cos_state = array<boltz_real,NFIELD>;

#include <gputiger/zero_order.hpp>

__device__
static void einstein_boltzmann(boltz_real* value,
		const zero_order_universe *uni_ptr, boltz_real k) {
	const auto &uni = *uni_ptr;
	cos_state U;
	cos_state U0;
	using real = boltz_real;
	const nvstd::function<boltz_real(boltz_real)>& Hubble = uni.hubble;
	real omega_gam = uni.params.omega_gam;
	real omega_nu = uni.params.omega_nu;
	real omega_b = uni.params.omega_b;
	real omega_c = uni.params.omega_c;
	real omega_m = omega_b + omega_c;
	real omega_r = omega_gam + omega_nu;
	real amin = uni.amin;
	real amax = uni.amax;
	real loga = log(amin);
	real logamax = log(amax);
	real logamin = loga;
	real eps = k / (amin * Hubble(amin));
	real C = (real) 1.0 * pow(eps, (real) -1.5) * 1e-8;
	real a = exp(loga);
	real hubble = Hubble(a);
	real Or = omega_r
			/ (omega_r + a * omega_m
					+ (a * a * a * a) * ((real) 1.0 - omega_m - omega_r));
	real Om = omega_m
			/ (omega_r / a + omega_m
					+ (a * a * a) * ((real) 1.0 - omega_m - omega_r));
	real Ogam = omega_gam * Or / omega_r;
	real Onu = omega_nu * Or / omega_r;
	real Ob = omega_b * Om / omega_m;
	real Oc = omega_c * Om / omega_m;
	real tau = (real) 1.0 / (amin * hubble);
	real Rnu = Onu / Or;
	real eta = (real) 2.0 * C
			- C * ((real) 5 + (real) 4 * Rnu)
					/ ((real) 6 * ((real) 15 + (real) 4 * Rnu)) * eps * eps;
	U[deltanui] = U[deltagami] = -(real) 2.0 / (real) 3.0 * C * eps * eps;
	U[deltabi] = U[deltaci] = (real) 3.0 / (real) 4.0 * U[deltagami];
	U[thetabi] = U[thetagami] = -C / (real) 18.0 * eps * eps * eps;
	U[thetanui] = ((real) 23 + (real) 4 * Rnu) / ((real) 15 + (real) 4 * Rnu)
			* U[thetagami];
	U[N2i] = (real) 0.5 * ((real) 4.0 * C)
			/ ((real) 3.0 * ((real) 15 + (real) 4 * Rnu)) * eps * eps;
	U[hdoti] = (real) (real) 2.0 * C * eps * eps;
	U[G0i] = U[G1i] = U[G2i] = U[F2i] = (real) 0.0;
	for (int l = 3; l < LMAX; l++) {
		U[FLi + l] = (real) 0.0;
		U[NLi + l] = (real) 0.0;
		U[GLi + l] = (real) 0.0;
	}
	eta = ((real) 0.5 * U[hdoti]
			- ((real) 1.5 * (Oc * U[deltaci] + Ob * U[deltabi])
					+ (real) 1.5 * (Ogam * U[deltagami] + Onu * U[deltanui])))
			/ (eps * eps);
	while (loga < logamax) {
		real a = exp(loga);
		real hubble = Hubble(a);
		real eps = k / (a * hubble);
		Or = omega_r
				/ (omega_r + a * omega_m
						+ (a * a * a * a) * ((real) 1.0 - omega_m - omega_r));
		Om = omega_m
				/ (omega_r / a + omega_m
						+ (a * a * a) * ((real) 1.0 - omega_m - omega_r));
		Ogam = omega_gam * Or / omega_r;
		Onu = omega_nu * Or / omega_r;
		Ob = omega_b * Om / omega_m;
		Oc = omega_c * Om / omega_m;
		real cs2 = uni.cs2(a);
		real lambda_i = 0.0;
		lambda_i = max(lambda_i,
				sqrt(((real) LMAX + (real) 1.0) / ((real) LMAX + (real) 3.0))
						* eps);
		lambda_i = max(lambda_i,
				sqrt((real) 3.0 * pow(eps, 4) + (real) 8.0 * eps * eps * Or)
						/ sqrt((real) 5) / eps);
		real lambda_r = (eps
				+ sqrt(eps * eps + (real) 4.0 * cs2 * pow(eps, (real) 4)))
				/ ((real) 2.0 * eps);
		real dloga_i = (real) 2.0 * (real) 1.73 / lambda_i;
		real dloga_r = (real) 2.0 * (real) 2.51 / lambda_r;
		real dloga = min(
				min((real) 1e-3,
						min((real) 0.9 * dloga_i, (real) 0.9 * dloga_r)),
				logamax - loga);
		real loga0 = loga;

		const auto compute_explicit =
				[&](int step) {
					U0 = U;
					cos_state dudt;
					constexpr real beta[3] = {1, 0.25, (2.0 / 3.0)};
					constexpr real tm[3] = {0, 1, 0.5};
					real tau0 = tau;
					real eta0 = eta;
					for (int i = 0; i < 3; i++) {
						loga = loga0 + (real) 0.5 * (tm[i] + step) * dloga;
						a = exp(loga);
						hubble = Hubble(a);
						eps = k / (a * hubble);
						Or = omega_r / (omega_r + a * omega_m + (a * a * a * a) * ((real) 1.0 - omega_m - omega_r));
						Om = omega_m / (omega_r / a + omega_m + (a * a * a) * ((real) 1.0 - omega_m - omega_r));
						Ogam = omega_gam * Or / omega_r;
						Onu = omega_nu * Or / omega_r;
						Ob = omega_b * Om / omega_m;
						Oc = omega_c * Om / omega_m;
						cs2 = uni.cs2(a);
						real dtau = (real) 1.0 / (a * hubble);
						real etadot = ((real) 1.5 * ((Ob * U[thetabi]) + ((real) 4.0 / (real) 3.0) * (Ogam * U[thetagami] + Onu * U[thetanui])) / eps);
						real factor = ((a * omega_m) + (real) 4 * a * a * a * a * ((real) 1 - omega_m - omega_r))
						/ ((real) 2 * a * omega_m + (real) 2 * omega_r + (real) 2 * a * a * a * a * ((real) 1 - omega_m - omega_r));
						dudt[hdoti] =
						(-factor * U[hdoti] - ((real) 3.0 * (Oc * U[deltaci] + Ob * U[deltabi]) + (real) 6.0 * (Ogam * U[deltagami] + Onu * U[deltanui])));
						dudt[deltaci] = -(real) 0.5 * U[hdoti];
						dudt[deltabi] = -eps * U[thetabi] - (real) 0.5 * U[hdoti];
						dudt[deltagami] = -(real) 4.0 / (real) 3.0 * eps * U[thetagami] - ((real) 2.0 / (real) 3.0) * U[hdoti];
						dudt[deltanui] = -(real) 4.0 / (real) 3.0 * eps * U[thetanui] - ((real) 2.0 / (real) 3.0) * U[hdoti];
						dudt[thetabi] = -U[thetabi] + cs2 * eps * U[deltabi];
						dudt[thetagami] = eps * ((real) 0.25 * U[deltagami] - (real) 0.5 * U[F2i]);
						dudt[thetanui] = eps * ((real) 0.25 * U[deltanui] - (real) 0.5 * U[N2i]);
						dudt[F2i] = ((real) 8.0 / (real) 15.0) * eps * U[thetagami] + ((real) 4.0 / (real) 15.0) * U[hdoti] + ((real) 8.0 / (real) 5.0) * etadot
						- ((real) 3.0 / (real) 5.0) * eps * U[FLi + 3];
						dudt[N2i] = ((real) 8.0 / (real) 15.0) * eps * U[thetanui] + ((real) 4.0 / (real) 15.0) * U[hdoti] + ((real) 8.0 / (real) 5.0) * etadot
						- ((real) 3.0 / (real) 5.0) * eps * U[NLi + 3];
						dudt[GLi + 0] = -eps * U[GLi + 1];
						dudt[GLi + 1] = eps / (real) (3) * (U[GLi + 0] - (real) 2 * U[GLi + 2]);
						dudt[GLi + 2] = eps / (real) (5) * ((real) 2 * U[GLi + 1] - (real) 3 * U[GLi + 3]);
						for (int l = 3; l < LMAX - 1; l++) {
							dudt[FLi + l] = eps / (real) (2 * l + 1) * ((real) l * U[FLi - 1 + l] - (real) (l + 1) * U[FLi + 1 + l]);
							dudt[NLi + l] = eps / (real) (2 * l + 1) * ((real) l * U[NLi - 1 + l] - (real) (l + 1) * U[NLi + 1 + l]);
							dudt[GLi + l] = eps / (real) (2 * l + 1) * ((real) l * U[GLi - 1 + l] - (real) (l + 1) * U[GLi + 1 + l]);
						}
						dudt[FLi + LMAX - 1] = (eps * U[FLi + LMAX - 2]) / (real) (2 * LMAX - 1);
						dudt[NLi + LMAX - 1] = (eps * U[NLi + LMAX - 2]) / (real) (2 * LMAX - 1);
						dudt[GLi + LMAX - 1] = (eps * U[GLi + LMAX - 2]) / (real) (2 * LMAX - 1);
						for (int f = 0; f < NFIELD; f++) {
							U[f] = ((real) 1 - beta[i]) * U0[f] + beta[i] * (U[f] + dudt[f] * dloga * (real) 0.5);
						}
						tau = ((real) 1 - beta[i]) * tau0 + beta[i] * (tau + dtau * dloga * (real) 0.5);
						eta = ((real) 1 - beta[i]) * eta0 + beta[i] * (eta + etadot * dloga * (real) 0.5);
					}
				};

		auto compute_implicit_dudt =
				[&](real loga, real dloga) {
					a = exp(loga);
					real thetab = U[thetabi];
					real thetagam = U[thetagami];
					real F2 = U[F2i];
					real G0 = U[G0i];
					real G1 = U[G1i];
					real G2 = U[G2i];
					real thetab0 = thetab;
					real thetagam0 = thetagam;
					real F20 = F2;
					real G00 = G0;
					real G10 = G1;
					real G20 = G2;
					real sigma = uni.sigma_T(a);

					thetab = -((-(real) 3 * Ob * thetab0 - (real) 3 * dloga * Ob * sigma * thetab0 - (real) 4 * dloga * Ogam * sigma * thetagam0)
							/ ((real) 3 * Ob + (real) 3 * dloga * Ob * sigma + (real) 4 * dloga * Ogam * sigma));
					thetagam = -((-(real) 3 * dloga * Ob * sigma * thetab0 - (real) 3 * Ob * thetagam0 - (real) 4 * dloga * Ogam * sigma * thetagam0)
							/ ((real) 3 * Ob + (real) 3 * dloga * (real) Ob * sigma + (real) 4 * dloga * Ogam * sigma));
					F2 = -((-(real) 10 * F20 - (real) 4 * dloga * F20 * sigma - dloga * G00 * sigma - dloga * G20 * sigma)
							/ (((real) 1 + dloga * sigma) * ((real) 10 + (real) 3 * dloga * sigma)));
					G0 = -((-(real) 10 * G00 - (real) 5 * dloga * F20 * sigma - (real) 8 * dloga * G00 * sigma - (real) 5 * dloga * G20 * sigma)
							/ (((real) 1 + dloga * sigma) * ((real) 10 + (real) 3 * dloga * sigma)));
					G1 = G10 / ((real) 1 + dloga * sigma);
					G2 = -((-(real) 10 * G20 - dloga * F20 * sigma - dloga * G00 * sigma - (real) 4 * dloga * G20 * sigma)
							/ (((real) 1 + dloga * sigma) * ((real) 10 + (real) 3 * dloga * sigma)));
					array<real, NFIELD> dudt;
					for (int f = 0; f < NFIELD; f++) {
						dudt[f] = (real) 0.0;
					}
					dudt[thetabi] = (thetab - thetab0) / dloga;
					dudt[thetagami] = (thetagam - thetagam0) / dloga;
					dudt[F2i] = (F2 - F20) / dloga;
					dudt[G0i] = (G0 - G00) / dloga;
					dudt[G1i] = (G1 - G10) / dloga;
					dudt[G2i] = (G2 - G20) / dloga;
					for (int l = 3; l < LMAX - 1; l++) {
						dudt[GLi + l] = U[GLi + l] * ((real) 1 / ((real) 1 + dloga * sigma) - (real) 1) / dloga;
						dudt[FLi + l] = U[FLi + l] * ((real) 1 / ((real) 1 + dloga * sigma) - (real) 1) / dloga;
					}
					dudt[GLi + LMAX - 1] = U[GLi + LMAX - 1]
					* ((real) 1 / ((real) 1 + (sigma + (real) LMAX / (tau * a * hubble) / ((real) 2 * (real) LMAX - (real) 1))) - (real) 1) / dloga;
					dudt[FLi + LMAX - 1] = U[FLi + LMAX - 1]
					* ((real) 1 / ((real) 1 + (sigma + (real) LMAX / (tau * a * hubble) / ((real) 2 * (real) LMAX - (real) 1))) - (real) 1) / dloga;
					return dudt;
				};

		compute_explicit(0);
		real gamma = (real) 1.0 - (real) 1.0 / sqrt((real) 2);

		auto dudt1 = compute_implicit_dudt(loga + gamma * dloga, gamma * dloga);
		for (int f = 0; f < NFIELD; f++) {
			U[f] += dudt1[f] * ((real) 1.0 - (real) 2.0 * gamma) * dloga;
		}
		auto dudt2 = compute_implicit_dudt(loga + ((real) 1.0 - gamma) * dloga,
				gamma * dloga);
		for (int f = 0; f < NFIELD; f++) {
			U[f] += (dudt1[f] * ((real) -0.5 + (real) 2.0 * gamma)
					+ dudt2[f] * (real) 0.5) * dloga;
		}

		compute_explicit(1);

		loga = loga0 + dloga;
	}
	const boltz_real power = pow(Oc * U[deltaci] + Ob * U[deltabi],2);
	*value = power;	printf( "Done with k = %e %e\n", k, power);
}

struct sigma8_integrand {
	const zero_order_universe* uni;
	__device__ boltz_real operator()(boltz_real x) const {
		constexpr boltz_real R = 8;
		constexpr boltz_real c0 = boltz_real(3)
				/ (boltz_real(2) * boltz_real(M_PI) * R * R * R);
		;
		boltz_real P1;
		boltz_real y = exp(x) * R;
		einstein_boltzmann(&P1, uni, y / R);
		return c0 * P1 * pow((sin(y) - y * R * cos(y)),2);
	}
};

#endif /* GPUTIGER_BOLTZMANN_HPP_ */
