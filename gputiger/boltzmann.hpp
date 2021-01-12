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
#include <gputiger/interp.hpp>
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

using cos_state = array<float,NFIELD>;

#include <gputiger/zero_order.hpp>

__device__
static float einstein_boltzmann(float* value, const zero_order_universe *uni_ptr, float k, float normalization = 1,
		float amp_cutoff = -1) {
	const auto &uni = *uni_ptr;
	cos_state U;
	cos_state U0;
	const nvstd::function<float(float)>& Hubble = uni.hubble;
	float den_amplitude;
	float omega_gam = uni.params.omega_gam;
	float omega_nu = uni.params.omega_nu;
	float omega_b = uni.params.omega_b;
	float omega_c = uni.params.omega_c;
	float omega_m = omega_b + omega_c;
	float omega_r = omega_gam + omega_nu;
	float amin = uni.amin;
	float amax = uni.amax;
	float loga = LOG(amin);
	float logamax = LOG(amax);
	float logamin = loga;
	float eps = k / (amin * Hubble(amin));
	float C = (float) 1.0 * POW(eps, (float ) -1.5) * normalization;
	float a = EXP(loga);
	float hubble = Hubble(a);
	float Or = omega_r / (omega_r + a * omega_m + (a * a * a * a) * ((float) 1.0 - omega_m - omega_r));
	float Om = omega_m / (omega_r / a + omega_m + (a * a * a) * ((float) 1.0 - omega_m - omega_r));
	float Ogam = omega_gam * Or / omega_r;
	float Onu = omega_nu * Or / omega_r;
	float Ob = omega_b * Om / omega_m;
	float Oc = omega_c * Om / omega_m;
	float tau = (float) 1.0 / (amin * hubble);
	float Rnu = Onu / Or;
	float eta = (float) 2.0 * C
			- C * ((float) 5 + (float) 4 * Rnu) / ((float) 6 * ((float) 15 + (float) 4 * Rnu)) * eps * eps;
	U[deltanui] = U[deltagami] = -(float) 2.0 / (float) 3.0 * C * eps * eps;
	U[deltabi] = U[deltaci] = (float) 3.0 / (float) 4.0 * U[deltagami];
	U[thetabi] = U[thetagami] = -C / (float) 18.0 * eps * eps * eps;
	U[thetanui] = ((float) 23 + (float) 4 * Rnu) / ((float) 15 + (float) 4 * Rnu) * U[thetagami];
	U[N2i] = (float) 0.5 * ((float) 4.0 * C) / ((float) 3.0 * ((float) 15 + (float) 4 * Rnu)) * eps * eps;
	U[hdoti] = (float) (float) 2.0 * C * eps * eps;
	U[G0i] = U[G1i] = U[G2i] = U[F2i] = (float) 0.0;
	for (int l = 3; l < LMAX; l++) {
		U[FLi + l] = (float) 0.0;
		U[NLi + l] = (float) 0.0;
		U[GLi + l] = (float) 0.0;
	}
	eta = ((float) 0.5 * U[hdoti]
			- ((float) 1.5 * (Oc * U[deltaci] + Ob * U[deltabi])
					+ (float) 1.5 * (Ogam * U[deltagami] + Onu * U[deltanui]))) / (eps * eps);
	float finish_time = amax;
	while (loga < logamax) {
		a = EXP(loga);
		float hubble = Hubble(a);
		float eps = k / (a * hubble);
		Or = omega_r / (omega_r + a * omega_m + (a * a * a * a) * ((float) 1.0 - omega_m - omega_r));
		Om = omega_m / (omega_r / a + omega_m + (a * a * a) * ((float) 1.0 - omega_m - omega_r));
		Ogam = omega_gam * Or / omega_r;
		Onu = omega_nu * Or / omega_r;
		Ob = omega_b * Om / omega_m;
		Oc = omega_c * Om / omega_m;
		float cs2 = uni.cs2(a);
		float lambda_i = 0.0;
		lambda_i = max(lambda_i,
		SQRT(
				((float) LMAX + (float) 1.0) / ((float) LMAX + (float) 3.0)) * eps);
		lambda_i = max(lambda_i,
		SQRT(
				(float) 3.0 * POW(eps, 4) + (float) 8.0 * eps * eps * Or) / SQRT((float ) 5) / eps);
		float lambda_r = (eps + SQRT(eps * eps + (float) 4.0 * cs2 * POW(eps, (float) 4))) / ((float) 2.0 * eps);
		float dloga_i = (float) 2.0 * (float) 1.73 / lambda_i;
		float dloga_r = (float) 2.0 * (float) 2.51 / lambda_r;
		float dloga = min(min((float) 5e-2, min((float) 0.9 * dloga_i, (float) 0.9 * dloga_r)), logamax - loga);
		float loga0 = loga;

		const auto compute_explicit =
				[&](int step) {
					U0 = U;
					cos_state dudt;
					constexpr float beta[3] = {1, 0.25, (2.0 / 3.0)};
					constexpr float tm[3] = {0, 1, 0.5};
					float tau0 = tau;
					float eta0 = eta;
					for (int i = 0; i < 3; i++) {
						loga = loga0 + (float) 0.5 * (tm[i] + step) * dloga;
						a = EXP(loga);
						hubble = Hubble(a);
						eps = k / (a * hubble);
						Or = omega_r / (omega_r + a * omega_m + (a * a * a * a) * ((float) 1.0 - omega_m - omega_r));
						Om = omega_m / (omega_r / a + omega_m + (a * a * a) * ((float) 1.0 - omega_m - omega_r));
						Ogam = omega_gam * Or / omega_r;
						Onu = omega_nu * Or / omega_r;
						Ob = omega_b * Om / omega_m;
						Oc = omega_c * Om / omega_m;
						cs2 = uni.cs2(a);
						float dtau = (float) 1.0 / (a * hubble);
						float etadot = ((float) 1.5 * ((Ob * U[thetabi]) + ((float) 4.0 / (float) 3.0) * (Ogam * U[thetagami] + Onu * U[thetanui])) / eps);
						float factor = ((a * omega_m) + (float) 4 * a * a * a * a * ((float) 1 - omega_m - omega_r))
						/ ((float) 2 * a * omega_m + (float) 2 * omega_r + (float) 2 * a * a * a * a * ((float) 1 - omega_m - omega_r));
						dudt[hdoti] =
						(-factor * U[hdoti] - ((float) 3.0 * (Oc * U[deltaci] + Ob * U[deltabi]) + (float) 6.0 * (Ogam * U[deltagami] + Onu * U[deltanui])));
						dudt[deltaci] = -(float) 0.5 * U[hdoti];
						dudt[deltabi] = -eps * U[thetabi] - (float) 0.5 * U[hdoti];
						dudt[deltagami] = -(float) 4.0 / (float) 3.0 * eps * U[thetagami] - ((float) 2.0 / (float) 3.0) * U[hdoti];
						dudt[deltanui] = -(float) 4.0 / (float) 3.0 * eps * U[thetanui] - ((float) 2.0 / (float) 3.0) * U[hdoti];
						dudt[thetabi] = -U[thetabi] + cs2 * eps * U[deltabi];
						dudt[thetagami] = eps * ((float) 0.25 * U[deltagami] - (float) 0.5 * U[F2i]);
						dudt[thetanui] = eps * ((float) 0.25 * U[deltanui] - (float) 0.5 * U[N2i]);
						dudt[F2i] = ((float) 8.0 / (float) 15.0) * eps * U[thetagami] + ((float) 4.0 / (float) 15.0) * U[hdoti] + ((float) 8.0 / (float) 5.0) * etadot
						- ((float) 3.0 / (float) 5.0) * eps * U[FLi + 3];
						dudt[N2i] = ((float) 8.0 / (float) 15.0) * eps * U[thetanui] + ((float) 4.0 / (float) 15.0) * U[hdoti] + ((float) 8.0 / (float) 5.0) * etadot
						- ((float) 3.0 / (float) 5.0) * eps * U[NLi + 3];
						dudt[GLi + 0] = -eps * U[GLi + 1];
						dudt[GLi + 1] = eps / (float) (3) * (U[GLi + 0] - (float) 2 * U[GLi + 2]);
						dudt[GLi + 2] = eps / (float) (5) * ((float) 2 * U[GLi + 1] - (float) 3 * U[GLi + 3]);
						for (int l = 3; l < LMAX - 1; l++) {
							dudt[FLi + l] = eps / (float) (2 * l + 1) * ((float) l * U[FLi - 1 + l] - (float) (l + 1) * U[FLi + 1 + l]);
							dudt[NLi + l] = eps / (float) (2 * l + 1) * ((float) l * U[NLi - 1 + l] - (float) (l + 1) * U[NLi + 1 + l]);
							dudt[GLi + l] = eps / (float) (2 * l + 1) * ((float) l * U[GLi - 1 + l] - (float) (l + 1) * U[GLi + 1 + l]);
						}
						dudt[FLi + LMAX - 1] = (eps * U[FLi + LMAX - 2]) / (float) (2 * LMAX - 1);
						dudt[NLi + LMAX - 1] = (eps * U[NLi + LMAX - 2]) / (float) (2 * LMAX - 1);
						dudt[GLi + LMAX - 1] = (eps * U[GLi + LMAX - 2]) / (float) (2 * LMAX - 1);
						for (int f = 0; f < NFIELD; f++) {
							U[f] = ((float) 1 - beta[i]) * U0[f] + beta[i] * (U[f] + dudt[f] * dloga * (float) 0.5);
						}
						tau = ((float) 1 - beta[i]) * tau0 + beta[i] * (tau + dtau * dloga * (float) 0.5);
						eta = ((float) 1 - beta[i]) * eta0 + beta[i] * (eta + etadot * dloga * (float) 0.5);
					}
				};

		auto compute_implicit_dudt =
				[&](float loga, float dloga) {
					a = EXP(loga);
					float thetab = U[thetabi];
					float thetagam = U[thetagami];
					float F2 = U[F2i];
					float G0 = U[G0i];
					float G1 = U[G1i];
					float G2 = U[G2i];
					float thetab0 = thetab;
					float thetagam0 = thetagam;
					float F20 = F2;
					float G00 = G0;
					float G10 = G1;
					float G20 = G2;
					float sigma = uni.sigma_T(a);

					thetab = -((-(float) 3 * Ob * thetab0 - (float) 3 * dloga * Ob * sigma * thetab0 - (float) 4 * dloga * Ogam * sigma * thetagam0)
							/ ((float) 3 * Ob + (float) 3 * dloga * Ob * sigma + (float) 4 * dloga * Ogam * sigma));
					thetagam = -((-(float) 3 * dloga * Ob * sigma * thetab0 - (float) 3 * Ob * thetagam0 - (float) 4 * dloga * Ogam * sigma * thetagam0)
							/ ((float) 3 * Ob + (float) 3 * dloga * (float) Ob * sigma + (float) 4 * dloga * Ogam * sigma));
					F2 = -((-(float) 10 * F20 - (float) 4 * dloga * F20 * sigma - dloga * G00 * sigma - dloga * G20 * sigma)
							/ (((float) 1 + dloga * sigma) * ((float) 10 + (float) 3 * dloga * sigma)));
					G0 = -((-(float) 10 * G00 - (float) 5 * dloga * F20 * sigma - (float) 8 * dloga * G00 * sigma - (float) 5 * dloga * G20 * sigma)
							/ (((float) 1 + dloga * sigma) * ((float) 10 + (float) 3 * dloga * sigma)));
					G1 = G10 / ((float) 1 + dloga * sigma);
					G2 = -((-(float) 10 * G20 - dloga * F20 * sigma - dloga * G00 * sigma - (float) 4 * dloga * G20 * sigma)
							/ (((float) 1 + dloga * sigma) * ((float) 10 + (float) 3 * dloga * sigma)));
					array<float, NFIELD> dudt;
					for (int f = 0; f < NFIELD; f++) {
						dudt[f] = (float) 0.0;
					}
					dudt[thetabi] = (thetab - thetab0) / dloga;
					dudt[thetagami] = (thetagam - thetagam0) / dloga;
					dudt[F2i] = (F2 - F20) / dloga;
					dudt[G0i] = (G0 - G00) / dloga;
					dudt[G1i] = (G1 - G10) / dloga;
					dudt[G2i] = (G2 - G20) / dloga;
					for (int l = 3; l < LMAX - 1; l++) {
						dudt[GLi + l] = U[GLi + l] * ((float) 1 / ((float) 1 + dloga * sigma) - (float) 1) / dloga;
						dudt[FLi + l] = U[FLi + l] * ((float) 1 / ((float) 1 + dloga * sigma) - (float) 1) / dloga;
					}
					dudt[GLi + LMAX - 1] = U[GLi + LMAX - 1]
					* ((float) 1 / ((float) 1 + (sigma + (float) LMAX / (tau * a * hubble) / ((float) 2 * (float) LMAX - (float) 1))) - (float) 1) / dloga;
					dudt[FLi + LMAX - 1] = U[FLi + LMAX - 1]
					* ((float) 1 / ((float) 1 + (sigma + (float) LMAX / (tau * a * hubble) / ((float) 2 * (float) LMAX - (float) 1))) - (float) 1) / dloga;
					return dudt;
				};

		compute_explicit(0);
		float gamma = (float) 1.0 - (float) 1.0 / SQRT((float ) 2);

		auto dudt1 = compute_implicit_dudt(loga + gamma * dloga, gamma * dloga);
		for (int f = 0; f < NFIELD; f++) {
			U[f] += dudt1[f] * ((float) 1.0 - (float) 2.0 * gamma) * dloga;
		}
		auto dudt2 = compute_implicit_dudt(loga + ((float) 1.0 - gamma) * dloga, gamma * dloga);
		for (int f = 0; f < NFIELD; f++) {
			U[f] += (dudt1[f] * ((float) -0.5 + (float) 2.0 * gamma) + dudt2[f] * (float) 0.5) * dloga;
		}

		compute_explicit(1);

		den_amplitude = abs(omega_c * U[deltaci] + omega_b * U[deltabi]) / omega_m;
		if (den_amplitude >= amp_cutoff && amp_cutoff > 0) {
			finish_time = a;
			break;
		}

		loga = loga0 + dloga;
	}
	*value = POW(den_amplitude,2);
	return finish_time;
}

struct sigma8_integrand {
	const zero_order_universe* uni;
	__device__ float operator()(float x) const {
		constexpr float R = 8;
		const float c0 = float(9) / (float(2) * float(M_PI) * pow(R, 6));
		float P1;
		float k = EXP(x);
		einstein_boltzmann(&P1, uni, k);
		return c0 * P1 * POW((SIN(k*R) - k * R *COS(k*R)), 2) * pow(k, -3);
	}
};

__device__
float find_nonlinear_time(const zero_order_universe* zeroverse, float kmin, float kmax, float cell_size,
		float normalization) {
	int thread = threadIdx.x;
	const int block_size = blockDim.x;
	__shared__
	float* mintime;
	float rtime;
	if (thread == 0) {
		mintime = new float[block_size];
	}
	__syncthreads();
	float logkmin = LOG(kmin);
	float dlogk = (LOG(kmax) - logkmin) / (float) (block_size - 1);
	float k = EXP(logkmin + (float ) thread * dlogk);
	float value;
	mintime[thread] = einstein_boltzmann(&value, zeroverse, k, normalization, pow(cell_size, (float) -1.5));
	__syncthreads();
	for (int M = block_size / 2; M >= 1; M /= 2) {
		if (thread < M) {
			mintime[thread] = min(mintime[thread], mintime[thread + M]);
		}
		__syncthreads();
	}
	rtime = mintime[0];
	__syncthreads();
	if (thread == 0) {
		delete[] mintime;
	}
	return rtime;
}

__device__ interp_functor<float> compute_einstein_boltzmann_interpolation_function(zero_order_universe* uni,
		float kmin, float kmax, float normalization, float time) {
	int thread = threadIdx.x;
	int block_size = blockDim.x;
	__shared__ vector<float>* pptr;
	interp_functor<float> func;
	float olda = uni->amax;
	uni->amax = time;
	float dlogk = 1.0e-2;
	float logkmin = LOG(kmin) - dlogk;
	float logkmax = LOG(kmax) + dlogk;
	int N = (logkmax - logkmin) / dlogk + 2;
	dlogk = (logkmax - logkmin) / (float) (N-1);
	if (thread == 0) {
		pptr = new vector<float>(N);
		printf("Computing power spectrum interpolation function with %i bins\n", N);
	}
	__syncthreads();
	for( int i = thread; i < N; i+= block_size) {
		float amp;
		float k = EXP(logkmin + (float) i * dlogk);
		einstein_boltzmann(&amp, uni, k, normalization);
		(*pptr)[i] = amp;
	}
	__syncthreads();
	if( thread == 0 ) {
		build_interpolation_function(&func, *pptr, EXP(logkmin), EXP(logkmax));
	}
	__syncthreads();
	if( thread == 0 ) {
		delete [] pptr;
	}
	uni->amax = olda;
	return func;
}

#endif /* GPUTIGER_BOLTZMANN_HPP_ */
