#include <gputiger/zero_order.hpp>

__device__
void zero_order_universe::compute_matter_fractions(float& Oc, float& Ob, float a) const {
	float omega_m = opts.omega_b + opts.omega_c;
	float omega_r = opts.omega_gam + opts.omega_nu;
	float Om = omega_m / (omega_r / a + omega_m + (a * a * a) * ((float) 1.0 - omega_m - omega_r));
	Ob = opts.omega_b * Om / omega_m;
	Oc = opts.omega_c * Om / omega_m;
}

__device__
void zero_order_universe::compute_radiation_fractions(float& Ogam, float& Onu, float a) const {
	float omega_m = opts.omega_b + opts.omega_c;
	float omega_r = opts.omega_gam + opts.omega_nu;
	float Or = omega_r / (omega_r + a * omega_m + (a * a * a * a) * ((float) 1.0 - omega_m - omega_r));
	Ogam = opts.omega_gam * Or / omega_r;
	Onu = opts.omega_nu * Or / omega_r;
}

__device__
float zero_order_universe::conformal_time_to_scale_factor(float taumax) {
	taumax *= constants::H0 / cosmic_constants::H0;
	float dlogtau = 1.0e-3;
	float a = amin;
	float logtaumax = LOG(taumax);
	float logtaumin = LOG(1.f / (a * hubble(a)));
	int N = (logtaumax - logtaumin) / dlogtau + 1;
	dlogtau = (logtaumax - logtaumin) / N;
	for (int i = 0; i < N; i++) {
		float logtau = logtaumin + (float) i * dlogtau;
		float tau = EXP(logtau);
		float a0 = a;
		a += tau * a * a * hubble(a) * dlogtau;
		logtau = logtaumin + (float) (i + 1) * dlogtau;
		tau = EXP(logtau);
		a = 0.75f * a0 + 0.25f * (a + tau * a * a * hubble(a) * dlogtau);
		logtau = logtaumin + ((float) i + 0.5f) * dlogtau;
		tau = EXP(logtau);
		a = 1.f / 3.f * a0 + 2.f / 3.f * (a + tau * a * a * hubble(a) * dlogtau);
	}
	return a;
}

__device__
float zero_order_universe::scale_factor_to_conformal_time(float a) {
	float amax = a;
	float dloga = 1e-2;
	float logamin = LOG(amin);
	float logamax = LOG(amax);
	int N = (logamax - logamin) / dloga + 1;
	dloga = (logamax - logamin) / (float) N;
	float tau = 1.f / (amin * hubble(amin));
	for (int i = 0; i < N; i++) {
		float loga = logamin + (float) i * dloga;
		float a = EXP(loga);
		float tau0 = tau;
		tau += dloga / (a * hubble(a));
		loga = logamin + (float) (i + 1) * dloga;
		a = EXP(loga);
		tau = 0.75f * tau0 + 0.25f * (tau + dloga / (a * hubble(a)));
		loga = logamin + ((float) i + 0.5f) * dloga;
		a = EXP(loga);
		tau = (1.f / 3.f) * tau0 + (2.f / 3.f) * (tau + dloga / (a * hubble(a)));
	}
	tau *= cosmic_constants::H0 / constants::H0;
	return tau;
}

__device__
float zero_order_universe::redshift_to_time(float z) const {
	float amax = 1.f / (1.f + z);
	float dloga = 1e-3;
	float logamin = LOG(amin);
	float logamax = LOG(amax);
	int N = (logamax - logamin) / dloga + 1;
	dloga = (logamax - logamin) / (float) N;
	float t = 0.0;
	for (int i = 0; i < N; i++) {
		float loga = logamin + (float) i * dloga;
		float a = EXP(loga);
		float t0 = t;
		t += dloga / hubble(a);
		loga = logamin + (float) (i + 1) * dloga;
		a = EXP(loga);
		t = 0.75f * t0 + 0.25f * (t + dloga / hubble(a));
		loga = logamin + ((float) i + 0.5f) * dloga;
		a = EXP(loga);
		t = (1.f / 3.f) * t0 + (2.f / 3.f) * (t + dloga / hubble(a));
	}
	t *= cosmic_constants::H0 / constants::H0;
	return t;
}

__device__
void create_zero_order_universe(zero_order_universe* uni_ptr, double amax) {
	zero_order_universe& uni = *uni_ptr;
	;
	printf("Creating zero order Universe\n");
	printf("Parameters:\n");
	using namespace constants;
	double omega_b = opts.omega_b;
	double omega_c = opts.omega_c;
	double omega_gam = opts.omega_gam;
	double omega_nu = opts.omega_nu;
	double omega_m = omega_b + omega_c;
	double omega_r = omega_gam + omega_nu;
	double Theta = opts.Theta;
	double littleh = opts.h;
	double Neff = opts.Neff;
	double Y = opts.Y;
	double amin = Theta * Tcmb / (0.07 * 1e6 * evtoK);
	double logamin = log(amin);
	double logamax = log(amax);
	int N = 512;
	double dloga = (logamax - logamin) / N;
	vector<float> thomson(N + 1);
	vector<float> sound_speed2(N + 1);

	printf("\t h                 = %f\n", littleh);
	printf("\t omega_m           = %f\n", omega_m);
	printf("\t omega_r           = %f\n", omega_r);
	printf("\t omega_lambda      = %f\n", 1 - omega_r - omega_m);
	printf("\t omega_b           = %f\n", omega_b);
	printf("\t omega_c           = %f\n", omega_c);
	printf("\t omega_gam         = %f\n", omega_gam);
	printf("\t omega_nu          = %f\n", omega_nu);
	printf("\t Neff              = %f\n", Neff);
	printf("\t temperature today = %f\n\n", 2.73 * Theta);

	dloga = (logamax - logamin) / N;
	const auto cosmic_hubble =
			[=](double a) {
				using namespace cosmic_constants;
				return littleh * cosmic_constants::H0 * sqrt(omega_r / (a * a * a * a) + omega_m / (a * a * a) + (1 - omega_r - omega_m));
			};
	auto cgs_hubble = [=](double a) {
		return constants::H0 / cosmic_constants::H0 * cosmic_hubble(a);
	};

	const auto rho_baryon = [=](double a) {
		using namespace constants;
		return 3.0 * pow(littleh * H0, 2) / (8.0 * M_PI * G) * omega_b / (a * a * a);
	};

	const auto T_radiation = [=](double a) {
		using namespace constants;
		return Tcmb * Theta / a;
	};

	double loga;
	double a;

	double rho_b, nH, nHp, nHe, nHep, nHepp, ne, Tgas, Trad;
	double hubble = cgs_hubble(amin);

	rho_b = rho_baryon(amin);
	double nnuc = rho_b / mh;
	nHepp = Y * nnuc / 4;
	nHp = (1 - Y) * nnuc;
	nH = nHe = nHep = 0.0;
	ne = nHp + 2 * nHepp;
	Trad = T_radiation(amin);
	Tgas = Trad;
	double n = nH + nHp + nHe + nHep + nHepp;
	double P = kb * (n + ne) * Tgas;
	double t = 0.0;
	double dt;
	for (int i = -10 / dloga; i <= 0; i++) {
		loga = logamin + i * dloga;
		dt = dloga / cgs_hubble(exp(loga));
		t += dt;
	}
	a = exp(logamin);
	printf("\n");
	print_time(t);
	printf(
			", redshift %.0f: Big Bang nucleosynthesis has ended. The Universe is dominated by radiation at a temperature of %8.2e K."
					" \n   Its total matter density is %.1f \% times the density of air at sea level.\n", 1 / a - 1,
			Trad, 100 * rho_b * omega_m / omega_b / 1.274e-3);
	double mu = (nH + nHp + 4 * nHe + 4 * nHep + 4 * nHepp) * mh / (nH + nHp + nHe + nHep + nHepp + ne);
	double sigmaC = mu / me * c * (8.0 / 3.0) * omega_gam / (a * omega_m) * sigma_T * ne / hubble;
	double sigmaT = c * sigma_T * ne / hubble;
	double Hionratio = nH != 0.0 ? nHp / nH : 1e+3;
	double Heionratio = nHe != 0.0 ? (nHep + nHepp) / nHe : 1e+3;
	bool sigmaT_decouple = false;
	bool sigmaC_decouple = false;
	bool radiation_matter_equality = false;
	bool Hdeionized = false;
	bool Hedeionized = false;
	bool dark_energy = false;
	bool habitable_begin = false;
	bool habitable_end = false;
	bool lowdensity = false;
	bool reallowdensity = false;
	bool earth_soundspeed = false;
	thomson[0] = sigmaT;
	double P1, P2;
	double rho1, rho2;
	double cs2;
	P1 = P2 = P;
	rho1 = rho2 = rho_b;
	for (int i = 1; i <= N; i++) {
		loga = logamin + i * dloga;
		a = exp(loga);
//		printf("%e %e %e %e %e %e\n", a, nH, nHp, nHe, nHep, nHepp);
		P2 = P1;
		P1 = P;
		rho2 = rho1;
		rho1 = rho_b;
		hubble = cgs_hubble(a);
		nH /= rho_b;
		nHp /= rho_b;
		nHe /= rho_b;
		nHep /= rho_b;
		nHepp /= rho_b;
		ne /= rho_b;
		rho_b = rho_baryon(a);
		nH *= rho_b;
		nHp *= rho_b;
		nHe *= rho_b;
		nHep *= rho_b;
		nHepp *= rho_b;
		ne *= rho_b;
		Trad = T_radiation(a);
		double dt = dloga / hubble;
		const double gamma = 1.0 - 1.0 / sqrt(2.0);
		chemistry_update(cgs_hubble, nH, nHp, nHe, nHep, nHepp, ne, Tgas, a, 0.5 * dt);
		mu = (nH + nHp + 4 * nHe + 4 * nHep + 4 * nHepp) * mh / (nH + nHp + nHe + nHep + nHepp + ne);
		sigmaC = mu / me * c * (8.0 / 3.0) * omega_gam / (a * omega_m) * sigma_T * ne / hubble;
		const double dTgasdT1 = ((Tgas + gamma * dloga * sigmaC * Trad) / (1 + gamma * dloga * (2 + sigmaC)) - Tgas)
				/ (gamma * dloga);
		const double T1 = Tgas + (1 - 2 * gamma) * dTgasdT1 * dloga;
		const double dTgasdT2 = ((T1 + gamma * dloga * sigmaC * Trad) / (1 + gamma * dloga * (2 + sigmaC)) - T1)
				/ (gamma * dloga);
		Tgas += 0.5 * (dTgasdT1 + dTgasdT2) * dloga;
		chemistry_update( cgs_hubble, nH, nHp, nHe, nHep, nHepp, ne, Tgas, a, 0.5 * dt);
		n = nH + nHp + nHe + nHep + nHepp;
		P = kb * (n + ne) * Tgas;
		sigmaT = c * sigma_T * ne / hubble;
		Hionratio = nH != 0.0 ? nHp / nH : 1e+3;
		Heionratio = nHe != 0.0 ? (nHep + nHepp) / nHe : 1e+3;
		t += dt;
		if (i == 1) {
			cs2 = (P - P1) / (rho_b - rho1);
		} else {
			cs2 = (P - P2) / (rho_b - rho2);
		}
		if (omega_m * a > omega_r && !radiation_matter_equality) {
			radiation_matter_equality = true;
			print_time(t);
			printf(", redshift %.0f: Radiation domination ends and matter domination begins at %8.2e K.\n", 1 / a - 1,
					Trad);
		}
		if (Heionratio < 1 && !Hedeionized) {
			Hedeionized = true;
			print_time(t);
			printf(", redshift %.1f: Electrons combine with Helium at a temperature of %8.2e K.\n", 1 / a - 1, Trad);
		}
		if (Hionratio < 1 && !Hdeionized) {
			Hdeionized = true;
			assert(Hdeionized);
			print_time(t);
			printf(", redshift %.1f: Electrons combine with Hydrogen at a temperature of %8.2e K.\n", 1 / a - 1, Trad);
		}
		if (sqrt(cs2) < 1 / 2.91e-5 && !earth_soundspeed) {
			earth_soundspeed = true;
			print_time(t);
			printf(
					", redshift %.1f: The speed of sound has dropped below the speed of sound at sea level at %8.2e K.\n",
					1 / a - 1, Trad);
		}
		if (Trad < 373 && habitable_begin == false) {
			habitable_begin = true;
			print_time(t);
			printf(", redshift %.1f: The Universe enters its first habitability period at %8.2e K.\n", 1 / a - 1, Trad);
		}
		if (Trad < 273 && habitable_end == false) {
			habitable_end = true;
			print_time(t);
			printf(", redshift %.1f: The Universe leaves its first habitability period at %8.2e K.\n", 1 / a - 1, Trad);
		}
		if (rho_b / mh < 1 && lowdensity == false) {
			lowdensity = true;
			print_time(t);
			printf(
					", redshift %.1f: The number of baryons per cubic centimeter has dropped to less than 1 at %8.2e K.\n",
					1 / a - 1, Trad);
		}
		if (rho_b / mh < 1e-6 && reallowdensity == false) {
			reallowdensity = true;
			print_time(t);
			printf(", redshift %.1f: The number of baryons per cubic meter has dropped to less than 1 at %8.2e K.\n",
					1 / a - 1, Trad);
		}
		if (sigmaT < 1.0 && !sigmaT_decouple) {
			sigmaT_decouple = true;
			print_time(t);
			printf(
					", redshift %.0f: the baryon velocities have decoupled from the radiation at a temperature of %8.2e K.\n",
					1 / a - 1, Trad);
		}
		if (sigmaC < 1.0 && !sigmaC_decouple) {
			sigmaC_decouple = true;
			print_time(t);
			printf(
					", redshift %.0f: the baryon temperature has decoupled from the radiation at a temperature of %8.2e K.\n",
					1 / a - 1, Trad);
		}
		if ((1 - omega_m - omega_r)
				> 0.5 * (omega_m / (a * a * a) + omega_r / (a * a * a * a) + (1 - omega_r - omega_m)) && !dark_energy) {
			dark_energy = true;
			print_time(t);
			printf(
					", redshift %.1f: Matter domination ends, dark energy is now the dominate force in the Universe, at a temperature of %8.2e K.\n",
					1 / a - 1, Trad);
		}
		sound_speed2[i - 1] = cs2 / (c * c);
		thomson[i] = sigmaT;
	}
	cs2 = (P - P1) / (rho_b - rho1);
	sound_speed2[N - 1] = cs2 / c;
	print_time(t);
	printf(
			", redshift %.0f. Present day. There are less than %i baryons per cubic meter. The Universe has cooled to %8.2e K\n",
			1 / a - 1, (int) (n * 1e6) + 1, Trad);
	uni.amin = amin;
	uni.amax = amax;
	build_interpolation_function(&uni.sigma_T, thomson, (float) amin, (float) amax);
	build_interpolation_function(&uni.cs2, sound_speed2, (float) amin, (float) amax);
	uni.hubble = std::move(cosmic_hubble);
}

