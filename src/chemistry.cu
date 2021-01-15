#include <gputiger/chemistry.hpp>

__device__
 void saha(const cosmic_parameters &opts, double rho, double T, double &H, double &Hp, double &He, double &Hep, double &Hepp, double &ne) {
	using namespace constants;
	constexpr double eps_1_H = 13.59844 * evtoerg;
	constexpr double eps_1_He = 24.58738 * evtoerg;
	constexpr double eps_2_He = 54.41776 * evtoerg;
	constexpr double g0_H = 2.0;
	constexpr double g1_H = 1.0;
	constexpr double g0_He = 1.0;
	constexpr double g1_He = 2.0;
	constexpr double g2_He = 1.0;
	const double lambda3 = pow(h * h / (2 * me * kb * T), 1.5);
	const double A0 = 2.0 / lambda3 * exp(-eps_1_H / (kb * T)) * g1_H / g0_H;
	const double B0 = 2.0 / lambda3 * exp(-eps_1_He / (kb * T)) * g1_He / g0_He;
	const double C0 = 2.0 / lambda3 * exp(-(eps_2_He - eps_1_He) / (kb * T)) * g2_He / g1_He;

	const double n_nuc = rho / mh;

	double H0 = n_nuc * (1 - opts.Y);
	double He0 = n_nuc * opts.Y / 4;
	double err;
	ne = 0.5 * (H0 + 2 * He0);
	int iters = 0;
	do {
		const double ne0 = ne;
		const double dne = -((ne - (A0 * H0) / (A0 + ne) - (B0 * He0 * (2 * C0 + ne)) / (pow(ne, 2) + B0 * (C0 + ne)))
				/ (1 + (A0 * H0) / pow(A0 + ne, 2) + (B0 * He0 * (B0 * C0 + ne * (4 * C0 + ne))) / pow(pow(ne, 2) + B0 * (C0 + ne), 2)));
		ne = min(max(0.5 * ne, ne + dne), 2 * ne);
		if (ne < 1e-100) {
			break;
		}
		err = abs(log(ne / ne0));
		iters++;
		if (iters > 1000) {
			printf("Max iters exceed in compute_electron_fraction\n");
			return;
		}
	} while (err > 1.0e-6);
	Hp = max(A0 * H0 / (A0 + ne), 0.0);
	H = max(H0 - Hp, 0.0);
	Hep = max(B0 * ne * He0 / (B0 * C0 + B0 * ne + ne * ne), 0.0);
	Hepp = max(B0 * C0 * He0 / (B0 * C0 + B0 * ne + ne * ne), 0.0);
	He = max(He0 - Hep - Hepp, 0.0);
}

__device__
 void chemistry_update(const cosmic_parameters &opts, const nvstd::function<double(double)> &Hubble, double &H, double &Hp, double &He, double &Hep, double &Hepp,
		double &ne, double T, double a, double dt) {
	using namespace constants;
	bool use_saha;
	double H1 = H;
	double Hp1 = Hp;
	double He1 = He;
	double Hep1 = Hep;
	double Hepp1 = Hepp;
	double ne1 = ne;
	double H0 = H;
	double Hp0 = Hp;
	double He0 = He;
	double Hep0 = Hep;
	double Hepp0 = Hepp;
	double rho = ((H + Hp) + (He + Hep + Hepp) * 4) * mh;
	if (ne > (H + Hp)) {
		saha(opts, rho, T, H1, Hp1, He1, Hep1, Hepp1, ne1);
		if (ne1 > (H1 + Hp1)) {
			use_saha = true;
		} else {
			use_saha = false;
		}
		use_saha = true;
	} else {
		use_saha = false;
	}
	if (use_saha) {
		H = H1;
		Hp = Hp1;
		He = He1;
		Hep = Hep1;
		Hepp = Hepp1;
		ne = ne1;
	} else {
		double nH = H + Hp;
		double x0 = Hp / nH;
		const auto dxdt = [=](double x0, double dt) {
			double hubble = Hubble(a);
			using namespace constants;
			const double B1 = 13.6 * evtoerg;
			const double phi2 = max(0.448 * log(B1 / (kb * T)), 0.0);
			const double alpha2 = 64.0 * M_PI / sqrt(27.0 * M_PI) * B1 * 2.0 * pow(hbar, 2) / pow(me * c, 3) / sqrt(kb * T / B1) * phi2;
			const double beta = pow((me * kb * T) / (2 * M_PI * hbar * hbar), 1.5) * exp(-B1 / (kb * T)) * alpha2;
			const double lambda_a = 8.0 * M_PI * hbar * c / (3.0 * B1);
			const double num = h * c / lambda_a / kb / T;
			const double beta2 = beta * exp(min(num, 80.0));
			const double La = 8.0 * M_PI * hubble / (a * pow(lambda_a, 3) * nH);
			const double L2s = 8.227;
			const auto func = [=](double x) {
				return (x - (dt * (L2s + La / (1 - x)) * (beta * (1 - x) - alpha2 * nH * pow(x, 2))) / (beta2 + L2s + La / (1 - x)) - x0) * (1 - x);
			};
			double x = find_root(func);
			return (x - x0) / dt;
		};
		double gam = 1.0 - 1.0 / sqrt(2.0);
		double dx1 = dxdt(x0, gam * dt);
		double dx2 = dxdt(x0 + (1 - 2 * gam) * dx1 * dt, gam * dt);
		double x = (x0 + 0.5 * (dx1 * dt + dx2 * dt));
		He = He0 + Hep0 + Hepp0;
		Hep = 0.0;
		Hepp = 0.0;
		H = (1.0 - x) * (H0 + Hp0);
		Hp = x * (H0 + Hp0);
		ne = Hp;
	}
}
