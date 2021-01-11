#pragma once

#include <boost/program_options.hpp>

#include <string>
#include <cstdint>

class options {
public:
	bool fiducial;
	double h;
	double Y;
	double Theta;
	double box_size;
	double omega_b;
	double omega_nu;
	double omega_gam;
	double omega_c;
	double omega_m;
	double omega_r;
	double Neff;
	double omega_lambda;
	std::uint64_t grid_dim;
	std::uint64_t nparts;
	bool process_options(int argc, char *argv[]);
};

