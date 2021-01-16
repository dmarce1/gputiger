/*
 * chemsitry.hpp
 *
 *  Created on: Jan 10, 2021
 *      Author: dmarce1
 */

#ifndef GPUTIGER_CHEMISTRY_HPP_
#define GPUTIGER_CHEMISTRY_HPP_

#include <gputiger/constants.hpp>
#include <gputiger/math.hpp>
#include <gputiger/params.hpp>
#include <nvfunctional>


__device__
void chemistry_update(const nvstd::function<double(double)> &Hubble, double &H, double &Hp, double &He, double &Hep, double &Hepp,
		double &ne, double T, double a, double dt);
#endif /* GPUTIGER_CHEMISTRY_HPP_ */
