#pragma once


#include <gputiger/array.hpp>
#include <gputiger/params.hpp>
#include <gputiger/particle.hpp>

struct monopole {
	float mass;
	array<float,NDIM> xcom;
};

