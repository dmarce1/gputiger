#pragma once


#include <gputiger/array.hpp>
#include <gputiger/params.hpp>
#include <gputiger/particle.hpp>

struct monopole {
	int count;
	array<fixed32,NDIM> xcom;
};

