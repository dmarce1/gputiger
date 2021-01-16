#pragma once


#include <gputiger/array.hpp>
#include <gputiger/params.hpp>
#include <gputiger/particle.hpp>

struct monopole {
	unsigned mass;
	array<pos_type,NDIM> xcom;
};

