#pragma once

#include <gputiger/params.hpp>
#include <gputiger/array.hpp>

#define EWALD_DIM 32
#define EWALD_DIM3 (EWALD_DIM*EWALD_DIM*EWALD_DIM)

using ewald_table_t = array<array<float,NDIM+1>,EWALD_DIM3>;

__global__
void compute_ewald_table(ewald_table_t* table);
