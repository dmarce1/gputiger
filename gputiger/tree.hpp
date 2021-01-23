/*
 * tree.hpp
 *
 *  Created on: Jan 14, 2021
 *      Author: dmarce1
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include <gputiger/particle.hpp>
#include <gputiger/range.hpp>
#include <gputiger/monopole.hpp>
#include <gputiger/ewald.hpp>

#define KICKWARPSIZE 32
#define KICKEWALDWARPSIZE 32


struct tree {
	array<tree*, NCHILD> children;
	range box;
	monopole pole;
	particle* part_begin;
	particle* part_end;
	int depth;
	bool leaf;

	__device__
	static void initialize(particle* parts, void* arena, size_t bytes);
	__device__
	  static tree* alloc();
	__device__
	monopole sort(particle* swap_space, int depth);
	__device__
	void kick(tree* root, int rung, float dt,cudaTextureObject_t* tex_ewald);

};


__global__
void root_tree_sort(void* dataspace, int space_size, particle* swap_space, particle* parts, int* cnt);

__global__
void tree_kick(int rung, float scale, int*, int*);

__global__
void tree_drift(particle* parts, double ainv, double dt);


#endif /* TREE_HPP_ */
