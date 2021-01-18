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

struct tree;

struct tree_sort_type {
	array<tree*, NCHILD> tree_ptrs;
	array<range, NCHILD> boxes;
	array<particle*, NCHILD> begins;
	array<particle*, NCHILD> ends;
	array<monopole, NCHILD> poles;
};

struct sort_workspace {
	particle* begin[NCHILD];
	particle* end[NCHILD];
	array<range, NCHILD> cranges;
	int hi;
	int lo;
	array<float, NDIM> poles[WARPSIZE];
	tree_sort_type* tree_sort;
	float count[WARPSIZE];
};

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
	monopole sort(sort_workspace*, particle* swap_space, particle* pbegin, particle* pend, range,  int depth, int rung);
	__device__
	void kick(tree* root, int rung, float dt);

};


__global__
void root_tree_sort(tree* root,particle* swap_space,  particle* pbegin, particle* pend, const range box, int rung);

#endif /* TREE_HPP_ */
