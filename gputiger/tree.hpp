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


struct sort_workspace {
	particle* begin[NCHILD];
	particle* end[NCHILD];
	array<range<pos_type>, NCHILD> cranges;
	int hi;
	int lo;
	array<int64_t, NDIM> poles[WARPSIZE];
	int count[WARPSIZE];
};

class tree {
	array<tree*, NCHILD> children;
	range<pos_type> box;
	monopole pole;
	particle* part_begin;
	particle* part_end;
	int depth;
	bool leaf;
public:
	__device__
	static void initialize(void* arena, size_t bytes);
	__device__
	  static tree* alloc();
	__device__
	monopole sort(sort_workspace*, particle* swap_space, particle* pbegin, particle* pend, range<pos_type>,  int depth);
	__device__
	void destroy();
};


__global__
void root_tree_sort(tree* root,particle* swap_space,  particle* pbegin, particle* pend, const range<pos_type> box);

#endif /* TREE_HPP_ */
