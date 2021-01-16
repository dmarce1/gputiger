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


struct sort_workspace {
	particle* begin[NCHILD];
	particle* end[NCHILD];
	array<range<pos_type>, NCHILD> cranges;
	int hi;
	int lo;
};

class tree {
	array<tree*, NCHILD> children;
	array<pos_type, NDIM> xcom;
	range<pos_type> box;
	particle* part_begin;
	particle* part_end;
	float mass;
	int depth;
	bool leaf;
public:
	__device__
	static void initialize(void* arena, size_t bytes);
	__device__
	  static tree* alloc();
	__device__
	void sort(sort_workspace*, particle* swap_space, particle* pbegin, particle* pend, range<pos_type>,  int depth);
	__device__
	void destroy();
};


__global__
void root_tree_sort(tree* root,particle* swap_space,  particle* pbegin, particle* pend, const range<pos_type> box);

#endif /* TREE_HPP_ */
