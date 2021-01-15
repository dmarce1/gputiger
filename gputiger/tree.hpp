/*
 * tree.hpp
 *
 *  Created on: Jan 14, 2021
 *      Author: dmarce1
 */

#ifndef TREE_HPP_
#define TREE_HPP_

#include <gputiger/stack.hpp>
#include <gputiger/mutex.hpp>
#include <gputiger/particle.hpp>
#include <gputiger/range.hpp>

struct tree_params {
	int parts_per_bucket;
	int kernel_depth;
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
	static void initialize(tree_params params, tree* arena, size_t bytes);
	__device__
	  static tree* alloc();
	__device__
	static void free(tree*);
	__device__
	void sort(particle* pbegin, particle* pend, const range<pos_type>&,  int depth);
	__device__
	void destroy();
};

#endif /* TREE_HPP_ */
