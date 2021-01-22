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

struct tree;
/*
struct tree_sort_type {
	array<monopole, NCHILD> poles;
	array<tree*, NCHILD> tree_ptrs;
	array<range, NCHILD> boxes;
	array<particle*, NCHILD> begins;
	array<particle*, NCHILD> ends;
};

struct sort_workspace {
	array<particle*, NCHILD> begin;
	array<particle*, NCHILD> end;
	array<range, NCHILD> cranges;
	int hi;
	int lo;
	array<array<float, NDIM>,WARPSIZE> poles;
	tree_sort_type* tree_sort;
	array<float,WARPSIZE> count;
};
*/
struct tree {
	array<tree*, NCHILD> children;
	range box;
	monopole pole;
	particle* part_begin;
	particle* part_end;
	int depth;
	bool leaf;

	__device__
	static void initialize(particle* parts, void* arena, size_t bytes, ewald_table_t* etable);
	__device__
	  static tree* alloc();
	__device__
	monopole sort(particle* swap_space, int depth);
	__device__
	void kick(tree* root, int rung, float dt,cudaTextureObject_t* tex_ewald);

};


__global__
void root_tree_sort(tree* root,particle* swap_space,  particle* pbegin, particle* pend, const range box);

#endif /* TREE_HPP_ */
