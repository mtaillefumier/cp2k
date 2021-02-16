/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#ifndef CPU_PRIVATE_HEADER_HPP
#define CPU_PRIVATE_HEADER_HPP

#include <cassert>
#include <vector>

extern "C" {
		/* everything here is specific to the cpu and gpu backends*/
#include "../common/grid_basis_set.h"
#include "../common/grid_buffer.h"
#include "../common/grid_common.h"
#include "../common/grid_constants.h"
}

#include "../common/task.hpp"
#include "../common/Interval.hpp"
#include "grid_info.hpp"
#include "cpu_handler.hpp"

enum checksum_ { task_checksum = 0x2384989, ctx_checksum = 0x2356734 };

class grid_context {
public:
		std::vector<int> block_offsets_;
		std::vector<double> atom_positions_;
		std::vector<int> atom_kinds_;
		std::vector<grid_basis_set *> basis_sets_;
		std::vector<task_info> tasks_list_;
		std::vector<int> tasks_per_level;
		std::vector<task_info*> queues_;
		int maxco{0};
		bool apply_cutoff_{false};
		int queue_length{0};
		std::vector<cpu_handler> handler;
		std::vector<grid_info> grid;
		double *scratch{nullptr};
		bool orthorhombic{false};
		enum grid_func func_;

		grid_context() {
				handler.clear();
		}

		~grid_context() {
				block_offsets_.clear();
				atom_positions_.clear();
				atom_kinds_.clear();
				basis_sets_.clear();
				tasks_list_.clear();
				tasks_per_level.clear();
				queues_.clear();
				handler.clear();
				grid.clear();
		}

		void collocate_one_grid_level(const int *const, const int *const,
																	const int level,
																	const grid_buffer *pab_blocks);

		void integrate_one_grid_level(const int level, const bool calculate_tau,
																	const bool calculate_forces, const bool calculate_virial,
																	const int *const shift_local, const int *const border_width,
																	const grid_buffer *const pab_blocks, grid_buffer *const hab_blocks,
																	tensor1<double, 2> &forces_, tensor1<double, 2> &virial_);

		void compute_coefficients(cpu_handler &handler,
															const task_info *previous_task, const task_info &task,
															const grid_buffer *pab_blocks,
															tensor1<double, 2> &pab, tensor1<double, 2> &work,
															tensor1<double, 2> &pab_prep);

		void extract_blocks(const task_info &task,
												const grid_buffer *pab_blocks,
												tensor1<double, 2> &work,
												tensor1<double, 2> &pab);

		void return_dh(const int level, double *const dh);
		void dh_inv(const int level, double *const dh_inv);
		void apply_cutoff(const bool apply_cutoff__) {
				apply_cutoff_ = apply_cutoff__;
		}
		const bool apply_cutoff() const {
				return apply_cutoff_;
		}
		void rotate_and_store_coefficients(const task_info *prev_task,
																			 const task_info *task, tensor1<double, 2> &hab,
																			 tensor1<double, 2> &work, // some scratch matrix
																			 double *blocks);
		const bool is_grid_orthorhombic() {
				return this->orthorhombic;
		}

		void set_function(enum grid_func func) {
				func_ = func;
		}

		void update_queue_length(const int queue_length);
		void update_atoms_position(const int natoms,
															 const double *atoms_positions);
		void update_atoms_kinds(const int natoms, const int *atoms_kinds);
		void update_block_offsets(const int nblocks, const int *const block_offsets);
		void update_basis_set(const int nkinds, const grid_basis_set **const basis_sets);
		void update_task_lists(const int nlevels, const int ntasks,
													 const int *const level_list, const int *const iatom_list,
													 const int *const jatom_list, const int *const iset_list,
													 const int *const jset_list, const int *const ipgf_list,
													 const int *const jpgf_list,
													 const int *const border_mask_list,
													 const int *block_num_list,
													 const double *const radius_list,
													 const double *rab_list);
		void update_grid(const int nlevels);
		void prepare_pab(const enum grid_func func, const int *const offset,
										 const int *const lmin, const int *const lmax,
										 const double *const zeta, tensor1<double, 2> &pab,
										 tensor1<double, 2> &pab_prep);
		void get_ldiffs(const enum grid_func func,
										int *const lmin_diff, int *const lmax_diff);
};

#endif
