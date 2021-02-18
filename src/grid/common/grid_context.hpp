#ifndef GRID_CONTEXT_HPP
#define GRID_CONTEXT_HPP

#include <cassert>
#include <vector>

extern "C" {
		/* everything here is specific to the cpu and gpu backends*/
#include "grid_basis_set.h"
#include "grid_buffer.h"
#include "grid_common.h"
#include "grid_constants.h"
}

#include "task.hpp"
#include "Interval.hpp"
#include "grid_info.hpp"


class grid_context {
private:
		bool calculate_tau_{false};
		bool calculate_forces_{false};
		bool calculate_virial_{false};
public:
		std::vector<int> block_offsets_;
		std::vector<double> atom_positions_;
		std::vector<int> atom_kinds_;
		std::vector<grid_basis_set *> basis_sets_;
		std::vector<task_info> tasks_list_;
		std::vector<int> tasks_per_level_;
		std::vector<task_info*> queues_;
		std::vector<grid_info> grid_;
		int maxco_{0};
		bool apply_cutoff_{false};
		int queue_length_{0};
		void *backend_ctx_{nullptr};
		void *ref_backend_{nullptr};
		double *scratch{nullptr};
		bool orthorhombic_{false};
		enum grid_func func_;
		enum grid_backend backend_;
		tensor1<double, 2> forces_;
		tensor1<double, 2> virial_;
		grid_buffer *pab_blocks_{nullptr};
		grid_buffer *hab_blocks_{nullptr};
		grid_context(const grid_backend backend__);

		grid_buffer &pab_blocks() {
				return *pab_blocks_;
		}

		grid_buffer &hab_blocks() {
				return *hab_blocks_;
		}

		void set_pab_blocks(grid_buffer *ptr) {
				pab_blocks_ = ptr;
		}

		void set_hab_blocks(grid_buffer *ptr) {
				hab_blocks_ = ptr;
		}

		enum grid_func func() const {
				return func_;
		}

		void set_func(enum grid_func func__) {
				func_ = func__;
		}

		void set_orthorhombic(const bool tt) {
				orthorhombic_ = tt;
		}

		const bool is_orthorhombic() const {
				return orthorhombic_;
		}
		enum grid_backend backend() const {
				return backend_;
		}

		void update_backend(const grid_backend backend__);

		void change_backend(const grid_backend backend__);

		~grid_context();

		// void return_dh(const int level, double *const dh);
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
				return this->orthorhombic_;
		}

		void set_function(enum grid_func func) {
				func_ = func;
		}

		void *backend_context() {
				return backend_ctx_;
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

		grid_info &grid(const int level) {
				return grid_[level];
		}

		std::vector<grid_info> &grid() {
				return grid_;
		}

		const grid_info &grid(const int level) const {
				return grid_[level];
		}

		tensor1<double, 2> &forces() {
				return forces_;
		}

		const tensor1<double, 2> &forces() const {
				return forces_;
		}

		const double forces(int i, int j) const {
				return forces_(i, j);
		}

		double &forces(int i, int j) {
				return forces_(i, j);
		}

		tensor1<double, 2> &virial() {
				return virial_;
		}

		const tensor1<double, 2> &virial() const {
				return virial_;
		}

		const double virial(int i, int j) const {
				return virial_(i, j);
		}

		double &virial(int i, int j) {
				return virial_(i, j);
		}

		void collocate();
		void integrate();

		void integrate_test();
		void collocate_test();

		void return_dh(const int level, double *const dh);
		void return_dh_inv(const int level, double *const dh);

		const int maxco() const {
				return maxco_;
		}

		const int tasks_per_level(const int i) const {
				return tasks_per_level_[i];
		}

		task_info *queues(const int level, const int i) const {
				return this->queues_[level] + i;
		}

		const bool calculate_tau() const {
				return calculate_tau_;
		}

		const bool calculate_forces() const {
				return calculate_forces_;
		}

		const bool calculate_virial() const {
				return calculate_virial_;
		}

		void calculate_tau(const bool value) {
				calculate_tau_ = value;
		}

		void calculate_forces(const bool value) {
				calculate_forces_ = value;
		}

		void calculate_virial(const bool value) {
				calculate_virial_ = value;
		}

};

#endif
