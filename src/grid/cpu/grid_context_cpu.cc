/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#include <cmath>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <iomanip>

extern "C" {
#include "../common/grid_library.h"
}

#include "task.hpp"
#include "grid_info.hpp"
#include "utils.hpp"
#include "grid_context_cpu.hpp"
#include "cpu_handler.hpp"

static void rotate_to_cartesian_harmonics(const grid_basis_set *ibasis,
																					const grid_basis_set *jbasis,
																					const int iatom, const int jatom,
																					const int iset, const int jset,
																					double *const block, tensor1<double, 2> &work,
																					tensor1<double, 2> &pab) {
		dgemm_params m1, m2;
	memset(&m1, 0, sizeof(dgemm_params));
	memset(&m2, 0, sizeof(dgemm_params));

	// Define some more convenient aliases.
	const int nsgf_seta = ibasis->nsgf_set[iset]; // size of spherical set
	const int nsgf_setb = jbasis->nsgf_set[jset];
	const int nsgfa = ibasis->nsgf; // size of entire spherical basis
	const int nsgfb = jbasis->nsgf;
	const int sgfa = ibasis->first_sgf[iset] - 1; // start of spherical set
	const int sgfb = jbasis->first_sgf[jset] - 1;
	const int maxcoa = ibasis->maxco;
	const int maxcob = jbasis->maxco;
	const int ncoseta = ncoset(ibasis->lmax[iset]);
	const int ncosetb = ncoset(jbasis->lmax[jset]);
	const int ncoa = ibasis->npgf[iset] * ncoseta; // size of carthesian set
	const int ncob = jbasis->npgf[jset] * ncosetb;

	work.resize(nsgf_setb, ncoa);
	pab.resize(ncob, ncoa);

	// the rotations happen here.
	if (iatom <= jatom) {
		m1.op1 = 'N';
		m1.op2 = 'N';
		m1.m = work.size(0);
		m1.n = work.size(1);
		m1.k = nsgf_seta;
		m1.alpha = 1.0;
		m1.beta = 0.0;
		m1.a = block + sgfb * nsgfa + sgfa;
		m1.lda = nsgfa;
		m1.b = &ibasis->sphi[sgfa * maxcoa];
		m1.ldb = maxcoa;
		m1.c = work.at();
		m1.ldc = work.ld();
	} else {
		m1.op1 = 'T';
		m1.op2 = 'N';
		m1.m = work.size(0);
		m1.n = work.size(1);
		m1.k = nsgf_seta;
		m1.alpha = 1.0;
		m1.beta = 0.0;
		m1.a = block + sgfa * nsgfb + sgfb;
		m1.lda = nsgfb;
		m1.b = &ibasis->sphi[sgfa * maxcoa];
		m1.ldb = maxcoa;
		m1.c = work.at();
		m1.ldc = work.ld();
	}
	m1.use_libxsmm = true;
	dgemm_simplified(&m1);

	m2.op1 = 'T';
	m2.op2 = 'N';
	m2.m = pab.size(0);
	m2.n = pab.size(1);
	m2.k = work.size(0);
	m2.alpha = 1.0;
	m2.beta = 0.0;
	m2.a = &jbasis->sphi[sgfb * maxcob];
	m2.lda = maxcob;
	m2.b = work.at();
	m2.ldb = work.ld();
	m2.c = pab.at();
	m2.ldc = pab.ld();
	m2.use_libxsmm = true;
	dgemm_simplified(&m2);
}


void grid_context::return_dh(const int level, double *const dh) {
	dh[0] = this->grid[level].dh[0][0];
	dh[1] = this->grid[level].dh[0][1];
	dh[2] = this->grid[level].dh[0][2];
	dh[3] = this->grid[level].dh[1][0];
	dh[4] = this->grid[level].dh[1][1];
	dh[5] = this->grid[level].dh[1][2];
	dh[6] = this->grid[level].dh[2][0];
	dh[7] = this->grid[level].dh[2][1];
	dh[8] = this->grid[level].dh[2][2];
}

void grid_context::dh_inv(const int level, double *const dh_inv) {
	dh_inv[0] = this->grid[level].dh_inv[0][0];
	dh_inv[1] = this->grid[level].dh_inv[0][1];
	dh_inv[2] = this->grid[level].dh_inv[0][2];
	dh_inv[3] = this->grid[level].dh_inv[1][0];
	dh_inv[4] = this->grid[level].dh_inv[1][1];
	dh_inv[5] = this->grid[level].dh_inv[1][2];
	dh_inv[6] = this->grid[level].dh_inv[2][0];
	dh_inv[7] = this->grid[level].dh_inv[2][1];
	dh_inv[8] = this->grid[level].dh_inv[2][2];
}


void grid_context::update_queue_length(const int queue_length) {
	this->queue_length = queue_length;
}

void grid_context::update_atoms_position(const int natoms,
																				 const double *atoms_positions) {
	this->atom_positions_.clear();
	this->atom_positions_.resize(3 * natoms);

	for (auto i = 0u; i < this->atom_positions_.size(); i++) {
			this->atom_positions_[i] = atoms_positions[i];
	}
}

void grid_context::update_atoms_kinds(const int natoms, const int *atoms_kinds) {
	this->atom_kinds_.clear();
	this->atom_kinds_.resize(natoms);

	memcpy(&this->atom_kinds_[0], atoms_kinds, sizeof(int) * natoms);

	for (auto i = 0u; i < this->atom_kinds_.size(); i++) {
		this->atom_kinds_[i] -= 1;
	}
}

void grid_context::update_block_offsets(const int nblocks, const int *const block_offsets) {
	if (nblocks == 0)
		return;

	this->block_offsets_.clear();
	this->block_offsets_.resize(nblocks);

	memcpy(&this->block_offsets_[0], block_offsets, nblocks * sizeof(int));
}

void grid_context::update_basis_set(const int nkinds, const grid_basis_set **const basis_sets) {
	this->basis_sets_.clear();
	this->basis_sets_.resize(nkinds);
	memcpy(&this->basis_sets_[0], basis_sets, nkinds * sizeof(grid_basis_set *));
}

void grid_context::update_task_lists(const int nlevels, const int ntasks,
											 const int *const level_list, const int *const iatom_list,
											 const int *const jatom_list, const int *const iset_list,
											 const int *const jset_list, const int *const ipgf_list,
											 const int *const jpgf_list,
											 const int *const border_mask_list,
											 const int *block_num_list,
											 const double *const radius_list,
											 const double *rab_list) {
		if (nlevels == 0)
				return;

		// Count tasks per level.
		this->tasks_per_level.clear();
		this->tasks_per_level.resize(nlevels);
		this->tasks_list_.clear();
		this->tasks_list_.resize(ntasks);
		memset(&this->tasks_list_[0], 0, sizeof(task_info) * this->tasks_list_.size());
		this->queues_.clear();
		this->queues_.resize(nlevels);

		memset(&this->tasks_per_level[0], 0, sizeof(int) * nlevels);
		for (int i = 0; i < ntasks; i++) {
				this->tasks_per_level[level_list[i] - 1]++;
				assert(i == 0 || level_list[i] >= level_list[i - 1]); // expect ordered list
		}

		this->queues_[0] = &this->tasks_list_[0];

		for (auto i = 1u; i < this->tasks_per_level.size(); i++) {
				this->queues_[i] = this->queues_[i - 1] + this->tasks_per_level[i - 1];
		}

		int prev_block_num = -1;
		int prev_iset = -1;
		int prev_jset = -1;
		int prev_level = -1;
		for (int i = 0; i < ntasks; i++) {
				auto &task_ = this->tasks_list_[i];
				if (prev_level != (level_list[i] - 1)) {
						prev_level = level_list[i] - 1;
						prev_block_num = -1;
						prev_iset = -1;
						prev_jset = -1;
				}
				task_.level = level_list[i] - 1;
				task_.iatom = iatom_list[i] - 1;
				task_.jatom = jatom_list[i] - 1;
				task_.iset = iset_list[i] - 1;
				task_.jset = jset_list[i] - 1;
				task_.ipgf = ipgf_list[i] - 1;
				task_.jpgf = jpgf_list[i] - 1;
				task_.border_mask = border_mask_list[i];
				task_.block_num = block_num_list[i] - 1;
				task_.radius = radius_list[i];
				task_.rab[0] = rab_list[3 * i];
				task_.rab[1] = rab_list[3 * i + 1];
				task_.rab[2] = rab_list[3 * i + 2];
				const int iatom = task_.iatom;
				const int jatom = task_.jatom;
				const int iset = task_.iset;
				const int jset = task_.jset;
				const int ipgf = task_.ipgf;
				const int jpgf = task_.jpgf;
				const int ikind = this->atom_kinds_[iatom];
				const int jkind = this->atom_kinds_[jatom];
				const grid_basis_set *ibasis = this->basis_sets_[ikind];
				const grid_basis_set *jbasis = this->basis_sets_[jkind];
				const int ncoseta = ncoset(ibasis->lmax[iset]);
				const int ncosetb = ncoset(jbasis->lmax[jset]);

				task_.zeta[0] = ibasis->zet[iset * ibasis->maxpgf + ipgf];
				task_.zeta[1] = jbasis->zet[jset * jbasis->maxpgf + jpgf];

				const double *ra = &this->atom_positions_[3 * iatom];
				const double zetp = task_.zeta[0] + task_.zeta[1];
				const double f = task_.zeta[1] / zetp;
				const double rab2 = task_.rab[0] * task_.rab[0] +
						task_.rab[1] * task_.rab[1] +
						task_.rab[2] * task_.rab[2];

				task_.prefactor = exp(-task_.zeta[0] * f * rab2);
				task_.zetp = zetp;

				const int block_num = task_.block_num;

				for (int i = 0; i < 3; i++) {
						task_.ra[i] = ra[i];
						task_.rp[i] = ra[i] + f * task_.rab[i];
						task_.rb[i] = ra[i] + task_.rab[i];
				}

				task_.lmax[0] = ibasis->lmax[iset];
				task_.lmax[1] = jbasis->lmax[jset];
				task_.lmin[0] = ibasis->lmin[iset];
				task_.lmin[1] = jbasis->lmin[jset];

				if ((block_num != prev_block_num) || (iset != prev_iset) ||
						(jset != prev_jset)) {
						task_.update_block_ = true;
						prev_block_num = block_num;
						prev_iset = iset;
						prev_jset = jset;
				} else {
						task_.update_block_ = false;
				}

				task_.offset[0] = ipgf * ncoseta;
				task_.offset[1] = jpgf * ncosetb;
		}

		// Find largest Cartesian subblock size.
		this->maxco = 0;
		for (auto kind: this->basis_sets_) {
				this->maxco = imax(this->maxco, kind->maxco);
		}
}

void grid_context::update_grid(const int nlevels) {
	if (nlevels == 0)
		return;
	this->grid.clear();
	this->grid.resize(nlevels);
}


void grid_context::collocate_one_grid_level(const int *const border_width,
																						const int *const shift_local,
																						const int level,
																						const grid_buffer *pab_blocks) {
		assert(this->handler.size());
		assert(this->grid.size());

		auto &grid = this->grid[level];
	// Using default(shared) because with GCC 9 the behavior around const changed:
	// https://www.gnu.org/software/gcc/gcc-9/porting_to.html
#pragma omp parallel default(shared)
	{
		const int num_threads = omp_get_num_threads();
		const int thread_id = omp_get_thread_num();

		auto &handler = this->handler[thread_id];

		handler.func = func_;
		this->get_ldiffs(handler.func,
										 handler.lmin_diff,
										 handler.lmax_diff);

		handler.apply_cutoff = this->apply_cutoff();

		// Allocate pab matrix for re-use across tasks.
		handler.pab().resize(this->maxco, this->maxco);
		handler.work().resize(this->maxco, this->maxco);
		handler.pab_prep().resize(this->maxco, this->maxco);

		handler.initialize_basis_vectors(grid.dh, grid.dh_inv);

		/* setup the grid parameters, window parameters (if the grid is split), etc
		 */

		handler.grid() = grid;

		for (int d = 0; d < 3; d++)
				handler.orthogonal[d] = handler.grid().orthogonal[d];

		if ((num_threads > 1) && (thread_id > 0)) {
				handler.grid().update_pointer(((double *)this->scratch) +
																			(thread_id - 1) * handler.grid().size());
				handler.grid().zero();
		}

		/* it is only useful when we split the list over multiple threads. The first
		 * iteration should load the block whatever status the task.block_update_
		 * has */
		const task_info *prevtask_info = NULL;
#pragma omp for schedule(static)
		for (int itask = 0; itask < this->tasks_per_level[level]; itask++) {
			// Define some convenient aliases.
			const task_info *task = this->queues_[level] + itask;

			if (task->level != level) {
				printf("level %d, %d\n", task->level, level);
				abort();
			}
			/* the grid is divided over several ranks or not periodic */
			if (handler.grid().is_distributed()) {
				/* unfortunately the window where the gaussian should be added depends
				 * on the bonds. So I have to adjust the window all the time. */

					handler.grid().setup_grid_window(shift_local, border_width,
																				 task->border_mask);
			}

			/* this is a three steps procedure. pab_blocks contains the coefficients
			 * of the operator in the spherical harmonic basis while we do computation
			 * in the cartesian harmonic basis.
			 *
			 * step 1 : rotate the coefficients from the harmonic to the cartesian
			 * basis

			 * step 2 : extract the subblock and apply additional transformations
			 * corresponding the spatial derivatives of the operator (the last is not
			 * always done)

			 * step 3 : change from (x - x_1)^\alpha (x - x_2)^\beta to (x -
			 * x_{12})^k. It is a transformation which involves binomial
			 * coefficients.
			 *
			 * \f[ (x - x_1) ^\alpha (x - x_2) ^ beta = \sum_{k_{1} k_{2}} ^
			 *     {\alpha\beta} \text{Binomial}(\alpha,k_1)
			 *     \text{Binomial}(\beta,k_2) (x - x_{12})^{k_1 + k_2} (x_12 - x_1)
			 *     ^{\alpha - k_1} (x_12 - x_2) ^{\beta - k_2} ]
			 *
			 * step 1 is done only when necessary, the two remaining steps are done
			 * for each bond.
			 */

			this->compute_coefficients(handler, prevtask_info, *task, pab_blocks, handler.pab(),
																 handler.work(), handler.pab_prep());

			handler.collocate(this->orthorhombic, *task);
			prevtask_info = task;
		}

		// Merge thread local grids into shared grid. Could be improved though....

		// thread 0 does nothing since the data are already placed in the final
		// destination
		if (num_threads > 1) {
				if ((grid.size() / (num_threads - 1)) >= 2) {
						const int block_size = grid.size() / (num_threads - 1) +
								(grid.size() % (num_threads - 1));

				for (int bk = 0; bk < num_threads; bk++) {
					if (thread_id > 0) {
						int bk_id = (bk + thread_id - 1) % num_threads;
						int begin = bk_id * block_size;
						int end = std::min((bk_id + 1) * block_size, grid.size());
						cblas_daxpy(end - begin, 1.0, handler.grid().at() + begin, 1,
												grid.at() + begin, 1);
					}
#pragma omp barrier
				}
			}
		} else {
#pragma omp critical
			if (thread_id > 0)
					cblas_daxpy(handler.grid().size(), 1.0,
											handler.grid().at(0, 0, 0), 1, grid.at(0, 0, 0),
										1);
		}
	}
}

void grid_context::rotate_and_store_coefficients(const task_info *prev_task,
																								 const task_info *task, tensor1<double, 2> &hab,
																								 tensor1<double, 2> &work, // some scratch matrix
																								 double *blocks) {
		if (prev_task != NULL) {
		/* prev_task is NULL when we deal with the first iteration. In that case
		 * we just need to initialize the hab matrix to the proper size. Note
		 * that resizing does not really occurs since memory allocation is done
		 * for the maximum size the matrix can have. */
		const int iatom = prev_task->iatom;
		const int jatom = prev_task->jatom;
		const int iset = prev_task->iset;
		const int jset = prev_task->jset;
		const int ikind = this->atom_kinds_[iatom];
		const int jkind = this->atom_kinds_[jatom];
		const grid_basis_set *ibasis = this->basis_sets_[ikind];
		const grid_basis_set *jbasis = this->basis_sets_[jkind];

		const int block_num = prev_task->block_num;
		double *const block = &blocks[this->block_offsets_[block_num]];

		const int ncoseta = ncoset(ibasis->lmax[iset]);
		const int ncosetb = ncoset(jbasis->lmax[jset]);

		const int ncoa = ibasis->npgf[iset] * ncoseta;
		const int ncob = jbasis->npgf[jset] * ncosetb;

		const int nsgf_seta = ibasis->nsgf_set[iset]; // size of spherical set */
		const int nsgf_setb = jbasis->nsgf_set[jset];
		const int nsgfa = ibasis->nsgf; // size of entire spherical basis
		const int nsgfb = jbasis->nsgf;
		const int sgfa = ibasis->first_sgf[iset] - 1; // start of spherical set
		const int sgfb = jbasis->first_sgf[jset] - 1;
		const int maxcoa = ibasis->maxco;
		const int maxcob = jbasis->maxco;

		work.resize(nsgf_setb, ncoa);

		// Warning these matrices are row major....

		dgemm_params m1, m2;
		memset(&m1, 0, sizeof(dgemm_params));
		memset(&m2, 0, sizeof(dgemm_params));

		m1.op1 = 'N';
		m1.op2 = 'N';
		m1.m = nsgf_setb;
		m1.n = ncoa;
		m1.k = ncob;
		m1.alpha = 1.0;
		m1.beta = 0.0;
		m1.a = &jbasis->sphi[sgfb * maxcob];
		m1.lda = maxcob;
		m1.b = hab.at();
		m1.ldb = ncoa;
		m1.c = work.at();
		m1.ldc = work.ld();

		// phi[b][ncob]
		// work[b][ncoa] = phi[b][ncob] * hab[ncob][ncoa]

		if (iatom <= jatom) {
			// I need to have the final result in the form

			// block[b][a] = work[b][ncoa] transpose(phi[a][ncoa])
			m2.op1 = 'N';
			m2.op2 = 'T';
			m2.m = nsgf_setb;
			m2.n = nsgf_seta;
			m2.k = ncoa;
			m2.a = work.at();
			m2.lda = work.ld();
			m2.b = &ibasis->sphi[sgfa * maxcoa];
			m2.ldb = maxcoa;
			m2.c = block + sgfb * nsgfa + sgfa;
			m2.ldc = nsgfa;
			m2.alpha = 1.0;
			m2.beta = 1.0;
		} else {
			// block[a][b] = phi[a][ncoa] Transpose(work[b][ncoa])
			m2.op1 = 'N';
			m2.op2 = 'T';
			m2.m = nsgf_seta;
			m2.n = nsgf_setb;
			m2.k = ncoa;
			m2.a = &ibasis->sphi[sgfa * maxcoa];
			m2.lda = maxcoa;
			m2.b = work.at();
			m2.ldb = work.ld();
			m2.c = block + sgfa * nsgfb + sgfb;
			m2.ldc = nsgfb;
			m2.alpha = 1.0;
			m2.beta = 1.0;
		}

		m1.use_libxsmm = true;
		m2.use_libxsmm = true;

		/* these dgemm are *row* major */
		dgemm_simplified(&m1);
		dgemm_simplified(&m2);
	}

	if (task != NULL) {
		const int iatom = task->iatom;
		const int jatom = task->jatom;
		const int ikind = this->atom_kinds_[iatom];
		const int jkind = this->atom_kinds_[jatom];
		const int iset = task->iset;
		const int jset = task->jset;
		const grid_basis_set *ibasis = this->basis_sets_[ikind];
		const grid_basis_set *jbasis = this->basis_sets_[jkind];
		const int ncoseta = ncoset(ibasis->lmax[iset]);
		const int ncosetb = ncoset(jbasis->lmax[jset]);

		const int ncoa = ibasis->npgf[iset] * ncoseta;
		const int ncob = jbasis->npgf[jset] * ncosetb;

		hab.resize(ncob, ncoa);
	}
}


void grid_context::extract_blocks(const task_info &task,
																	const grid_buffer *pab_blocks,
																	tensor1<double, 2> &work,
																	tensor1<double, 2> &pab) {
		const int iatom = task.iatom;
		const int jatom = task.jatom;
		const int iset = task.iset;
		const int jset = task.jset;
		const int ikind = this->atom_kinds_[iatom];
		const int jkind = this->atom_kinds_[jatom];
		const grid_basis_set *ibasis = this->basis_sets_[ikind];
		const grid_basis_set *jbasis = this->basis_sets_[jkind];

		const int block_num = task.block_num;

		// Locate current matrix block within the buffer. This block
		// contains the weights of the gaussian pairs in the spherical
		// harmonic basis, but we do computation in the cartesian
		// harmonic basis so we have to rotate the coefficients. It is nothing
		// else than a basis change and it done with two dgemm.

		const int block_offset = this->block_offsets_[block_num]; // zero based
		double *const block = &pab_blocks->host_buffer[block_offset];

		rotate_to_cartesian_harmonics(ibasis, jbasis, iatom, jatom, iset, jset, block,
																	work, pab);
}

void grid_context::compute_coefficients(cpu_handler &handler,
																				const task_info *const previous_task,
																				const task_info &task,
																				const grid_buffer *pab_blocks, tensor1<double, 2> &pab,
																				tensor1<double, 2> &work, tensor1<double, 2> &pab_prep) {
		// Load subblock from buffer and decontract into Cartesian sublock pab.
		// The previous pab can be reused when only ipgf or jpgf has changed.
		if (task.update_block_ || (previous_task == NULL)) {
				this->extract_blocks(task, pab_blocks, work, pab);
		}

		int lmin_prep[2];
		int lmax_prep[2];

		lmin_prep[0] = std::max(task.lmin[0] + handler.lmin_diff[0], 0);
		lmin_prep[1] = std::max(task.lmin[1] + handler.lmin_diff[1], 0);

		lmax_prep[0] = task.lmax[0] + handler.lmax_diff[0];
		lmax_prep[1] = task.lmax[1] + handler.lmax_diff[1];

		const int n1_prep = ncoset(lmax_prep[0]);
		const int n2_prep = ncoset(lmax_prep[1]);

	/* we do not reallocate memory. We initialized the structure with the
	 * maximum lmax of the all list already.
	 */
	pab_prep.resize(n2_prep, n1_prep);

	this->prepare_pab(handler.func, task.offset, task.lmin, task.lmax,
										&task.zeta[0], pab, pab_prep);

	//   *** initialise the coefficient matrix, we transform the sum
	//
	// sum_{lxa,lya,lza,lxb,lyb,lzb} P_{lxa,lya,lza,lxb,lyb,lzb} *
	//         (x-a_x)**lxa (y-a_y)**lya (z-a_z)**lza (x-b_x)**lxb
	//         (y-a_y)**lya (z-a_z)**lza
	//
	// into
	//
	// sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-p_x)**lxp (y-p_y)**lyp
	// (z-p_z)**lzp
	//
	// where p is center of the product gaussian, and lp = la_max + lb_max
	// (current implementation is l**7)
	//

	/* precautionary tail since I probably intitialize data to NULL when I
	 * initialize a new tensor. I want to keep the memory (I put a ridiculous
	 * amount already) */

	handler.alpha().resize(3, lmax_prep[1] + 1, lmax_prep[0] + 1,
												 lmax_prep[0] + lmax_prep[1] + 1);

	const int lp = lmax_prep[0] + lmax_prep[1];

	handler.coef().resize(lp + 1, lp + 1, lp + 1);

	// these two functions can be done with dgemm again....

	handler.prepare_alpha(task, lmax_prep);

	// compute the coefficients after applying the function of interest
	// coef[x][z][y]
	handler.compute_coefficients(lmin_prep, lmax_prep, lp,
															 task.prefactor * ((task.iatom == task.jatom) ? 1.0 : 2.0), pab_prep);
}


/*******************************************************************************
 * \brief Allocates a task list for the cpu backend.
 *        See grid_task_list.h for details.
 ******************************************************************************/
extern "C" void grid_cpu_create_task_list(
		const int ntasks, const int nlevels, const int natoms, const int nkinds,
		const int nblocks, const int *block_offsets,
		const double *atom_positions, const int *atom_kinds,
		const grid_basis_set **basis_sets, const int *level_list,
		const int *iatom_list, const int *jatom_list,
		const int *iset_list, const int *jset_list,
		const int *ipgf_list, const int *jpgf_list,
		const int *border_mask_list, const int *block_num_list,
		const double *radius_list, const double *rab_list,
		void *ptr) {

		grid_context **task_list = static_cast<grid_context **>(ptr);
		grid_context *ctx = nullptr;
		if (*task_list == nullptr) {
				ctx = new grid_context();
				const int max_threads = omp_get_max_threads();

				ctx->handler.clear();
				ctx->handler.resize(max_threads);

				for (auto &h : ctx->handler)
						h.initialize(1);
		} else {
				ctx = *task_list;
		}

		ctx->update_block_offsets(nblocks, block_offsets);
		ctx->update_atoms_position(natoms, atom_positions);
		ctx->update_atoms_kinds(natoms, atom_kinds);
		ctx->update_basis_set(nkinds, basis_sets);
		ctx->update_task_lists(nlevels, ntasks, level_list, iatom_list, jatom_list,
											iset_list, jset_list, ipgf_list, jpgf_list,
											border_mask_list, block_num_list, radius_list, rab_list);
		ctx->update_grid(nlevels);

		// Find largest Cartesian subblock size.
		ctx->maxco = 0;
		for (int i = 0; i < nkinds; i++) {
				ctx->maxco = imax(ctx->maxco, ctx->basis_sets_[i]->maxco);
		}

		const grid_library_config config = grid_library_get_config();
		if (config.apply_cutoff) {
				ctx->apply_cutoff(true);
		}
		*task_list = ctx;
}

/*******************************************************************************
 * \brief Deallocates given task list, basis_sets have to be freed separately.
 ******************************************************************************/
extern "C" void grid_cpu_free_task_list(void *ptr) {
		grid_context *ctx = static_cast<grid_context *>(ptr);
		ctx->block_offsets_.clear();
		ctx->atom_positions_.clear();
		ctx->atom_kinds_.clear();
		ctx->basis_sets_.clear();
		ctx->tasks_list_.clear();
		ctx->tasks_per_level.clear();
		ctx->queues_.clear();
		ctx->handler.clear();
		delete ctx;
}
