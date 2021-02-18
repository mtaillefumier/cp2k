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

#include "../common/task.hpp"
#include "../common/grid_info.hpp"
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


void cpu_backend::collocate_one_grid_level(const int level) {
		assert(this->handler_.size());
		assert(ctx_.grid().size());

		auto &grid = ctx_.grid(level);
	// Using default(shared) because with GCC 9 the behavior around const changed:
	// https://www.gnu.org/software/gcc/gcc-9/porting_to.html
#pragma omp parallel default(shared)
	{
		const int num_threads = omp_get_num_threads();
		const int thread_id = omp_get_thread_num();

		auto &handler = this->handler_[thread_id];

		handler.func = ctx_.func();
		this->get_ldiffs(handler.func,
										 handler.lmin_diff,
										 handler.lmax_diff);

		handler.apply_cutoff = this->apply_cutoff();

		// Allocate pab matrix for re-use across tasks.
		handler.pab().resize(this->ctx_.maxco(), this->ctx_.maxco());
		handler.work().resize(this->ctx_.maxco(), this->ctx_.maxco());
		handler.pab_prep().resize(this->ctx_.maxco(), this->ctx_.maxco());

		handler.initialize_basis_vectors(grid.dh, grid.dh_inv);

		/* setup the grid parameters, window parameters (if the grid is split), etc
		 */

		handler.grid() = grid;

		for (int d = 0; d < 3; d++)
				handler.orthogonal[d] = handler.grid().orthogonal(d);

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
		for (int itask = 0; itask < this->ctx_.tasks_per_level(level); itask++) {
			// Define some convenient aliases.
				const task_info *task = this->ctx_.queues(level, itask);

			if (task->level != level) {
				printf("level %d, %d\n", task->level, level);
				abort();
			}
			/* the grid is divided over several ranks or not periodic */
			if (handler.grid().is_distributed()) {
				/* unfortunately the window where the gaussian should be added depends
				 * on the bonds. So I have to adjust the window all the time. */

					handler.grid().setup_grid_window(task->border_mask);
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

			this->compute_coefficients(handler, prevtask_info, *task, &ctx_.pab_blocks(), handler.pab(),
																 handler.work(), handler.pab_prep());

			handler.collocate(this->orthorhombic_, *task);
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

void cpu_backend::rotate_and_store_coefficients(const task_info *prev_task,
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
		const int ikind = prev_task->ikind;
		const int jkind = prev_task->jkind;
		const grid_basis_set *ibasis = this->ctx_.basis_sets_[ikind];
		const grid_basis_set *jbasis = this->ctx_.basis_sets_[jkind];

		const int block_num = prev_task->block_num;
		double *const block = &blocks[this->ctx_.block_offsets_[block_num]];

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
		const int ikind = task->ikind;
		const int jkind = task->jkind;
		const int iset = task->iset;
		const int jset = task->jset;
		const grid_basis_set *ibasis = this->ctx_.basis_sets_[ikind];
		const grid_basis_set *jbasis = this->ctx_.basis_sets_[jkind];
		const int ncoseta = ncoset(ibasis->lmax[iset]);
		const int ncosetb = ncoset(jbasis->lmax[jset]);

		const int ncoa = ibasis->npgf[iset] * ncoseta;
		const int ncob = jbasis->npgf[jset] * ncosetb;

		hab.resize(ncob, ncoa);
	}
}


void cpu_backend::extract_blocks(const task_info &task,
																	const grid_buffer *pab_blocks,
																	tensor1<double, 2> &work,
																	tensor1<double, 2> &pab) {
		const int iatom = task.iatom;
		const int jatom = task.jatom;
		const int iset = task.iset;
		const int jset = task.jset;
		const int ikind = task.ikind;
		const int jkind = task.jkind;
		const grid_basis_set *ibasis = this->ctx_.basis_sets_[ikind];
		const grid_basis_set *jbasis = this->ctx_.basis_sets_[jkind];

		const int block_num = task.block_num;

		// Locate current matrix block within the buffer. This block
		// contains the weights of the gaussian pairs in the spherical
		// harmonic basis, but we do computation in the cartesian
		// harmonic basis so we have to rotate the coefficients. It is nothing
		// else than a basis change and it done with two dgemm.

		const int block_offset = this->ctx_.block_offsets_[block_num]; // zero based
		double *const block = &pab_blocks->host_buffer[block_offset];

		rotate_to_cartesian_harmonics(ibasis, jbasis, iatom, jatom, iset, jset, block,
																	work, pab);
}

void cpu_backend::compute_coefficients(cpu_handler &handler,
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

void cpu_backend::collocate()
{
		const int max_threads = omp_get_max_threads();
		if (this->scratch == NULL) {
				int max_size = ctx_.grid(0).size();

				/* compute the size of the largest grid. It is used afterwards to allocate
				 * scratch memory for the grid on each omp thread */
				for (int x = 1; x < (int)ctx_.grid().size(); x++) {
						max_size = std::max(ctx_.grid(x).size(), max_size);
				}

				max_size = ((max_size / 4096) + (max_size % 4096 != 0)) * 4096;

				/* scratch is a void pointer !!!!! */
				this->scratch =
						(double *)grid_allocate_scratch(max_size * max_threads * sizeof(double));
		}


		// std::setprecision(15);
		for (int level = 0; level < (int)ctx_.grid().size(); level++) {
				this->collocate_one_grid_level(level);
				// std::cout << ctx->grid[level].integrate() << "\n";
		}

		grid_free_scratch(scratch);
		scratch = NULL;
}

void cpu_backend::integrate() {
		// Zero result arrays.
		memset(ctx_.hab_blocks().host_buffer, 0, ctx_.hab_blocks().size);

		const int max_threads = omp_get_max_threads();

		if (this->scratch == NULL)
				this->scratch = static_cast<double *>(malloc(ctx_.hab_blocks().size * max_threads));

		this->orthorhombic_ = ctx_.is_orthorhombic();

		for (int level = 0; level < (int)ctx_.grid().size(); level++) {
				this->integrate_one_grid_level(level);
		}
		free(this->scratch);
		this->scratch = NULL;
}
