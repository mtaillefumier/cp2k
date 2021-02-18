/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <omp.h>

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

extern "C" {
#include "../common/grid_common.h"
}

#include "../common/tensor.hpp"
#include "../common/task.hpp"
#include "utils.hpp"
#include "../common/grid_info.hpp"
#include "grid_context_cpu.hpp"
#include "cpu_handler.hpp"


void update_force_pair(orbital a, orbital b, const double pab,
											 const double ftz[2], const double *const rab,
											 const tensor1<double, 2> &vab, tensor1<double, 2> &force) {
		const double axpm0 = vab(idx(b), idx(a));
	for (int i = 0; i < 3; i++) {
			const double aip1 = vab(idx(b), idx(up(i, a)));
			const double aim1 = vab(idx(b), idx(down(i, a)));
			const double bim1 = vab(idx(down(i, b)), idx(a));
			force(0, i) += pab * (ftz[0] * aip1 - a.l[i] * aim1);
			force(1, i) +=
				pab * (ftz[1] * (aip1 - rab[i] * axpm0) - b.l[i] * bim1);
	}
}

void update_virial_pair(orbital a, orbital b, const double pab,
												const double ftz[2], const double *const rab,
												const tensor1<double, 2> &vab, tensor1<double, 3> &virial) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
				virial(0, i, j) +=
						pab * ftz[0] * vab(idx(b), idx(up(i, up(j, a)))) -
						pab * a.l[j] * vab(idx(b), idx(up(i, down(j, a))));
		}
	}

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
				virial(1, i, j) +=
					pab * ftz[1] *
						(vab(idx(b), idx(up(i, up(j, a)))) -
						 vab(idx(b), idx(up(i, a))) * rab[j] -
						 vab(idx(b), idx(up(j, a))) * rab[i] +
						 vab(idx(b), idx(a)) * rab[j] * rab[i]) -
						pab * b.l[j] * vab(idx(up(i, down(j, b))), idx(a));
		}
	}
}

void update_all(const bool compute_forces, const bool compute_virial,
								const orbital a, const orbital b, const double f,
								const double *const ftz, const double *rab, const tensor1<double, 2> &vab,
								const double pab, double *hab, tensor1<double, 2> &forces,
								tensor1<double, 3> &virials) {

		*hab += f * vab(idx(b), idx(a));

		if (compute_forces) {
				update_force_pair(a, b, f * pab, ftz, rab, vab, forces);
		}

		if (compute_virial) {
				update_virial_pair(a, b, f * pab, ftz, rab, vab, virials);
	}
}

static void update_tau(const bool compute_forces, const bool compute_virial,
											 const orbital a, const orbital b, const double ftz[2],
											 const double *const rab, const tensor1<double, 2> &vab,
											 const double pab, double *const hab, tensor1<double, 2> &forces,
											 tensor1<double, 3> &virials) {

	for (int i = 0; i < 3; i++) {
		update_all(compute_forces, compute_virial, down(i, a), down(i, b),
							 0.5 * a.l[i] * b.l[i], ftz, rab, vab, pab, hab, forces, virials);
		update_all(compute_forces, compute_virial, up(i, a), down(i, b),
							 -0.5 * ftz[0] * b.l[i], ftz, rab, vab, pab, hab, forces,
							 virials);
		update_all(compute_forces, compute_virial, down(i, a), up(i, b),
							 -0.5 * a.l[i] * ftz[1], ftz, rab, vab, pab, hab, forces,
							 virials);
		update_all(compute_forces, compute_virial, up(i, a), up(i, b),
							 0.5 * ftz[0] * ftz[1], ftz, rab, vab, pab, hab, forces, virials);
	}
}

static void
update_hab_forces_and_stress(const task_info *task, const tensor1<double, 2> &vab,
														 const tensor1<double, 2> &pab, const bool compute_tau,
														 const bool compute_forces,
														 const bool compute_virial, tensor1<double, 2> &forces,
														 tensor1<double, 3> &virial, tensor1<double, 2> &hab) {
	double zeta[2] = {task->zeta[0] * 2.0, task->zeta[1] * 2.0};
	for (int lb = task->lmin[1]; lb <= task->lmax[1]; lb++) {
		for (int la = task->lmin[0]; la <= task->lmax[0]; la++) {
			for (int bx = 0; bx <= lb; bx++) {
				for (int by = 0; by <= lb - bx; by++) {
					const int bz = lb - bx - by;
					const orbital b = {{bx, by, bz}};
					const int idx_b = task->offset[1] + idx(b);
					for (int ax = 0; ax <= la; ax++) {
						for (int ay = 0; ay <= la - ax; ay++) {
							const int az = la - ax - ay;
							const orbital a = {{ax, ay, az}};
							const int idx_a = task->offset[0] + idx(a);
							double *habval = hab.at(idx_b, idx_a);
							const double prefactor = pab(idx_b, idx_a);

							// now compute the forces
							if (compute_tau) {
								update_tau(compute_forces, compute_virial, a, b, zeta,
													 task->rab, vab, prefactor, habval, forces, virial);
							} else {
								update_all(compute_forces, compute_virial, a, b, 1.0, zeta,
													 task->rab, vab, prefactor, habval, forces, virial);
							}
						}
					}
				}
			}
		}
	}
}

void cpu_handler::integrate(const bool use_ortho, const task_info &task, tensor1<double, 2> &vab)
{
		int lmax[2] = {task.lmax[0] + lmax_diff[0],
				task.lmax[1] + lmax_diff[1]};
		int lmin[2] = {task.lmin[0] + lmin_diff[0],
				task.lmin[1] + lmin_diff[1]};

		lmin[0] = std::max(lmin[0], 0);
		lmin[1] = std::max(lmin[1], 0);

		alpha_.resize(3, lmax[1] + 1, lmax[0] + 1,
									lmax[0] + lmax[1] + 1);

		const int lp = lmax[0] + lmax[1];

		coef_.resize(lp + 1, lp + 1, lp + 1);


	/* cube : grid comtaining pointlike product between polynomials
	 *
	 * pol : grid  containing the polynomials in all three directions
	 */

	/* seting up the cube parameters */
	this->cmax_ = compute_cube_properties(use_ortho,
																				task.radius,
																				this->dh,
																				this->dh_inv,
																				task.rp,
																				&this->disr_radius_,
																				this->roffset_,
																				this->cube_center_,
																				this->lb_cube_,
																				this->ub_cube_,
																				this->cube_size_);

	/* initialize the multidimensional array containing the polynomials */
	if (lp != 0) {
			this->pol_.resize(3, this->cmax_, this->coef_.size(0));
	} else {
			this->pol_.resize(3, this->coef_.size(0), this->cmax_);
	}

	// this->pol_.zero();

	/* allocate memory for the polynomial and the cube */
	this->cube_.resize(this->cube_size_[0], this->cube_size_[1], this->cube_size_[2]);

	/* compute the polynomials */

	// WARNING : do not reverse the order in pol otherwise you will have to
	// reverse the order in collocate_dgemm as well.

	/* the tensor contraction is done for a given order so I have to be careful
	 * how the tensors X, Y, Z are stored. In collocate, they are stored
	 * normally 0 (Z), (1) Y, (2) X in the table pol. but in integrate (which
	 * uses the same tensor reduction), I have a special treatment for l = 0. In
	 * that case the order *should* be the same than for collocate. For l > 0,
	 * we need a different storage order which is (X) 2, (Y) 0, and (Z) 1.
	 *
	 * the reason for this is that the cube is stored as cube[z][y][x] so the
	 * first direction taken for the dgemm is along x.
	 */

	int perm[3] = {2, 0, 1};

	if (lp == 0) {
		/* I need to restore the same order than for collocate */
		perm[0] = 0;
		perm[1] = 1;
		perm[2] = 2;
	}

	bool use_ortho_forced = this->orthogonal[0] && this->orthogonal[1] &&
													this->orthogonal[2];
	if (use_ortho) {
			for (int  d= 0; d < 3; d++)
					calculate_polynomials((lp != 0), this->dh[2 - d][2 - d], this->roffset_[d], 0, this->lb_cube_[d],
																this->ub_cube_[d], lp, this->cmax_, task.zetp,
																this->pol_.at(perm[d], 0, 0)); /* i indice */
	} else {
		double dx[3];

		dx[2] = this->dh[0][0] * this->dh[0][0] +
						this->dh[0][1] * this->dh[0][1] +
						this->dh[0][2] * this->dh[0][2];

		dx[1] = this->dh[1][0] * this->dh[1][0] +
						this->dh[1][1] * this->dh[1][1] +
						this->dh[1][2] * this->dh[1][2];

		dx[0] = this->dh[2][0] * this->dh[2][0] +
						this->dh[2][1] * this->dh[2][1] +
						this->dh[2][2] * this->dh[2][2];
		for (int  d= 0; d < 3; d++)
				calculate_polynomials((lp != 0), 1.0, this->roffset_[d], 0, this->lb_cube_[d], this->ub_cube_[d],
															lp, this->cmax_, task.zetp * dx[d],
															this->pol_.at(perm[d], 0, 0)); /* i indice */

		/* the three remaining tensors are initialized in the function */
		calculate_non_orthorombic_corrections_tensor(task.zetp);
	}

	/* extract the data from the grid and apply the cutoff eventually */
	if (this->apply_cutoff) {
			this->cube_.zero();
			if (!use_ortho && !use_ortho_forced) {
					this->apply_spherical_cutoff_ortho<false>();
			} else {
					this->apply_spherical_cutoff_generic<false>(task.radius);
			}
	} else {
			this->extract_add_cube<false>();
	}

	if (!use_ortho && !use_ortho_forced)
			this->apply_non_orthorombic_corrections();

	/* apply the tensor reduction. It is the *same* reduction as the collocate one */
	if (lp != 0) {
			this->tensor_reduction(true, 1.0, this->cube_, this->coef_);
	} else {
		/* it is very specific to integrate because we might end up with a
		 * single element after the tensor product/contraction. In that case, I
		 * compute the cube and then do a scalar product between the two. */

		/* we could also do this with 2 matrix-vector multiplications and a scalar
		 * product
		 *
		 * H_{jk} = C_{ijk} . P_i (along x) C_{ijk} is *stored C[k][j][i]* !!!!!!
		 * L_{k} = H_{jk} . P_j (along y)
		 * v_{ab} = L_k . P_k (along z)
		 */

			cube_tmp.resize(this->cube_size_[0], this->cube_size_[1]);

			/* first along x */
			cblas_dgemv(CblasRowMajor, CblasNoTrans,
									this->cube_.size(0) * this->cube_.size(1),
									this->cube_.size(2), 1.0, this->cube_.at(),
									this->cube_.ld(), this->pol_.at(2, 0, 0), 1, 0.0,
									cube_tmp.at(), 1);

			/* second along j */
			cblas_dgemv(CblasRowMajor, CblasNoTrans, this->cube_.size(0),
									this->cube_.size(1), 1.0, cube_tmp.at(), cube_tmp.ld(),
									this->pol_.at(1, 0, 0), 1, 0.0, this->scratch, 1);

			/* finally along k, it is a scalar product.... */
			this->coef_(0, 0, 0) = cblas_ddot(this->cube_.size(0), (double *)this->scratch,
																				1, this->pol_.at(0, 0, 0), 1);
			cube_tmp.clear();
	}

	/* go from ijk -> xyz */
	if (!use_ortho)
		transform_coef_jik_to_yxz(this->dh, this->coef_);

	/* compute the transformation matrix */
	this->prepare_alpha(task, lmax);

	vab.resize(ncoset(lmax[1]), ncoset(lmax[0]));
	this->compute_vab(lmin, lmax, lp, task.prefactor, vab); // contains the coefficients of the potential
}

void cpu_backend::integrate_one_grid_level(const int level) {
		grid_info &grid = ctx_.grid(level);
		auto &forces_ = ctx_.forces();
		auto &virial_ = ctx_.virial();
	// Using default(shared) because with GCC 9 the behavior around const changed:
	// https://www.gnu.org/software/gcc/gcc-9/porting_to.html
#pragma omp parallel default(shared)
	{
		const int num_threads = omp_get_num_threads();
		const int thread_id = omp_get_thread_num();

		double *hab_block_local = NULL;

		if (num_threads == 1) {
				hab_block_local = ctx_.hab_blocks().host_buffer;
		} else {
			hab_block_local =
					((double *)this->scratch) +
					thread_id * (ctx_.hab_blocks().size / sizeof(double));
			memset(hab_block_local, 0, ctx_.hab_blocks().size);
		}

		tensor1<double, 2> hab, vab, forces_local_, virial_local_,
				forces_local_pair_;
		tensor1<double, 3> virial_local_pair_;

		auto &handler = this->handler_[thread_id];
		handler.apply_cutoff = this->apply_cutoff();
		handler.lmax_diff[0] = 0;
		handler.lmax_diff[1] = 0;
		handler.lmin_diff[0] = 0;
		handler.lmin_diff[1] = 0;

		if (ctx_.calculate_tau() || ctx_.calculate_forces() || ctx_.calculate_virial()) {
			handler.lmax_diff[0] = 1;
			handler.lmax_diff[1] = 0;
			handler.lmin_diff[0] = -1;
			handler.lmin_diff[1] = -1;
		}

		if (ctx_.calculate_virial()) {
			handler.lmax_diff[0]++;
			handler.lmax_diff[1]++;
		}

		if (ctx_.calculate_tau()) {
			handler.lmax_diff[0]++;
			handler.lmax_diff[1]++;
			handler.lmin_diff[0]--;
			handler.lmin_diff[1]--;
		}

		// Allocate pab matrix for re-use across tasks.
		handler.pab().resize(this->ctx_.maxco(), this->ctx_.maxco());
		vab.resize(this->ctx_.maxco(), this->ctx_.maxco());
		handler.work().resize(this->ctx_.maxco(), this->ctx_.maxco());
		hab.resize(this->ctx_.maxco(), this->ctx_.maxco());

		if (ctx_.calculate_forces()) {
				forces_local_.resize(ctx_.forces().size(0), ctx_.forces().size(1));
				virial_local_.resize(3, 3);
				forces_local_pair_.resize(2, 3);
				virial_local_pair_.resize(2, 3, 3);
				forces_local_.zero();
				virial_local_.zero();
				forces_local_pair_.zero();
				virial_local_pair_.zero();
		}

		handler.initialize_basis_vectors(grid.dh, grid.dh_inv);

		handler.grid() = grid;

		for (int d = 0; d < 3; d++)
				handler.orthogonal[d] = grid.orthogonal(d);

		/* it is only useful when we split the list over multiple threads. The
		 * first iteration should load the block whatever status the
		 * task->block_update_ variable has */
		const task_info *prev_task = NULL;
#pragma omp for schedule(static)
		for (int itask = 0; itask < ctx_.tasks_per_level(level); itask++) {
			// Define some convenient aliases.
				const task_info *task = ctx_.queues(level, itask);

			if (task->level != level) {
				printf("level %d, %d\n", task->level, level);
				abort();
			}

			if (task->update_block_ || (prev_task == NULL)) {
				/* need to load pab if forces are needed */
					if (ctx_.calculate_forces()) {
						this->extract_blocks(*task, &ctx_.pab_blocks(), handler.work(), handler.pab());
				}
				/* store the coefficients of the operator after rotation to
				 * the spherical harmonic basis */

				this->rotate_and_store_coefficients(prev_task, task, hab, handler.work(),
																						hab_block_local);

				/* There is a difference here between collocate and integrate.
				 * For collocate, I do not need to know the task where blocks
				 * have been updated the last time. For integrate this
				 * information is crucial to update the coefficients of the
				 * potential */
				prev_task = task;
				hab.zero();
			}

			/* the grid is divided over several ranks or not periodic */
			if (handler.grid().is_distributed()) {
				/* unfortunately the window where the gaussian should be added depends
				 * on the bonds. So I have to adjust the window all the time. */

					handler.grid().setup_grid_window(task->border_mask);
			}

			handler.integrate(ctx_.is_orthorhombic(), *task, vab);

			// in the (x - x_1)(x - x_2) basis

			if (ctx_.calculate_forces()) {
					forces_local_pair_.zero();
					virial_local_pair_.zero();
			}

			update_hab_forces_and_stress(
					task, vab, handler.pab(), ctx_.calculate_tau(), ctx_.calculate_forces(), ctx_.calculate_virial(),
					forces_local_pair_, /* matrix
																* containing the
																* contribution of
																* the gaussian
																* pair for each
																* atom */
					virial_local_pair_, /* same but for the virial term (stress tensor)
																*/
					hab);

			if (ctx_.calculate_forces()) {
				const double scaling = (task->iatom == task->jatom) ? 1.0 : 2.0;
				forces_local_(task->iatom, 0) +=
						scaling * forces_local_pair_(0, 0);
				forces_local_(task->iatom, 1) +=
						scaling * forces_local_pair_(0, 1);
				forces_local_(task->iatom, 2) +=
						scaling * forces_local_pair_(0, 2);

				forces_local_(task->jatom, 0) +=
						scaling * forces_local_pair_(1, 0);
				forces_local_(task->jatom, 1) +=
						scaling * forces_local_pair_(1, 1);
				forces_local_(task->jatom, 2) +=
						scaling * forces_local_pair_(1, 2);
				if (ctx_.calculate_virial()) {
						for (int i = 0; i < 3; i++) {
								for (int j = 0; j < 3; j++) {
										virial_local_(i, j) +=
												scaling * (virial_local_pair_(0, i, j) +
																	 virial_local_pair_(1, i, j));
						}
					}
				}
			}
		}

		this->rotate_and_store_coefficients(prev_task, nullptr, hab, handler.work(),
																				hab_block_local);

		// now reduction over the hab blocks
		if (num_threads > 1) {
			// does not store the number of elements but the amount of memory
			// occupied. That's a strange choice.
				const int hab_size = ctx_.hab_blocks().size / sizeof(double);
			if ((hab_size / num_threads) >= 2) {
				const int block_size =
						hab_size / num_threads + (hab_size % num_threads);

				for (int bk = 0; bk < num_threads; bk++) {
					int bk_id = (bk + thread_id) % num_threads;
					size_t begin = bk_id * block_size;
					size_t end = std::min((bk_id + 1) * block_size, hab_size);
					cblas_daxpy(end - begin, 1.0, hab_block_local + begin, 1,
											ctx_.hab_blocks().host_buffer + begin, 1);
#pragma omp barrier
				}
			} else {
					const int hab_size = ctx_.hab_blocks().size / sizeof(double);
#pragma omp critical
					cblas_daxpy(hab_size, 1.0, hab_block_local, 1, ctx_.hab_blocks().host_buffer,
											1);
			}
		}

		if (ctx_.calculate_forces()) {
				if (num_threads > 1) {
						if ((forces_.size() / num_threads) >= 2) {
								const int block_size = forces_.size() / num_threads +
										(forces_.size() % num_threads);

								for (int bk = 0; bk < num_threads; bk++) {
										size_t bk_id = (bk + thread_id) % num_threads;
										size_t begin = bk_id * block_size;
										size_t end = std::min((bk_id + 1) * block_size, forces_.size());
										cblas_daxpy(end - begin, 1.0, forces_local_.at() + begin, 1,
																ctx_.forces().at() + begin, 1);
#pragma omp barrier
								}
						} else {
#pragma omp critical
								cblas_daxpy(forces_local_.size(), 1.0, forces_local_.at(), 1,
														ctx_.forces().at(), 1);
						}
				} else {
						memcpy(ctx_.forces().at(), forces_local_.at(), sizeof(double) * forces_local_.size());
				}

		}

		if (ctx_.calculate_virial()) {
#pragma omp critical
				for (int i = 0; i < 3; i++) {
						for (int j = 0; j < 3; j++) {
								virial_(i, j) += virial_local_(i, j);
						}
				}
		}

		vab.clear();
		hab.clear();
		forces_local_.clear();
		virial_local_.clear();
	}
}
