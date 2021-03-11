/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#include <cassert>
#include <limits.h>
#include <cmath>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

extern"C" {
#include "../common/grid_basis_set.h"
#include "../common/grid_constants.h"
}
#include "../common/grid_info.hpp"
#include "../common/task.hpp"

#include "grid_context_cpu.hpp"
#include "cpu_handler.hpp"


/* compute the functions (x - x_i)^l exp (-eta (x - x_i)^2) for l = 0..lp using
 * a recursive relation to avoid computing the exponential on each grid point. I
 * think it is not really necessary anymore since it is *not* the dominating
 * contribution to computation of collocate and integrate */

void cpu_handler::calculate_polynomials(const bool transpose, const double dr,
																				const double roffset, const int pol_offset,
																				const int xmin, const int xmax, const int lp,
																				const int cmax, const double zetp, double *pol_)
{
	const double t_exp_1 = exp(-zetp * dr * dr);
	const double t_exp_2 = t_exp_1 * t_exp_1;

	double t_exp_min_1 = exp(-zetp * (dr - roffset) * (dr - roffset));
	double t_exp_min_2 = exp(2.0 * zetp * (dr - roffset) * dr);

	double t_exp_plus_1 = exp(-zetp * roffset * roffset);
	double t_exp_plus_2 = exp(2.0 * zetp * roffset * dr);

	if (transpose) {
			tensor1<double, 2> pol(pol_, cmax, lp + 1);
			/* It is original Ole code. I need to transpose the polynomials for the
			 * integration routine and Ole code already does it. */
			for (int ig = 0; ig >= xmin; ig--) {
					const double rpg = ig * dr - roffset;
					t_exp_min_1 *= t_exp_min_2 * t_exp_1;
					t_exp_min_2 *= t_exp_2;
					double pg = t_exp_min_1;
					for (int icoef = 0; icoef <= lp; icoef++) {
							pol(pol_offset + ig - xmin, icoef) = pg;
							pg *= rpg;
					}
			}

		double t_exp_plus_1 = exp(-zetp * roffset * roffset);
		double t_exp_plus_2 = exp(2 * zetp * roffset * dr);
		for (int ig = 1; ig <= xmax; ig++) {
			const double rpg = ig * dr - roffset;
			t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
			t_exp_plus_2 *= t_exp_2;
			double pg = t_exp_plus_1;
			for (int icoef = 0; icoef <= lp; icoef++) {
					pol(pol_offset + ig - xmin, icoef) = pg;
				pg *= rpg;
			}
		}

	} else {
			tensor1<double, 2> pol(pol_, lp + 1, cmax);
			/* memset(pol.data, 0, sizeof(double) * pol.alloc_size_); */
			/*
			 *   compute the values of all (x-xp)**lp*exp(..)
			 *
			 *  still requires the old trick:
			 *  new trick to avoid to many exps (reuse the result from the previous
			 * gridpoint): exp( -a*(x+d)**2)=exp(-a*x**2)*exp(-2*a*x*d)*exp(-a*d**2)
			 *  exp(-2*a*(x+d)*d)=exp(-2*a*x*d)*exp(-2*a*d**2)
			 */

			/* compute the exponential recursively and store the polynomial prefactors
			 * as well */
			for (int ig = 0; ig >= xmin; ig--) {
					const double rpg = ig * dr - roffset;
					t_exp_min_1 *= t_exp_min_2 * t_exp_1;
					t_exp_min_2 *= t_exp_2;
					double pg = t_exp_min_1;
					pol(0, pol_offset + ig - xmin) = pg;
					if (lp > 0)
							pol(1, pol_offset + ig - xmin) = rpg;
		}

		for (int ig = 1; ig <= xmax; ig++) {
			const double rpg = ig * dr - roffset;
			t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
			t_exp_plus_2 *= t_exp_2;
			double pg = t_exp_plus_1;
			pol(0, pol_offset + ig - xmin) = pg;
			if (lp > 0)
					pol(1, pol_offset + ig - xmin) = rpg;
		}

		/* compute the remaining powers using previously computed stuff */
		if (lp >= 2) {
				double *__restrict__ poly = pol.at(1, 0);
			double *__restrict__ src1 = pol.at(0, 0);
			double *__restrict__ dst = pol.at(2, 0);
//#pragma omp simd
#pragma GCC ivdep
			for (int ig = 0; ig < (xmax - xmin + 1 + pol_offset); ig++)
				dst[ig] = src1[ig] * poly[ig] * poly[ig];
		}

		for (int icoef = 3; icoef <= lp; icoef++) {
				const double *__restrict__ poly = pol.at(1, 0);
				const double *__restrict__ src1 = pol.at(icoef - 1, 0);
				double *__restrict__ dst = pol.at(icoef, 0);
//#pragma omp simd
#pragma GCC ivdep
			for (int ig = 0; ig < (xmax - xmin + 1 + pol_offset); ig++) {
				dst[ig] = src1[ig] * poly[ig];
			}
		}

		//
		if (lp > 0) {
				double *__restrict__ dst = pol.at(1, 0);
				const double *__restrict__ src = pol.at(0, 0);
#pragma GCC ivdep
			for (int ig = 0; ig < (xmax - xmin + 1 + pol_offset); ig++) {
				dst[ig] *= src[ig];
			}
		}
	}
}

void cpu_handler::collocate_l0(const double alpha, tensor1<double, 3> &cube) {
		const double *__restrict__ pz =
			pol_.at(0, 0, 0); /* k indice */
	const double *__restrict__ py =
			pol_.at(1, 0, 0); /* j indice */
	const double *__restrict__ px =
			pol_.at(2, 0, 0); /* i indice */

	tensor1<double, 2> exp_xy(Exp_.at(2, 0, 0), Exp_.size(1), Exp_.size(2));

	if (this->scratch_size_ < cube.size(1) * cube.ld()) {
			this->scratch_size_ = cube.size(1) * cube.ld();
			this->scratch_size_ = ((this->scratch_size_ >> 3) + 1) << 3;
			this->scratch = static_cast<double *>(realloc(static_cast<void*>(this->scratch),
																										sizeof(double) * this->scratch_size_));
	}

	memset(this->scratch, 0, sizeof(double) * cube.size(1) * cube.ld());

	cblas_dger(CblasRowMajor, cube.size(1), cube.size(2), alpha, py, 1, px, 1,
						 scratch, cube.ld());

	if (!orthogonal[2]) {
			for (int y = 0; y < (int)cube.size(1); y++) {
					const double *__restrict__ src = exp_xy.at(y, 0);
					double *__restrict__ dst = &scratch[y * cube.ld()];

#pragma GCC ivdep
					for (auto x = 0; x < cube.size(2); x++) {
							dst[x] *= src[x];
					}
			}
	}

	cblas_dger(CblasRowMajor, cube.size(0), cube.size(1) * cube.ld(), 1.0, pz,
						 1, scratch, 1, cube.at(), cube.size(1) * cube.ld());
}

/* compute the following operation (variant) it is a tensor contraction

	 V_{kji} = \sum_{\alpha\beta\gamma} C_{\alpha\gamma\beta} T_{2,\alpha,i}
	 T_{1,\beta,j} T_{0,\gamma,k}

*/
void cpu_handler::tensor_reduction(const bool integrate,
																							 const double alpha,
																							 tensor1<double, 3> &co,
																							 tensor1<double, 3> &cube) {
		if (co.size(0) > 1) {
		dgemm_params m1, m2, m3;

		memset(&m1, 0, sizeof(dgemm_params));
		memset(&m2, 0, sizeof(dgemm_params));
		memset(&m3, 0, sizeof(dgemm_params));
		this->T.resize(co.size(0) /* alpha */, co.size(1) /* gamma */,
											cube.size(1) /* j */);
		this->W.resize(co.size(1) /* gamma */, cube.size(1) /* j */,
											cube.size(2) /* i */);


		// initialize_tensor_3(&T, co.size(0) /* alpha */, co->size[1] /* gamma */,
		//										cube.size(1) /* j */);
		// initialize_tensor_3(&W, co->size[1] /* gamma */, cube.size(1) /* j */,
		//										cube.size(2) /* i */);

		// T.data = scratch;
		// W.data = scratch + T.alloc_size_;
		/* WARNING we are in row major layout. cblas allows it and it is more
		 * natural to read left to right than top to bottom
		 *
		 * we do first T_{\alpha,\gamma,j} = \sum_beta C_{alpha\gamma\beta}
		 * Y_{\beta, j}
		 *
		 * keep in mind that Y_{\beta, j} = p_alpha_beta_reduced(1, \beta, j)
		 * and the order of indices is also important. the last indice is the
		 * fastest one. it can be done with one dgemm.
		 */

		m1.op1 = 'N';
		m1.op2 = 'N';
		m1.alpha = alpha;
		m1.beta = 0.0;
		m1.m = co.size(0) * co.size(1); /* alpha gamma */
		m1.n = cube.size(1);             /* j */
		m1.k = co.size(2);               /* beta */
		m1.a = co.at();                  // Coef_{alpha,gamma,beta} Coef_xzy
		m1.lda = co.ld();
		m1.b = pol_.at(1, 0, 0); // Y_{beta, j} = p_alpha_beta_reduced(1, beta, j)
		m1.ldb = pol_.ld();
		m1.c = this->T.at(); // T_{\alpha, \gamma, j} = T(alpha, gamma, j)
		m1.ldc = this->T.ld();
		m1.use_libxsmm = true;
		/*
		 * the next step is a reduction along the alpha index.
		 *
		 * We compute then
		 *
		 * W_{gamma, j, i} = sum_{\alpha} T_{\gamma, j, alpha} X_{\alpha, i}
		 *
		 * which means we need to transpose T_{\alpha, \gamma, j} to get
		 * T_{\gamma, j, \alpha}. Fortunately we can do it while doing the
		 * matrix - matrix multiplication
		 */

		m2.op1 = 'T';
		m2.op2 = 'N';
		m2.alpha = 1.0;
		m2.beta = 0.0;
		m2.m = cube.size(1) * co.size(1); // (\gamma j) direction
		m2.n = cube.size(2);               // i
		m2.k = co.size(0);                 // alpha
		m2.a = this->T.at();                      // T_{\alpha, \gamma, j}
		m2.lda = this->T.ld() * co.size(1);
		m2.b = pol_.at(2, 0, 0); // X_{alpha, i}  = p_alpha_beta_reduced(0, alpha, i)
		m2.ldb = pol_.ld();
		m2.c = this->W.at(); // W_{\gamma, j, i}
		m2.ldc = this->W.ld();
		m2.use_libxsmm = true;
		/* the final step is again a reduction along the gamma indice. It can
		 * again be done with one dgemm. The operation is simply
		 *
		 * Cube_{k, j, i} = \sum_{alpha} Z_{k, \gamma} W_{\gamma, j, i}
		 *
		 * which means we need to transpose Z_{\gamma, k}.
		 */

		m3.op1 = 'T';
		m3.op2 = 'N';
		m3.alpha = alpha;
		m3.beta = 0.0;
		m3.m = cube.size(0);                 // Z_{k \gamma}
		m3.n = cube.size(1) * cube.size(2); // (ji) direction
		m3.k = co.size(1);                   // \gamma
		m3.a = pol_.at(0, 0, 0); // p_alpha_beta_reduced(0, gamma, i)
		m3.lda = pol_.ld();
		m3.b = this->W.at(); // W_{\gamma, j, i}
		m3.ldb = this->W.size(1) * this->W.ld();
		m3.c = cube.at(); // cube_{kji}
		m3.ldc = cube.ld() * cube.size(1);
		m3.use_libxsmm = true;
		dgemm_simplified(&m1);
		dgemm_simplified(&m2);

		// apply the non orthorombic corrections in the xy plane
		if (!orthogonal[2] && !integrate) {
				tensor1<double, 2> exp_xy(Exp_.at(2, 0, 0), Exp_.size(1), Exp_.size(2));
				this->apply_non_orthorombic_corrections_xy_blocked(exp_xy, this->W);
		}

		dgemm_simplified(&m3);
		} else {
				if (!integrate) {
						cube.zero();
						this->collocate_l0(co(0, 0, 0) * alpha, cube);
				}
		}

		if (!integrate) {
				if (!orthogonal[0] && !orthogonal[1]) {
						tensor1<double, 2> exp_xz(Exp_.at(0, 0, 0), Exp_.size(1), Exp_.size(2));
						tensor1<double, 2> exp_yz(Exp_.at(1, 0, 0), Exp_.size(1), Exp_.size(2));
						this->apply_non_orthorombic_corrections_xz_yz_blocked(exp_xz, exp_yz, cube);
						return;
				}

				if (!orthogonal[0]) {
						tensor1<double, 2> exp_xy(Exp_.at(0, 0, 0), Exp_.size(1), Exp_.size(2));
						this->apply_non_orthorombic_corrections_xz_blocked(exp_xy, cube);
				}

				if (!orthogonal[1]) {
						tensor1<double, 2> exp_xy(Exp_.at(1, 0, 0), Exp_.size(1), Exp_.size(2));
						this->apply_non_orthorombic_corrections_yz_blocked(exp_xy, cube);
				}
	}

	return;
}

// *****************************************************************************
void cpu_handler::collocate(const bool use_ortho,	const task_info &task) {
	// *** position of the gaussian product
	//
	// this is the actual definition of the position on the grid
	// i.e. a point rp(:) gets here grid coordinates
	// MODULO(rp(:)/dr(:),npts(:))+1
	// hence (0.0,0.0,0.0) in real space is rsgrid%lb on the rsgrid ((1,1,1) on
	// grid)

	// cubecenter(:) = FLOOR(MATMUL(dh_inv, rp))
	/* cube : grid containing pointlike product between polynomials
	 *
	 * pol : grid  containing the polynomials in all three directions
	 *
	 * pol_folded : grid containing the polynomials after folding for periodic
	 * boundaries conditions
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
	this->pol_.resize(3, this->coef_.size(0), this->cmax_);
	//this->pol_.zero();
	/* compute the polynomials */

	// WARNING : do not reverse the order in pol otherwise you will have to
	// reverse the order in collocate_dgemm as well.

	if (use_ortho) {
			for (int i = 0; i < 3; i++)
					this->calculate_polynomials(false, this->dh[2 - i][2 - i], this->roffset_[i], 0,
																				this->lb_cube_[i], this->ub_cube_[i], this->coef_.size(i) - 1, this->cmax_,
																				task.zetp, this->pol_.at(i, 0, 0)); /* i indice */
	} else {
			for (int i = 0; i < 3; i++)
					this->calculate_polynomials(false, 1.0, this->roffset_[i], 0, this->lb_cube_[i],
																			this->ub_cube_[i], this->coef_.size(i) - 1, this->cmax_,
																			task.zetp * this->dx[i],
																			this->pol_.at(i, 0, 0)); /* k indice */

			this->calculate_non_orthorombic_corrections_tensor(task.zetp);

		/* Use a slightly modified version of Ole code */
		this->transform_coef_xzy_to_ikj(this->dh, this->coef_);
	}

	/* allocate memory for the polynomial and the cube */

	this->cube_.resize(this->cube_size_[0], this->cube_size_[1], this->cube_size_[2]);

	this->tensor_reduction(false,
												 1.0,
												 this->coef_,
												 this->cube_);

	if (this->apply_cutoff) {
		if (use_ortho) {
				this->apply_spherical_cutoff_ortho<true>();
		} else {
				this->apply_spherical_cutoff_generic<true>(task.radius);
		}
		return;
	}

	this->extract_add_cube<true>();
}
