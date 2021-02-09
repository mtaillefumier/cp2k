/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cpu_handler.hpp"
#include "utils.hpp"

static double exp_recursive(const double c_exp, const double c_exp_minus_1,
														const int index) {
		if (index == -1)
				return c_exp_minus_1;

		if (index == 1)
				return c_exp;

		double res = 1.0;

		if (index < 0) {
		for (int i = 0; i < -index; i++) {
			res *= c_exp_minus_1;
		}
		return res;
	}

	if (index > 0) {
		for (int i = 0; i < index; i++) {
			res *= c_exp;
		}
		return res;
	}

	return 1.0;
}

static void exp_i(const double alpha, const int imin, const int imax,
					 double *__restrict__ const res) {
		const double c_exp_co = std::exp(alpha);
	/* const double c_exp_minus_1 = 1/ c_exp; */
		res[0] = std::exp(imin * alpha);
		for (int i = 1; i < (imax - imin); i++) {
				res[i] = res[i - 1] *
						c_exp_co; // exp_recursive(c_exp_co, 1.0 / c_exp_co, i + imin);
		}
}

static void exp_ij(const double alpha, const int offset_i, const int imin,
						const int imax, const int offset_j, const int jmin, const int jmax,
						tensor1<double, 2> &exp_ij_) {
		double c_exp = std::exp(alpha * imin);
		const double c_exp_co =  std::exp(alpha);

		for (int i = 0; i < (imax - imin); i++) {
				double *__restrict__ dst = exp_ij_.at(i + offset_i, offset_j);
				double ctmp = exp_recursive(c_exp, 1.0 / c_exp, jmin);

#pragma GCC ivdep
				for (int j = 0; j < (jmax - jmin); j++) {
						dst[j] *= ctmp;
						ctmp *= c_exp;
				}
				c_exp *= c_exp_co;
		}
}

void cpu_handler::calculate_non_orthorombic_corrections_tensor(const double mu_mean) {
		// zx, zy, yx
		const int n[3][2] = {{0, 2}, {0, 1}, {1, 2}};

		// need to review this
		const double c[3] = {
				/* alpha gamma */
				-2.0 * mu_mean *
				(dh[0][0] * dh[2][0] + dh[0][1] * dh[2][1] + dh[0][2] * dh[2][2]),
				/* beta gamma */
				-2.0 * mu_mean *
				(dh[1][0] * dh[2][0] + dh[1][1] * dh[2][1] + dh[1][2] * dh[2][2]),
				/* alpha beta */
				-2.0 * mu_mean *
				(dh[0][0] * dh[1][0] + dh[0][1] * dh[1][1] + dh[0][2] * dh[1][2])
		};

		/* a naive implementation of the computation of exp(-2 (v_i . v_j) (i
		 * - r_i) (j _ r_j)) requires n m exponentials but we can do much much
		 * better with only 7 exponentials
		 *
		 * first expand it. we get exp(2 (v_i . v_j) i j) exp(2 (v_i . v_j) i r_j)
		 * exp(2 (v_i . v_j) j r_i) exp(2 (v_i . v_j) r_i r_j). we can use the fact
		 * that the sum of two terms in an exponential is equal to the product of
		 * the two exponentials.
		 *
		 * this means that exp (a i) with i integer can be computed recursively with
		 * one exponential only
		 */

		/* we have a orthorombic case */
		if (orthogonal[0] && orthogonal[1] && orthogonal[2])
				return;

		double *x1, *x2;
		/* printf("%d %d %d\n", plane[0], plane[1], plane[2]); */
		const int max_elem =
				std::max(std::max(ub_cube_[0] - lb_cube_[0], ub_cube_[1] - lb_cube_[1]), ub_cube_[2] - lb_cube_[2]) + 1;
		Exp_.resize(3, max_elem, max_elem);
		Exp_.zero();
		x1 = (double *) malloc(sizeof(double) * max_elem);
		x2 = (double *) malloc(sizeof(double) * max_elem);

		for (int dir = 0; dir < 3; dir++) {
				int d1 = n[dir][0];
				int d2 = n[dir][1];

				if (!orthogonal[dir]) {
						const double c_exp_const = std::exp(c[dir] * roffset_[d1] * roffset_[d2]);
						tensor1<double, 2> exp_tmp(Exp_.at(dir, 0, 0), Exp_.size(1), Exp_.size(2));

						exp_i(-roffset_[d2] * c[dir], lb_cube_[d1], ub_cube_[d1] + 1, x1);
						exp_i(-roffset_[d1] * c[dir], lb_cube_[d2], ub_cube_[d2] + 1, x2);

						cblas_dger(CblasRowMajor, ub_cube_[d1] - lb_cube_[d1] + 1,
											 ub_cube_[d2] - lb_cube_[d2] + 1, c_exp_const, x1, 1, x2, 1,
											 exp_tmp.at(0, 0), exp_tmp.ld());
						exp_ij(c[dir], 0, lb_cube_[d1], ub_cube_[d1] + 1, 0, lb_cube_[d2], ub_cube_[d2] + 1,
									 exp_tmp);
				}
		}
		free(x1);
		free(x2);
}


void cpu_handler::apply_non_orthorombic_corrections(const bool *__restrict__ plane,
																																const tensor1<double, 3> &Exp,
																																tensor1<double, 3> &cube) {
	// Well we should never call non orthorombic corrections if everything is
	// orthorombic
	if (plane[0] && plane[1] && plane[2])
		return;

	/*k and i are orthogonal, k and j as well */
	if (plane[0] && plane[1]) {
			for (int z = 0; z < cube.size(0); z++) {
			for (int y = 0; y < cube.size(1); y++) {
					const double *__restrict__ yx = Exp.at(2, y, 0);
				double *__restrict__ dst = cube.at(z, y, 0);

				for (int x = 0; x < cube.size(2); x++) {
					dst[x] *= yx[x];
				}
			}
		}
		return;
	}

	/* k and i are orhogonal, i and j as well */
	if (plane[0] && plane[2]) {
		for (int z = 0; z < cube.size(0); z++) {
			for (int y = 0; y < cube.size(1); y++) {
					const double zy = Exp(1, z, y);
				double *__restrict__ dst = cube.at(z, y, 0);

				for (int x = 0; x < cube.size(2); x++) {
					dst[x] *= zy;
				}
			}
		}
		return;
	}

	/* j, k are orthognal, i and j are orthognal */
	if (plane[1] && plane[2]) {
		for (int z = 0; z < cube.size(0); z++) {
				const double *__restrict__ zx = Exp.at(0, z, 0);
			for (int y = 0; y < cube.size(1); y++) {
					double *__restrict__ dst = cube.at(z, y, 0);

				for (int x = 0; x < cube.size(2); x++) {
					dst[x] *= zx[x];
				}
			}
		}
		return;
	}

	if (plane[0]) {
		// z perpendicular to x. but y non perpendicular to any
		for (int z = 0; z < cube.size(0); z++) {
			for (int y = 0; y < cube.size(1); y++) {
					const double zy = Exp(1, z, y);
				const double *__restrict__ yx = Exp.at(2, y, 0);
				double *__restrict__ dst = cube.at(z, y, 0);

				for (int x = 0; x < cube.size(2); x++) {
					dst[x] *= zy * yx[x];
				}
			}
		}
		return;
	}

	if (plane[1]) {
		// z perpendicular to y, but x and z are not and y and x neither
		for (int z = 0; z < cube.size(0); z++) {
				const double *__restrict__ zx = Exp.at(0, z, 0);
			for (int y = 0; y < cube.size(1); y++) {
					const double *__restrict__ yx = Exp.at(2, y, 0);
				double *__restrict__ dst = cube.at(z, y, 0);

				for (int x = 0; x < cube.size(2); x++) {
					dst[x] *= zx[x] * yx[x];
				}
			}
		}
		return;
	}

	if (plane[2]) {
		// x perpendicular to y, but x and z are not and y and z neither
		for (int z = 0; z < cube.size(0); z++) {
				const double *__restrict__ zx = Exp.at(0, z, 0);
			for (int y = 0; y < cube.size(1); y++) {
					const double zy = Exp(1, z, y);
				double *__restrict__ dst = cube.at(z, y, 0);

				for (int x = 0; x < cube.size(2); x++) {
					dst[x] *= zx[x] * zy;
				}
			}
		}
		return;
	}

	/* generic  case */

	for (int z = 0; z < cube.size(0); z++) {
			const double *__restrict__ zx = Exp.at(0, z, 0);
		for (int y = 0; y < cube.size(1); y++) {
				const double zy = Exp(1, z, y);
			const double *__restrict__ yx = Exp.at(2, y, 0);
			double *__restrict__ dst = cube.at(z, y, 0);

			for (int x = 0; x < cube.size(2); x++) {
				dst[x] *= zx[x] * zy * yx[x];
			}
		}
	}
	return;
}

void cpu_handler::apply_non_orthorombic_corrections_xy_blocked(
		const tensor1<double, 2> &Exp,  tensor1<double, 3> &m) {
		for (int gamma = 0; gamma < m.size(0); gamma++) {
				for (int y1 = 0; y1 < m.size(1); y1++) {
						double *__restrict__ dst = m.at(gamma, y1, 0);
						const double *__restrict__ src = Exp.at(y1, 0);
#pragma GCC ivdep
						for (int x1 = 0; x1 < m.size(2); x1++) {
								dst[x1] *= src[x1];
						}
				}
		}
}

void cpu_handler::apply_non_orthorombic_corrections_xz_blocked(
		const tensor1<double, 2> &Exp, tensor1<double, 3> &m) {
	for (int z1 = 0; z1 < m.size(0); z1++) {
			const double *__restrict__ src = Exp.at(z1, 0);
		for (int y1 = 0; y1 < m.size(1); y1++) {
				double *__restrict__ dst = m.at(z1, y1, 0);
#pragma GCC ivdep
			for (int x1 = 0; x1 < m.size(2); x1++) {
				dst[x1] *= src[x1];
			}
		}
	}
}

void cpu_handler::apply_non_orthorombic_corrections_yz_blocked(
		const tensor1<double, 2> &Exp, tensor1<double, 3> &m) {
	for (int z1 = 0; z1 < m.size(0); z1++) {
		for (int y1 = 0; y1 < m.size(1); y1++) {
				const double src = Exp(z1, y1);
			double *__restrict__ dst = m.at(z1, y1, 0);
#pragma GCC ivdep
			for (int x1 = 0; x1 < m.size(2); x1++) {
				dst[x1] *= src;
			}
		}
	}
}

void cpu_handler::apply_non_orthorombic_corrections_xz_yz_blocked(
		const tensor1<double, 2> &Exp_xz, const tensor1<double, 2> &Exp_yz,
		tensor1<double, 3> &m) {
	for (int z1 = 0; z1 < m.size(0); z1++) {
			const double *__restrict__ src_xz = Exp_xz.at(z1, 0);
		for (int y1 = 0; y1 < m.size(1); y1++) {
				const double src = Exp_yz(z1, y1);
			double *__restrict__ dst = m.at(z1, y1, 0);
#pragma GCC ivdep
			for (int x1 = 0; x1 < m.size(2); x1++) {
				dst[x1] *= src * src_xz[x1];
			}
		}
	}
}
