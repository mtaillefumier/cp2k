/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__LIBXSMM)
#include <libxsmm.h>
#endif

extern "C" {
		#include "../common/grid_common.h"
}

#include "cpu_handler.hpp"

void cpu_handler::transform_xyz_to_triangular(const tensor1<double, 3> &coef,
																							double *const coef_xyz) {
	assert(coef_xyz != NULL);

	int lxyz = 0;
	const int lp = (coef.size(0) - 1);
	for (int lzp = 0; lzp <= lp; lzp++) {
		for (int lyp = 0; lyp <= lp - lzp; lyp++) {
			for (int lxp = 0; lxp <= lp - lzp - lyp; lxp++, lxyz++) {
					coef_xyz[lxyz] = coef(lzp, lyp, lxp);
			}
		}
	}
}

void cpu_handler::transform_yxz_to_triangular(const tensor1<double, 3> &coef,
																 double *const coef_xyz) {
	assert(coef_xyz != NULL);
	int lxyz = 0;
	const int lp = (coef.size(0) - 1);
	for (int lzp = 0; lzp <= lp; lzp++) {
		for (int lyp = 0; lyp <= lp - lzp; lyp++) {
			for (int lxp = 0; lxp <= lp - lzp - lyp; lxp++, lxyz++) {
					coef_xyz[lxyz] = coef(lyp, lxp, lzp);
			}
		}
	}
}

void cpu_handler::transform_triangular_to_xyz(const double *const coef_xyz,
																 tensor1<double, 3> &coef) {
	assert(coef_xyz != NULL);
	int lxyz = 0;
	const int lp = coef.size(0) - 1;
	for (int lzp = 0; lzp <= lp; lzp++) {
		for (int lyp = 0; lyp <= lp - lzp; lyp++) {
			for (int lxp = 0; lxp <= lp - lzp - lyp; lxp++, lxyz++) {
					coef(lzp, lyp, lxp) = coef_xyz[lxyz];
			}
			/* initialize the remaining coefficients to zero */
			for (int lxp = lp - lzp - lyp + 1; lxp <= lp; lxp++)
					coef(lzp, lyp, lxp) = 0.0;
		}
	}
}

/* Rotate from the (x - x_1) ^ alpha_1 (x - x_2) ^ \alpha_2 to (x - x_{12}) ^ k
 * in all three directions */

void cpu_handler::compute_coefficients(
		const int *lmin, const int *lmax, const int lp, const double prefactor,
		const tensor1<double, 2> &pab)
{
	/* can be done with dgemms as well, since it is a change of basis from (x -
	 * x1) (x - x2) to (x - x12)^alpha */

	coef_.zero();
	// we need a proper fix for that. We can use the tensor structure for this

	for (int lzb = 0; lzb <= lmax[1]; lzb++) {
		for (int lyb = 0; lyb <= lmax[1] - lzb; lyb++) {
			const int lxb_min = imax(lmin[1] - lzb - lyb, 0);
			for (int lxb = lxb_min; lxb <= lmax[1] - lzb - lyb; lxb++) {
				const int jco = coset(lxb, lyb, lzb);
				for (int lza = 0; lza <= lmax[0]; lza++) {
					for (int lya = 0; lya <= lmax[0] - lza; lya++) {
						const int lxa_min = imax(lmin[0] - lza - lya, 0);
						for (int lxa = lxa_min; lxa <= lmax[0] - lza - lya; lxa++) {
							const int ico = coset(lxa, lya, lza);
							const double pab_ = pab(jco, ico);
							for (int lxp = 0; lxp <= lxa + lxb; lxp++) {
								const double p1 =
										alpha_(0, lxb, lxa, lxp) * pab_ * prefactor;
								for (int lzp = 0; lzp <= lp - lxa - lxb; lzp++) {
									for (int lyp = 0; lyp <= lp - lxa - lxb - lzp; lyp++) {
											const double p2 = alpha_(1, lyb, lya, lyp) *
													alpha_(2, lzb, lza, lzp) * p1;
											coef_(lxp, lzp, lyp) += p2;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

/* Rotate from (x - x_{12}) ^ k to (x - x_1) ^ alpha_1 (x - x_2) ^ \alpha_2 in
 * all three directions */

void cpu_handler::compute_vab(const int *const lmin,
																					const int *const lmax,
																					const int lp,
																					const double prefactor, // transformation parameters (x - x_1)^n (x -
																					// x_2)^m -> (x - x_12) ^ l
																					tensor1<double, 2> &vab)
{
		/* can be done with dgemms as well, since it is a change of basis from (x -
		 * x1) (x - x2) to (x - x12)^alpha */

		vab.zero();
		// we need a proper fix for that. We can use the tensor structure for this

		for (int lzb = 0; lzb <= lmax[1]; lzb++) {
				for (int lyb = 0; lyb <= lmax[1] - lzb; lyb++) {
						const int lxb_min = imax(lmin[1] - lzb - lyb, 0);
						for (int lxb = lxb_min; lxb <= lmax[1] - lzb - lyb; lxb++) {
								const int jco = coset(lxb, lyb, lzb);
								for (int lza = 0; lza <= lmax[0]; lza++) {
										for (int lya = 0; lya <= lmax[0] - lza; lya++) {
												const int lxa_min = imax(lmin[0] - lza - lya, 0);
												for (int lxa = lxa_min; lxa <= lmax[0] - lza - lya; lxa++) {
														const int ico = coset(lxa, lya, lza);
														double pab_ = 0.0;

														/* this can be done with 3 dgemms actually but need to
														 * set coef accordingly (triangular along the second
														 * diagonal) */

														for (int lxp = 0; lxp <= lxa + lxb; lxp++) {
																for (int lzp = 0; lzp <= lp - lxa - lxb; lzp++) {
																		for (int lyp = 0; lyp <= lp - lxa - lxb - lzp; lyp++) {
																				const double p2 = alpha_(1, lyb, lya, lyp) *
																						alpha_(2, lzb, lza, lzp) *
																						alpha_(0, lxb, lxa, lxp) *
																						prefactor;
																				pab_ += coef_(lyp, lxp, lzp) * p2;
																		}
																}
														}
														vab(jco, ico) += pab_;
												}
										}
								}
						}
				}
		}
}

// *****************************************************************************
void cpu_handler::prepare_alpha(const task_info &task, const int *lmax) {

		alpha_.zero();
	//
	//   compute polynomial expansion coefs -> (x-a)**lxa (x-b)**lxb -> sum_{ls}
	//   alpha(ls,lxa,lxb,1)*(x-p)**ls
	//

	for (int iaxis = 0; iaxis < 3; iaxis++) {
			const double drpa = task.rp[iaxis] - task.ra[iaxis];
			const double drpb = task.rp[iaxis] - task.rb[iaxis];
			for (int lxa = 0; lxa <= lmax[0]; lxa++) {
					for (int lxb = 0; lxb <= lmax[1]; lxb++) {
							double binomial_k_lxa = 1.0;
							double a = 1.0;
							for (int k = 0; k <= lxa; k++) {
									double binomial_l_lxb = 1.0;
									double b = 1.0;
									for (int l = 0; l <= lxb; l++) {
											alpha_(iaxis, lxb, lxa, lxa - l + lxb - k) +=
													binomial_k_lxa * binomial_l_lxb * a * b;
											binomial_l_lxb *= ((double)(lxb - l)) / ((double)(l + 1));
											b *= drpb;
									}
									binomial_k_lxa *= ((double)(lxa - k)) / ((double)(k + 1));
									a *= drpa;
							}
					}
			}
	}
}


/* this function computes the coefficients initially expressed in the cartesian
 * space to the grid space. It is inplane and can also be done with
 * matrix-matrix multiplication. It is in fact a tensor reduction. */

void cpu_handler::transform_coef_xzy_to_ikj(const double dh[3][3],
																												tensor1<double, 3> &coef_xyz) {
		const int lp = coef_xyz.size(0) - 1;
		tensor1<double, 3> coef_ijk(coef_xyz.size(0), coef_xyz.size(1),
																coef_xyz.size(2));

		/* this tensor corresponds to the term
		 * $v_{11}^{k_{11}}v_{12}^{k_{12}}v_{13}^{k_{13}}
		 * v_{21}^{k_{21}}v_{22}^{k_{22}}v_{23}^{k_{23}}
		 * v_{31}^{k_{31}}v_{32}^{k_{32}} v_{33}^{k_{33}}$ in Eq.26 found section
		 * III.A of the notes */
		tensor1<double, 3> hmatgridp(coef_xyz.size(0), 3, 3);

		coef_ijk.zero();

		// transform using multinomials
		for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
						hmatgridp(0, j, i) = 1.0;
						for (int k = 1; k <= lp; k++) {
								hmatgridp(k, j, i) = hmatgridp(k - 1, j, i) * dh[j][i];
						}
				}
		}

		const int lpx = lp;
		for (int klx = 0; klx <= lpx; klx++) {
				for (int jlx = 0; jlx <= lpx - klx; jlx++) {
						for (int ilx = 0; ilx <= lpx - klx - jlx; ilx++) {
								const int lx = ilx + jlx + klx;
								const int lpy = lp - lx;
								const double tx = hmatgridp(ilx, 0, 0) *
										hmatgridp(jlx, 1, 0) *
										hmatgridp(klx, 2, 0) * fac(lx) * inv_fac[klx] *
										inv_fac[jlx] * inv_fac[ilx];

								for (int kly = 0; kly <= lpy; kly++) {
										for (int jly = 0; jly <= lpy - kly; jly++) {
												for (int ily = 0; ily <= lpy - kly - jly; ily++) {
														const int ly = ily + jly + kly;
														const int lpz = lp - lx - ly;
														const double ty = tx * hmatgridp(ily, 0, 1) *
																hmatgridp(jly, 1, 1) *
																hmatgridp(kly, 2, 1) * fac(ly) *
																inv_fac[kly] * inv_fac[jly] * inv_fac[ily];
														for (int klz = 0; klz <= lpz; klz++) {
																for (int jlz = 0; jlz <= lpz - klz; jlz++) {
																		for (int ilz = 0; ilz <= lpz - klz - jlz; ilz++) {
																				const int lz = ilz + jlz + klz;
																				const int il = ilx + ily + ilz;
																				const int jl = jlx + jly + jlz;
																				const int kl = klx + kly + klz;
																				// const int lijk= coef_map[kl][jl][il];
																				/* the fac table is the factorial. It
																				 * would be better to use the
																				 * multinomials. */
																				coef_ijk(il, kl, jl) +=
																						coef_xyz(lx, lz, ly) * ty *
																						hmatgridp(ilz, 0, 2) *
																						hmatgridp(jlz, 1, 2) *
																						hmatgridp(klz, 2, 2) * fac(lz) * inv_fac[klz] *
																						inv_fac[jlz] * inv_fac[ilz];
																		}
																}
														}
												}
										}
								}
						}
				}
		}

		memcpy(coef_xyz.at(), coef_ijk.at(), sizeof(double) * coef_ijk.size());
		coef_ijk.clear();
		hmatgridp.clear();
}

/* Rotate the coefficients computed in the local grid coordinates to the
 * cartesians coorinates. The order of the indices indicates how the
 * coefficients are stored */
void cpu_handler::transform_coef_jik_to_yxz(const double dh_[3][3], tensor1<double, 3> &coef_xyz) {
	const int lp = coef_xyz.size(0) - 1;
	coef_ijk.resize(coef_xyz.size(0), coef_xyz.size(1),	coef_xyz.size(2));

	/* this tensor corresponds to the term
	 * $v_{11}^{k_{11}}v_{12}^{k_{12}}v_{13}^{k_{13}}
	 * v_{21}^{k_{21}}v_{22}^{k_{22}}v_{23}^{k_{23}}
	 * v_{31}^{k_{31}}v_{32}^{k_{32}} v_{33}^{k_{33}}$ in Eq.26 found section
	 * III.A of the notes */
	hmatgridp.resize(coef_xyz.size(0), 3, 3);

	coef_ijk.zero();

	// transform using multinomials
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
				hmatgridp(0, j, i) = 1.0;
			for (int k = 1; k <= lp; k++) {
					hmatgridp(k, j, i) = hmatgridp(k - 1, j, i) * dh_[j][i];
			}
		}
	}

	const int lpx = lp;
	for (int klx = 0; klx <= lpx; klx++) {
		for (int jlx = 0; jlx <= lpx - klx; jlx++) {
			for (int ilx = 0; ilx <= lpx - klx - jlx; ilx++) {
				const int lx = ilx + jlx + klx;
				const int lpy = lp - lx;
				for (int kly = 0; kly <= lpy; kly++) {
					for (int jly = 0; jly <= lpy - kly; jly++) {
						for (int ily = 0; ily <= lpy - kly - jly; ily++) {
							const int ly = ily + jly + kly;
							const int lpz = lp - lx - ly;
							for (int klz = 0; klz <= lpz; klz++) {
								for (int jlz = 0; jlz <= lpz - klz; jlz++) {
									for (int ilz = 0; ilz <= lpz - klz - jlz; ilz++) {
										const int lz = ilz + jlz + klz;
										const int il = ilx + ily + ilz;
										const int jl = jlx + jly + jlz;
										const int kl = klx + kly + klz;
										// const int lijk= coef_map[kl][jl][il];
										/* the fac table is the factorial. It
										 * would be better to use the
										 * multinomials. */
										coef_ijk(ly, lx, lz) +=
												coef_xyz(jl, il, kl) *
												hmatgridp(ilx, 0, 0) *
												hmatgridp(jlx, 1, 0) *
												hmatgridp(klx, 2, 0) *
												hmatgridp(ily, 0, 1) *
												hmatgridp(jly, 1, 1) *
												hmatgridp(kly, 2, 1) *
												hmatgridp(ilz, 0, 2) *
												hmatgridp(jlz, 1, 2) *
												hmatgridp(klz, 2, 2) * fac(lx) * fac(ly) *
												fac(lz) /
												(fac(ilx) * fac(ily) * fac(ilz) * fac(jlx) * fac(jly) *
												 fac(jlz) * fac(klx) * fac(kly) * fac(klz));
									}
								}
							}
						}
					}
				}
			}
		}
	}
	memcpy(coef_xyz.at(), coef_ijk.at(), sizeof(double) * coef_ijk.size());
}
