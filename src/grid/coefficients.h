#ifndef COEFFICIENTS_H
#define COEFFICIENTS_H
#include <stdbool.h>
#include <string.h>
#include "utils.h"
#include "grid_common.h"

// *****************************************************************************
extern void grid_prepare_alpha(const double ra[3],
                               const double rb[3],
                               const double rp[3],
                               const int *lmax,
                               tensor *alpha);

extern void grid_prepare_coef(const int *const lmin,
                              const int *const lmax,
                              const int lp,
                              const double prefactor,
                              const tensor *alpha,
                              const double pab[ncoset[lmax[1]]][ncoset[lmax[0]]],
                              tensor *coef_xyz);

extern void compute_compact_polynomial_coefficients(const tensor *coef,
                                                    const int *coef_offset_,
                                                    const int *lmin,
                                                    const int *lmax,
                                                    const double *ra,
                                                    const double *rb,
                                                    const double *rab,
                                                    const double prefactor,
                                                    tensor *co);

extern void grid_transform_coef_xyz_to_ijk(const double dh[3][3],
                                           const double dh_inv[3][3],
                                           const tensor *coef_xyz);

extern void transform_triangular_to_xyz(const double *const coef_xyz, tensor *const coef);
extern void transform_xyz_to_triangular(const tensor *const coef, double *const coef_xyz);
extern void transform_yxz_to_triangular(const tensor *const coef, double *const coef_xyz);
#endif
