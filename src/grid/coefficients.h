#ifndef COEFFICIENTS_H
#define COEFFICIENTS_H
#include <string.h>
#include "utils.h"
#include "grid_common.h"

// *****************************************************************************
extern void grid_prepare_coef_ortho(const int *lmax,
                                    const int *lmin,
                                    const int lp,
                                    const double prefactor,
                                    const tensor *alpha, // [3][lb_max+1][la_max+1][lp+1]
                                    const double pab[ncoset[lmax[1]]][ncoset[lmax[0]]],
                                    tensor *coef_xyz); //[lp+1][lp+1][lp+1]

extern void grid_prepare_alpha(const double ra[3],
                               const double rb[3],
                               const double rp[3],
                               const int *lmax,
                               tensor *alpha);

extern void grid_prepare_coef(const int *lmax,
                              const int *lmin,
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
#endif
