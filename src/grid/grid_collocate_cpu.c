/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2020  CP2K developers group                         *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#if defined(__MKL) || defined(HAVE_MKL)
#include <mkl.h>
#include <mkl_cblas.h>
#endif

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

#include "grid_collocate_replay.h"
#include "grid_collocate_cpu.h"
#include "grid_prepare_pab.h"
#include "grid_common.h"
#include "tensor_local.h"
#include "utils.h"
#include "coefficients.h"

#define TEST 1

void collocate_l0(const struct tensor_ *co,
                  const struct tensor_ *p_alpha_beta_reduced_,
                  struct tensor_ *cube);

void collocate_core_rectangular(char *scratch,
                                const struct tensor_ *co,
                                const struct tensor_ *p_alpha_beta_reduced_,
                                struct tensor_ *Vtmp);


/* this function will replace the function grid_prepare_coef when I get the
 * tensor coef right. This need change elsewhere in cp2k, something I do not
 * want to do right now. I need to rethink the order of the parameters */

static void compute_compact_polynomial_coefficients(const tensor *coef,
                                                    const int *coef_offset_,
                                                    const int *lmin,
                                                    const int *lmax,
                                                    const double *ra,
                                                    const double *rb,
                                                    const double *rab,
                                                    const double prefactor,
                                                    tensor *co)
{
    // binomial coefficients n = 0 ... 20
    const int binomial[21][21] = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 6, 15, 20, 15, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 7, 21, 35, 35, 21, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 8, 28, 56, 70, 56, 28, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 9, 36, 84, 126, 126, 84, 36, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1,
         0, 0,  0,  0,   0,   0,   0,   0,   0,  0},
        {1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11,
         1, 0,  0,  0,   0,   0,   0,   0,   0,   0},
        {1,  12, 66, 220, 495, 792, 924, 792, 495, 220, 66,
         12, 1,  0,  0,   0,   0,   0,   0,   0,   0},
        {1,  13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286,
         78, 13, 1,  0,   0,   0,    0,    0,    0,    0},
        {1,   14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001,
         364, 91, 14, 1,   0,    0,    0,    0,    0,    0},
        {1,    15,  105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003,
         1365, 455, 105, 15,  1,    0,    0,    0,    0,    0},
        {1,    16,   120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008,
         4368, 1820, 560, 120, 16,   1,    0,    0,     0,     0},
        {1,     17,   136,  680, 2380, 6188, 12376, 19448, 24310, 24310, 19448,
         12376, 6188, 2380, 680, 136,  17,   1,     0,     0,     0},
        {1,     18,    153,  816,  3060, 8568, 18564, 31824, 43758, 48620, 43758,
         31824, 18564, 8568, 3060, 816,  153,  18,    1,     0,     0},
        {1,     19,    171,   969,   3876,  11628, 27132,
         50388, 75582, 92378, 92378, 75582, 50388, 27132,
         11628, 3876,  969,   171,   19,    1,     0},
        {1,     20,     190,    1140,   4845,   15504,  38760,
         77520, 125970, 167960, 184756, 167960, 125970, 77520,
         38760, 15504,  4845,   1140,   190,    20,     1}};

    if (lmax[0] + lmax[1] == 0) {
        idx3(co[0], 0, 0, 0) = prefactor * idx2(coef[0], 0, 0);
        return;
    }

    /*
     * Notation alpha = lx, beta = ly, gamma = lz (l angular momentum cp2k
     * notation)
     *
     * this routine computes the coefficients coef_xyz in the ordering for the
     * collocate coming afterwards. The final order of coef_xyz should be
     * coef[x][z][y] so we avoid quite few dgemm calls doing so.
     *
     * The initial coefficients are stored as co[x1y1z1][x2y2z2] so we need to
     * reshufle the data such that co[x1x2][y1y2][z1z2]. With
     * co[gamma][beta][alpha] and
     *
     *     pol[0] = pol_alpha (i indice)
     *     pol[1] = pol_beta, (j indice)
     *     pol[2] = pol_gamma (k indice)
     *
     * collocate_rectangular returns coef[beta][gamma][alpha] but we want
     * coef[alpha][gamma][beta]. this means that we need co[beta][alpha][gamma],
     * with
     *
     *     pol[2] = pol_beta
     *     pol[1] = pol_alpha
     *     pol[0] = pol_gamma,
     *
     */

    tensor power, px, coef_tmp;

    initialize_tensor_4(&power, 2, 3, lmax[0] + lmax[1] + 1, lmax[0] + lmax[1] + 1);
    initialize_tensor_3(&px, 3, (lmax[0] + 1) * (lmax[1] + 1), lmax[0] + lmax[1] + 1);
    initialize_tensor_3(&coef_tmp, (lmax[0] + 1) * (lmax[1] + 1), // alpha
                        (lmax[0] + 1) * (lmax[1] + 1),  // gamma
                        (lmax[0] + 1) * (lmax[1] + 1)); // beta

#if defined(__LIBXSMM)
    power.data = libxsmm_aligned_scratch(sizeof(double) * power.alloc_size_, 0/*auto-alignment*/);
    px.data = libxsmm_aligned_scratch(sizeof(double) * px.alloc_size_, 0/*auto-alignment*/);
    coef_tmp.data = libxsmm_aligned_scratch(sizeof(double) * coef_tmp.alloc_size_, 0/*auto-alignment*/);
#else
#error "Need implementation"
#endif
    /* I compute (x - xa) ^ k Binomial(alpha, k), for alpha = 0.. l1 + l2 + 1
     * and k = 0 .. l1 + l2 + 1. It is used everywhere here and make the economy
     * of multiplications and function calls
     */
    for (int dir = 0; dir < 3; dir++) {
        double tmp = rab[dir] - ra[dir];
        idx4(power, 0, dir, 0, 0) = 1.0;
        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++) {
            idx4(power, 0, dir, 0, k) =
                tmp * idx4(power, 0, dir, 0, k - 1);
        }

        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            memcpy(&idx4(power, 0, dir, k, 0),
                   &idx4(power, 0, dir, k - 1, 0),
                   sizeof(double) * power.size[3]);

        for (int a1 = 0; a1 < lmax[0] + lmax[1] + 1; a1++)
            for (int k = 0; k < lmax[0] + lmax[1] + 1; k++)
                idx4(power, 0, dir, a1, k) *= binomial[a1][k];

        tmp = rab[dir] - rb[dir];

        idx4(power, 1, dir, 0, 0) = 1.0;
        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            idx4(power, 1, dir, 0, k) =
                tmp * idx4(power, 1, dir, 0, k - 1);

        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            memcpy(&idx4(power, 1, dir, k, 0),
                   &idx4(power, 1, dir, k - 1, 0),
                   sizeof(double) * power.size[3]);

        for (int a1 = 0; a1 < lmax[0] + lmax[1] + 1; a1++)
            for (int k = 0; k < lmax[0] + lmax[1] + 1; k++)
                idx4(power, 1, dir, a1, k) *= binomial[a1][k];
    }

    memset(px.data, 0, sizeof(double) * px.alloc_size_);
    int perm[3] = {1, 2, 0};
    for (int i = 0; i < 3; i++) {
        for (int a1 = 0; a1 <= lmax[0]; a1++) {
            for (int k1 = 0; k1 <= a1; k1++) {
                const double c1 = idx4(power, 0, i, a1, a1 - k1);
                for (int a2 = 0; a2 <= lmax[1]; a2++) {
                    const double *__restrict__ src = &idx4(power, 1, i, a2, 0);
                    double *__restrict__ dst =
                        &idx3(px, perm[i], a1 * (lmax[1] + 1) + a2, k1);
                    for (int k2 = 0; k2 <= a2; k2++) {
                        dst[k2] += c1 * src[a2 - k2];
                    }
                }
            }
        }
    }

    /* Now we need to reorder the coefficients. */
    memset(coef_tmp.data, 0, sizeof(double) * coef_tmp.alloc_size_);

    for (int a1 = 0; a1 <= lmax[0]; a1++) {
        for (int a2 = 0; a2 <= lmax[1]; a2++) {
            const int i_a = a1 * (lmax[1] + 1) + a2;
            for (int b1 = 0; b1 <= lmax[0]; b1++) {
                for (int b2 = 0; b2 <= lmax[1]; b2++) {
                    const int i_b = b1 * (lmax[1] + 1) + b2;
                    for (int g1 = max(0, lmin[0] - a1 - b1); g1 <= lmax[0]; g1++) {
                        const int l1 = g1 + a1 + b1;
                        if ((l1 >= lmin[0]) && (l1 <=lmax[0])) {
                            const int i1 = coef_offset_[0]  +
                                return_linear_index_from_exponents(a1, b1, g1);
                            const double *__restrict__ src = &idx2(coef[0], i1, 0);
                            for (int g2 = 0; g2 <= lmax[1]; g2++) {
                                const int l2 = g2 + a2 + b2;
                                if ((l2 >= lmin[1]) && (l2 <=lmax[1])) {
                                    const int i_g = g1 * (lmax[1] + 1) + g2;
                                    const int i2 = coef_offset_[1] +
                                        return_linear_index_from_exponents(a2, b2, g2);
                                    idx3(coef_tmp, i_b, i_a, i_g) += src[i2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

// it is a collocate now....

    collocate_core_rectangular(NULL, &coef_tmp, &px, co);

    // rescale the all thing with the prefactor
    cblas_dscal(co->alloc_size_, prefactor, co->data, 1);
}

void compute_non_zero_elements(const int center,
                               const int pol_length,
                               const int period,
                               const int grid_size,
                               const int grid_lower_boundaries,
                               int *__restrict non_zero_elements, // we can compress it using bit encoding
                               int *number_of_non_zero_elements)
{
    const int start = (center - grid_lower_boundaries - (pol_length - 1) / 2 + 32 * period) % period;

    memset(non_zero_elements, 0, sizeof(int) * grid_size);

    for (int s = 0; s < min(grid_size - start, pol_length); s++) {
        non_zero_elements[s + start] = 1;
    }

    for (int s = min(grid_size - start, pol_length); s < pol_length; s++) {
        for (int s1 = 0; s1 < min(grid_size, pol_length - s); s1++)
            non_zero_elements[s1] = 1;
        s += period;
    }


    *number_of_non_zero_elements = 0;
    for (int s = 0; s < grid_size; s++)
        *number_of_non_zero_elements += non_zero_elements[s];

}

void compute_folded_polynomial(const int cube_center,
                               const int pol_length,
                               const int period,
                               const int grid_size,
                               const int grid_lower_boundaries,
                               const int *__restrict non_zero_elements,
                               const double *__restrict pol,
                               double *__restrict res,
                               double *__restrict tmp)
{

    const int start = (cube_center - grid_lower_boundaries - (pol_length - 1) / 2 + 32 * period) % period;

    for (int s = start; s < min(grid_size, pol_length + start); s++) {
        tmp[s] = pol[s - start];
    }

    for (int s = min(grid_size - start, pol_length); s < pol_length; s++) {
        for (int s1 = 0; s1 < min(grid_size, pol_length - s); s1++)
            tmp[s1] += pol[s + s1];
        s += period;
    }

    // now compress the all thing. I do not care about zeros.
    int offset = 0;

    for (int s = 0; s < grid_size; s++) {
        /* if (non_zero_elements[s] == 0) */
        /*     continue; */
        for (; (s < grid_size - 1) && (non_zero_elements[s] == 0); s++);
        int smin = s;

        for (; (s < grid_size - 1) && (non_zero_elements[s] == 1); s++);

        int smax = s + non_zero_elements[s];

        for (int si = 0; si < (smax - smin); si++)
            res[offset + si] = tmp[smin + si];
        offset += smax - smin;
    }
}


void apply_pcb_orthogonal_case(const int *cube_center,
                               const int *pol_length,
                                      const int *period,
                                      const int *grid_size,
                                      const int *grid_lower_boundaries,
                                      const tensor *pol_unfolded,
                                      int *non_zero_elements[3],
                                      int *number_of_non_zero_elements,
                                      tensor *pol_folded)
{

    for (int d = 0; d < 3; d++) {
        compute_non_zero_elements(cube_center[d],
                                  pol_length[d],
                                  period[d],
                                  grid_size[d],
                                  grid_lower_boundaries[d],
                                  non_zero_elements[d], // we can compress it using bit encoding
                                  number_of_non_zero_elements + d);
    }

    /* printf("non zero elements: %d %d %d\n", */
    /*        number_of_non_zero_elements[2], */
    /*        number_of_non_zero_elements[1], */
    /*        number_of_non_zero_elements[0]); */

    const int smax = max(max(number_of_non_zero_elements[0],
                             number_of_non_zero_elements[1]),
                         number_of_non_zero_elements[2]);

    initialize_tensor_3(pol_folded, 3, pol_unfolded->size[1], smax);

#if defined(__LIBXSMM)
    pol_folded->data = libxsmm_aligned_scratch(sizeof(double) * pol_folded->alloc_size_, 0/*auto-alignment*/);
    const int max_grid_size = max(max(grid_size[0], grid_size[1]), grid_size[2]);
    double *scratch = libxsmm_aligned_scratch(sizeof(double) * max_grid_size, 0/*auto-alignment*/);
#else
#error "Need implementation"
#endif

    memset(pol_folded->data, 0, sizeof(double) * pol_folded->alloc_size_);

    for (int d = 0; d < 3; d++) {
        for (int l = 0; l < pol_unfolded->size[1]; l++) {
            memset(scratch, 0, sizeof(double) * max_grid_size);
            compute_folded_polynomial(cube_center[d],
                                      pol_length[d],
                                      period[d],
                                      grid_size[d],
                                      grid_lower_boundaries[d],
                                      non_zero_elements[d],
                                      &idx3(pol_unfolded[0], d, l, 0),
                                      &idx3(pol_folded[0], d, l, 0),
                                      scratch);
        }
    }

#if defined(__LIBXSMM)
    libxsmm_free(scratch);
#else
#error "Need implementation"
#endif
}

void collocate_l0(const struct tensor_ *co,
                  const struct tensor_ *p_alpha_beta_reduced_,
                  struct tensor_ *cube)
{
    const double *__restrict pz = &idx3(p_alpha_beta_reduced_[0], 2, 0, 0); /* k indice */
    const double *__restrict py = &idx3(p_alpha_beta_reduced_[0], 1, 0, 0); /* j indice */
    const double *__restrict px = &idx3(p_alpha_beta_reduced_[0], 0, 0, 0); /* i indice */
    double *__restrict dst = &idx3(cube[0], 0, 0, 0);
    const double coo = idx3 (co[0], 0, 0, 0);
    const double tz1 = pz[0];

    for (int y1 = 0; y1 < cube->size[1]; y1++) {
        const double tmp = coo * py[y1];
#pragma GCC unroll 4
#pragma GCC ivdep
        for (int x1 = 0; x1 < cube->size[2]; x1++) {
            dst[x1] = tmp * px[x1];
        }
        dst += cube->ld_;
    }

    const double *__restrict src1 = &idx3(cube[0], 0, 0, 0);

    for (int z1 = 1; z1 < cube->size[0]; z1++) {
        const double tz = pz[z1];

#pragma GCC unroll 4
#pragma GCC ivdep
        for (int y1 = 0; y1 < cube->size[1] * cube->ld_; y1++) {
            dst[y1] = src1[y1] * tz;
        }
        dst += cube->ld_ * cube->size[1];
    }

    dst =  src1;
#pragma GCC unroll 4
#pragma GCC ivdep
    for (int y1 = 0; y1 < cube->size[1] * cube->ld_; y1++) {
        dst[y1] *= tz1;
    }
}

// *****************************************************************************
static void grid_fill_map(const bool periodic,
                          const int lb_cube,
                          const int ub_cube,
                          const int cubecenter,
                          const int lb_grid,
                          const int npts,
                          const int ngrid,
                          const int cmax,
                          int *map)
{
    if (periodic) {
        /* for (int i=0; i <= 2*cmax; i++) */
        /*     map[i] = mod(cubecenter + i - cmax, npts) + 1; */
        int start = lb_cube;
        while (true) {
            const int offset = mod(cubecenter + start, npts)  + 1 - start;
            const int length = min(ub_cube, npts - offset) - start;
            for (int ig = start; ig <= start + length; ig++) {
                map[ig + cmax] = ig + offset;
            }
            if (start + length >= ub_cube){
                break;
            }
            start += length + 1;
        }
    } else {
        // this takes partial grid + border regions into account
        const int offset = mod(cubecenter + lb_cube + lb_grid, npts) + 1 - lb_cube;
        // check for out of bounds
        assert(ub_cube + offset <= ngrid);
        assert(lb_cube + offset >= 1);
        for (int ig = lb_cube; ig <= ub_cube; ig++) {
            map[ig + cmax] = ig + offset;
        }
    }
}


/* compute the functions (x - x_i)^l exp (-eta (x - x_i)^2) for l = 0..lp using
 * a recursive relation to avoid computing the exponential on each grid point. I
 * think it is not really necessary anymore since it is *not* the dominating
 * contribution to computation of collocate and integrate */


static void grid_fill_pol(const double dr,
                          const double roffset,
                          const int lb_cube,
                          const int lp,
                          const int cmax,
                          const double zetp,
                          double *pol_)
{
    tensor pol;
    initialize_tensor_2(&pol, lp + 1, 2 * cmax + 1);
    pol.data = pol_;
//
//   compute the values of all (x-xp)**lp*exp(..)
//
//  still requires the old trick:
//  new trick to avoid to many exps (reuse the result from the previous gridpoint):
//  exp( -a*(x+d)**2)=exp(-a*x**2)*exp(-2*a*x*d)*exp(-a*d**2)
//  exp(-2*a*(x+d)*d)=exp(-2*a*x*d)*exp(-2*a*d**2)
//
    const double t_exp_1 = exp(-zetp * dr * dr);
    const double t_exp_2 = t_exp_1 * t_exp_1;

    double t_exp_min_1 = exp(-zetp * (dr - roffset) * (dr - roffset));
    double t_exp_min_2 = exp(2.0 * zetp * (dr - roffset) * dr);

    for (int ig = 0; ig >= lb_cube; ig--) {
        const double rpg = ig * dr - roffset;
        t_exp_min_1 *= t_exp_min_2 * t_exp_1;
        t_exp_min_2 *= t_exp_2;
        double pg = t_exp_min_1;
        // pg  = EXP(-zetp*rpg**2)
        for (int icoef=0; icoef <= lp; icoef++) {
            idx2(pol, icoef, ig - lb_cube) = pg;
            pg *= rpg;
        }
    }

    double t_exp_plus_1 = exp(-zetp * roffset * roffset);
    double t_exp_plus_2 = exp(2.0 * zetp * roffset * dr);
    for (int ig=0; ig >= lb_cube; ig--) {
        const double rpg = (1-ig) * dr - roffset;
        t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
        t_exp_plus_2 *= t_exp_2;
        double pg = t_exp_plus_1;
        // pg  = EXP(-zetp*rpg**2)
        for (int icoef = 0; icoef <= lp; icoef++) {
            idx2(pol, icoef, 1 - ig - lb_cube) = pg;
            pg *= rpg;
        }
    }
}

void collocate_core_rectangular1(char *scratch,
                                 const int zmin,
                                 const int ymin,
                                 const int xmin,
                                 const int zmax,
                                 const int ymax,
                                 const int xmax,
                                 const int zoffset,
                                 const int yoffset,
                                 const int xoffset,
                                 const struct tensor_ *co,
                                 const struct tensor_ *p_alpha_beta_reduced_,
                                 struct tensor_ *grid)
{
    /* printf("cube size %d %d %d\n" , zmax - zmin, ymax - ymin, xmax - xmin); */
    /* printf("coord : %d %d %d\n" , zmin, ymin, xmin); */
    /* printf("grid size : %d %d %d\n" , grid->size[0], grid->size[1], grid->size[2]); */
    if (co->size[1] > 1) {
        // an helper structure for the dgemm parameters (*ROW MAJOR* format for
        // the matrices)
        struct {
            double alpha;
            double beta;
            double *a, *b, *c;
            int m, n, k, lda, ldb, ldc;
        } m1, m2, m3;

        tensor T;
        tensor W;

        initialize_tensor_3(&T, co->size[0] /* alpha */, co->size[1] /* gamma */, ymax - ymin /* j */);

        initialize_tensor_3(&W, co->size[1] /* alpha */ , zmax - zmin /* k */, ymax - ymin /* j */);

#if defined(__LIBXSMM)
            T.data = libxsmm_aligned_scratch(sizeof(double) * T.alloc_size_, 0/*auto-alignment*/);
        W.data = libxsmm_aligned_scratch(sizeof(double) * W.alloc_size_, 0/*auto-alignment*/);
#else
        T.data = (double *)scratch;
        W.data = ((double *)scratch) + T.alloc_size_ * sizeof(double);
#endif

/* WARNING we are in row major layout. cblas allows it and it is more
 * natural to read left to right than top to bottom
 *
 * we do first T_{\alpha,\gamma,j} = \sum_beta C_{alpha\gamma\beta} Y_{\beta, j}
 *
 * keep in mind that Y_{\beta, j} = p_alpha_beta_reduced(1, \beta, j)
 * and the order of indices is also important. the last indice is the
 * fastest one. it can be done with one dgemm.
 */

        m1.alpha = 1.0;
        m1.beta = 0.0;
        m1.m = co->size[0] * co->size[1]; /* alpha gamma */
        m1.n = T.size[2]; /* j */
        m1.k = co->size[2]; /* beta */
        m1.a = co->data; // Coef_{alpha,gamma,beta} Coef_xzy
        m1.lda = co->ld_;
        m1.b = &idx3(p_alpha_beta_reduced_[0], 1, 0, yoffset); // Y_{beta, j} = p_alpha_beta_reduced(1, beta, j)
        m1.ldb = p_alpha_beta_reduced_->ld_;
        m1.c = T.data; // T_{\alpha, \gamma, j} = T(alpha, gamma, j)
        m1.ldc = T.ld_;

/*
 * the next step is a reduction along the gamma index. Unfortunately, it
 * can not be done with one dgemm call, because we want that the order
 * of the indices to be W_{alpha, k, j}....
 *
 * We compute then
 *
 * W_{alpha, k, j} = sum_{\gamma} Z_{k, \gamma} T_{\alpha, \gamma, j}
 *
 * which means we need to transpose Z_{\gamma, k} = p_alpha_beta_reduced(2, 0, 0)
 */

        m2.alpha = 1.0;
        m2.beta = 0.0;
        m2.m = zmax - zmin; // k direction
        m2.n = ymax - ymin; // j direction
        m2.k = co->size[2]; // gamma
        m2.a = &idx3(p_alpha_beta_reduced_[0], 2, 0, zoffset); // p_alpha_beta_reduced(0, gamma, j)
        m2.lda = p_alpha_beta_reduced_->ld_;
        m2.b = T.data; // T_{\alpha, \gamma, j}
        m2.ldb = T.ld_;
        m2.c = W.data; // W_{\alpha, k, j}
        m2.ldc = W.ld_;

/* the final step is again a reduction along the alpha indice. It can
 * again be done with one dgemm. The operation is simply
 *
 * Cube_{k, j, i} = \sum_{alpha} W_{k, j, alpha} X_{alpha, i}
 *
 * which means we need to permute W_{\alpha, k, j} in to W_{k, j,
 * \alpha} which can be done with one transposition if we consider (k,j)
 * as a composite index.
 */

        m3.alpha = 1.0;
        m3.beta = 0.0;
        m3.m = ymax - ymin; // (k, j)
        m3.n = xmax - xmin; // i direction
        m3.k = co->size[2]; // alpha
        m3.a = &idx3(W, 0, 0, 0); // W_{\alpha, k, j}
        m3.lda = W.size[1] * W.ld_;
        m3.b = &idx3(p_alpha_beta_reduced_[0], 0, 0, xoffset); // p_alpha_beta_reduced(2, alpha, i)
        m3.ldb = p_alpha_beta_reduced_->ld_;
        m3.c = &idx3(grid[0], 0, 0, 0); // cube_{kji}
        m3.ldc = grid->ld_;

// openblas and mkl have a C interface
#if defined(__LIBXSMM)
        libxsmm_dgemm("N",
                      "N",
                      &m1.n,
                      &m1.m,
                      &m1.k,
                      &m1.alpha,
                      m1.b,
                      &m1.ldb,
                      m1.a,
                      &m1.lda,
                      &m1.beta,
                      m1.c, // tmp_{alpha, gamma, j}
                      &m1.ldc);
#elif defined(__MKL)
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m1.m,
                    m1.n,
                    m1.k,
                    1.0,
                    m1.a,
                    m1.lda,
                    m1.b,
                    m1.ldb,
                    0.0,
                    m1.c, // tmp_{alpha, gamma, j}
                    m1.ldc);
#else
        dgemm_("N",
               "N",
               &m1.n,
               &m1.m,
               &m1.k,
               &m1.alpha,
               m1.b,
               &m1.ldb,
               m1.a,
               &m1.lda,
               &m1.beta,
               m1.c, // tmp_{alpha, gamma, j}
               &m1.ldc);
#endif


/* #if defined(__LIBXSMM) */
/*         /\* batched interface *\/ */
/*         int prefetch = LIBXSMM_PREFETCH_AUTO; */
/*         int flags = LIBXSMM_GEMM_FLAG_TRANS_B; /\* LIBXSMM_FLAGS *\/; */
/*         libxsmm_dmmfunction xmm1 = NULL; */

/*         xmm1 = libxsmm_dmmdispatch(m2.n, m2.m, m2.k, */
/*                                   &m2.ldb, &m2.lda, &m2.ldc, */
/*                                   &m2.alpha, &m2.beta, */
/*                                   &flags, &prefetch); */

/*         for (int a1 = 0; a1 < co->size[0] - 1; a1++) { */
/*             xmm1(m2.b + a1 * T.offsets[0], m2.a, m2.c + a1 * W.offsets[0], */
/*                  m2.b + (a1 + 1) * T.offsets[0], m2.a, m2.c + (a1 + 1) * W.offsets[0]); */
/*         } */

/*         xmm1(m2.b + (co->size[0] - 1) * T.offsets[0], m2.a, m2.c + (co->size[0] - 1) * W.offsets[0], */
/*              m2.b + (co->size[0] - 1) * T.offsets[0], m2.a, m2.c + (co->size[0] - 1) * W.offsets[0]); */

/* #elif
 */
#if defined(__MKL)
        for (int a1 = 0; a1 < co->size[0]; a1++) {
            cblas_dgemm(CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        m2.m,
                        m2.n,
                        m2.k,
                        1.0,
                        m2.a,
                        m2.lda,
                        m2.b,
                        m2.ldb,
                        0.0,
                        m2.c,
                        m2.ldc);

            m2.b +=  T.offsets[0];
            m2.c +=  W.offsets[0];
        }
#else
        for (int a1 = 0; a1 < co->size[0]; a1++) {
            dgemm_("N",
                   "T",
                   &m2.n,
                   &m2.m,
                   &m2.k,
                   &m2.alpha,
                   m2.b,
                   &m2.ldb,
                   m2.a,
                   &m2.lda,
                   &m2.beta,
                   m2.c,
                   &m2.ldc);
            m2.b +=  T.offsets[0];
            m2.c +=  W.offsets[0];
        }
#endif

        /*
          In a normal collocate, I would not have the for loop, but here the
          the third dimension, needs a special care because I add the result
          directly to the grid without storing the result in a temporary grid.
        */

#if defined(__LIBXSMM)
        libxsmm_dmmfunction xmm2 = NULL;
        int prefetch = LIBXSMM_PREFETCH_NONE;
        int flags = LIBXSMM_GEMM_FLAG_TRANS_B; /* LIBXSMM_FLAGS */;
        xmm2 = libxsmm_dmmdispatch(m3.n,
                                   m3.m,
                                   m3.k,
                                   &m3.ldb,
                                   &m3.lda,
                                   &grid[0].ld_,
                                   &m3.alpha,
                                   &m3.beta,
                                   &flags,
                                   &prefetch);

        for (int z = zmin; z < zmax; z++) {
            xmm2(m3.b,
                 &idx3(W, 0, z - zmin, 0),
                 &idx3(grid[0], z, ymin, xmin));
        }


        /* libxsmm_release_kernel(xmm1); */
        /* libxsmm_release_kernel(xmm2); */

#elif defined(__MKL)
        for (int z = zmin; z < zmax; z++) {
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m3.m, m3.n, m3.k,
                        1.0,
                        &idx3(W, 0, z - zmin, 0),
                        m3.lda,
                        m3.b,
                        m3.ldb,
                        1.0,
                        &idx3(grid[0],
                              z,
                              ymin,
                              xmin),
                        grid[0].ld_);
        }
#else
        for (int z = zmin; z < zmax; z++) {
            dgemm_(, "N", "T",
                   &m3.n,
                   &m3.m,
                   &m3.k,
                   &m3.alpha,
                   m3.b,
                   &m3.ldb,
                   &idx3(W, 0, z - zmin, 0),
                   &m3.lda,
                   &m3.beta,
                   &idx3(grid[0],
                         z,
                         ymin,
                         xmin),
                   &grid[0].ld_);
        }
#endif

#if defined(__LIBXSMM)
        libxsmm_free(W.data);
        libxsmm_free(T.data);
#endif

        return;
    }

    /* l = 0 case */
    const double *__restrict pz = &idx3(p_alpha_beta_reduced_[0], 2, 0, zoffset); /* k indice */
    const double *__restrict py = &idx3(p_alpha_beta_reduced_[0], 1, 0, yoffset); /* j indice */
    const double *__restrict px = &idx3(p_alpha_beta_reduced_[0], 0, 0, xoffset); /* i indice */
    const double coo = idx3 (co[0], 0, 0, 0);
    const double tz1 = pz[0];
    tensor tmp;

    initialize_tensor_2(&tmp, ymax - ymin, xmax - xmin);

#if defined(__LIBXSMM)
    tmp.data = libxsmm_aligned_scratch(sizeof(double) * tmp.alloc_size_, 0/*auto-alignment*/);
#else
#endif
    double *__restrict dst = &idx2(tmp, 0, 0);

    for (int y = 0; y < ymax - ymin; y++) {
        const double tp1 = coo * py[y];
#pragma GCC unroll 4
#pragma GCC ivdep
        for (int x1 = 0; x1 < xmax - xmin; x1++) {
            dst[x1] = tp1 * px[x1];
        }
        dst += tmp.ld_;
    }

    for (int z = 0; z < zmax - zmin; z++) {
        const double tz = pz[z];

        dst = &idx3(grid[0], zmin + z, ymin, xmin);
        const double *__restrict src1 = &idx2(tmp, 0, 0);


        for (int y = 0; y < tmp.size[0]; y++) {
#pragma GCC unroll 4
#pragma GCC ivdep
            for (int x = 0; x < tmp.size[1]; x++) {
                dst[x] += src1[x] * tz;
            }
            src1 += tmp.ld_;
            dst += grid->ld_;
        }
    }

#if defined(__LIBXSMM)
    libxsmm_free(tmp.data);
#endif
}
/* compute the following operation

   V_{kji} = \sum_{\alpha\beta\gamma} C_{\alpha\gamma\beta} T_{0,\alpha,i} T_{1,\beta,j} T_{2,\gamma,k}

*/

void collocate_core_rectangular(char *scratch,
                                const struct tensor_ *co,
                                const struct tensor_ *p_alpha_beta_reduced_,
                                struct tensor_ *cube)
{
    if (co->size[0] > 1) {

        // an helper structure for the dgemm parameters (*ROW MAJOR* format for
        // the matrices)
        struct {
            double one;
            double zero;
            double *a, *b, *c;
            int m, n, k, lda, ldb, ldc;
        } m1, m2, m3;

        tensor T;
        tensor W;

        initialize_tensor_3(&T, co->size[0] /* alpha */, co->size[1] /* gamma */, cube->size[1] /* j */);

        initialize_tensor_3(&W, co->size[1] /* alpha */ , cube->size[0] /* k */, cube->size[1] /* j */);

#if defined(__LIBXSMM)
        T.data = libxsmm_aligned_scratch(sizeof(double) * T.alloc_size_, 0/*auto-alignment*/);
        W.data = libxsmm_aligned_scratch(sizeof(double) * W.alloc_size_, 0/*auto-alignment*/);
#else
        T.data = (double *)scratch;
        W.data = ((double *)scratch) + T.alloc_size_ * sizeof(double);
#endif

        /* WARNING we are in row major layout. cblas allows it and it is more
         * natural to read left to right than top to bottom
         *
         * we do first T_{\alpha,\gamma,j} = \sum_beta C_{alpha\gamma\beta} Y_{\beta, j}
         *
         * keep in mind that Y_{\beta, j} = p_alpha_beta_reduced(1, \beta, j)
         * and the order of indices is also important. the last indice is the
         * fastest one. it can be done with one dgemm.
         */

        m1.one = 1.0;
        m1.zero = 0.0;
        m1.m = co->size[0] * co->size[1]; /* alpha gamma */
        m1.n = cube->size[1]; /* j */
        m1.k = co->size[2]; /* beta */
        m1.a = co->data; // Coef_{alpha,gamma,beta} Coef_xzy
        m1.lda = co->ld_;
        m1.b = &idx3(p_alpha_beta_reduced_[0], 1, 0, 0); // Y_{beta, j} = p_alpha_beta_reduced(1, beta, j)
        m1.ldb = p_alpha_beta_reduced_->ld_;
        m1.c = T.data; // T_{\alpha, \gamma, j} = T(alpha, gamma, j)
        m1.ldc = T.ld_;

        /*
         * the next step is a reduction along the gamma index. Unfortunately, it
         * can not be done with one dgemm call, because we want that the order
         * of the indices to be W_{alpha, k, j}....
         *
         * We compute then
         *
         * W_{alpha, k, j} = sum_{\gamma} Z_{k, \gamma} T_{\alpha, \gamma, j}
         *
         * which means we need to transpose Z_{\gamma, k} = p_alpha_beta_reduced(2, 0, 0)
         */

        m2.one = 1.0;
        m2.zero = 0.0;
        m2.m = cube->size[0]; // k direction
        m2.n = cube->size[1]; // j direction
        m2.k = co->size[2]; // gamma
        m2.a = &idx3(p_alpha_beta_reduced_[0], 2, 0, 0); // p_alpha_beta_reduced(0, gamma, j)
        m2.lda = p_alpha_beta_reduced_->ld_;
        m2.b = T.data; // T_{\alpha, \gamma, j}
        m2.ldb = T.ld_;
        m2.c = W.data; // W_{\alpha, k, j}
        m2.ldc = W.ld_;

        /* the final step is again a reduction along the alpha indice. It can
         * again be done with one dgemm. The operation is simply
         *
         * Cube_{k, j, i} = \sum_{alpha} W_{k, j, alpha} X_{alpha, i}
         *
         * which means we need to permute W_{\alpha, k, j} in to W_{k, j,
         * \alpha} which can be done with one transposition if we consider (k,j)
         * as a composite index.
         */

        m3.one = 1.0;
        m3.zero = 0.0;
        m3.m = cube->size[0] * cube->size[1]; // (k, j)
        m3.n = cube->size[2]; // i direction
        m3.k = co->size[2]; // alpha
        m3.a = &idx3(W, 0, 0, 0); // W_{\alpha, k, j}
        m3.lda = W.size[1] * W.ld_;
        m3.b = &idx3(p_alpha_beta_reduced_[0], 0, 0, 0); // p_alpha_beta_reduced(2, alpha, i)
        m3.ldb = p_alpha_beta_reduced_->ld_;
        m3.c = &idx3(cube[0], 0, 0, 0); // cube_{kji}
        m3.ldc = cube->ld_;


// openblas and mkl have a C interface
#if defined(HAVE_MKL) || defined(HAVE_CBLAS)

        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m1.m,
                    m1.n,
                    m1.k,
                    1.0,
                    m1.a,
                    m1.lda,
                    m1.b,
                    m1.ldb,
                    0.0,
                    m1.c, // tmp_{alpha, gamma, j}
                    m1.ldc);

        for (int a1 = 0; a1 < co->size[0]; a1++) {
            cblas_dgemm(CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        m2.m,
                        m2.n,
                        m2.k,
                        1.0,
                        m2.a,
                        m2.lda,
                        m2.b,
                        m2.ldb,
                        0.0,
                        m2.c,
                        m2.ldc);

            m2.b +=  T.offsets[0];
            m2.c +=  W.offsets[0];
        }

        cblas_dgemm(CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m3.m,
                    m3.n,
                    m3.k,
                    1.0,
                    m3.a,
                    m3.lda,
                    m3.b,
                    m3.ldb,
                    0.0,
                    m3.c,
                    m3.ldc);
#else

        /* WARNING : We are in row major order but we use the fortran interface,
         * which means that dgemm see all matrices transposed. The basic
         * algorithm described above still holds but we have to reverse the
         * order of the matrices.
         *
         * there is a trick (thanks Hans) to get it is right with matrix-matrix
         * multiplication in row major format but calling fortran interfaces of
         * dgemm
         *
         * TA, TB, M, N, K, A, LDA, B, LDB (cblas_dgemm, row major)
         * Changes to
         * TB, TA, N, M, K, B, LDB, A, LDA (dgemm, col major)
         *
         */

// we need to replace this with libxsmm
#if defined(__LIBXSMM)
        libxsmm_dgemm("N",
                      "N",
                      &m1.n/*required*/,
                      &m1.m/*required*/,
                      &m1.k/*required*/,
                      &m1.one,
                      m1.b/*required*/,
                      &m1.ldb/*ldb*/,
                      m1.a/*required*/,
                      &m1.lda/*lda*/,
                      &m1.zero,
                      m1.c/*required*/,
                      &m1.ldc);

        for (int a1 = 0; a1 < co->size[0]; a1++) {
            libxsmm_dgemm("N",
                          "T",
                          &m2.n,
                          &m2.m,
                          &m2.k,
                          &m2.one,
                          m2.b,
                          &m2.ldb,
                          m2.a,
                          &m2.lda,
                          &m2.zero,
                          m2.c,
                          &m2.ldc);


            m2.b +=  T.offsets[0];
            m2.c +=  W.offsets[0];
        }

        libxsmm_dgemm("N",
                      "T",
                      &m3.n,
                      &m3.m,
                      &m3.k,
                      &m3.one,
                      m3.b,
                      &m3.ldb,
                      m3.a,
                      &m3.lda,
                      &m3.zero,
                      m3.c,
                      &m3.ldc);
#else
        dgemm_("N",
               "N",
               &m1.n,
               &m1.m,
               &m1.k,
               &m1.one,
               m1.b,
               &m1.ldb,
               m1.a,
               &m1.lda,
               &m1.zero,
               m1.c,
               &m1.ldc);

        for (int a1 = 0; a1 < co->size[0]; a1++) {
            dgemm_("N",
                   "T",
                   &m2.n,
                   &m2.m,
                   &m2.k,
                   &m2.one,
                   m2.b,
                   &m2.ldb,
                   &m2.a,
                   &m2.lda,
                   &m2.zero,
                   m2.c,
                   &m2.ldc);


            m2.b +=  T.offsets[0];
            m2.c +=  W.offsets[0];
        }

        dgemm_("N",
               "T",
               &m3.n,
               &m3.m,
               &m3.k,
               &m3.done,
               m3.b,
               &m3.ldb,
               m3.a,
               &m3.lda,
               &m3.zero,
               m3.c,
               &m3.ldc);
#endif
#endif

#if defined(__LIBXSMM)
        libxsmm_free(W.data);
        libxsmm_free(T.data);
#endif
    } else {
        collocate_l0(co,
                     p_alpha_beta_reduced_,
                     cube);
    }
}


/* this function needs serious rethinking. Basically we apply a spherical mask
 * on a 3D grid and then add the result on a grid with PBC */

void apply_spherical_cutoff(const double disr_radius,
                            const double dh[3][3],
                            const double dh_inv[3][3],
                            const int lb_cube[3],
                            const int cmax,
                            tensor *cube)
{
    const int kgmin = (cube->size[0] - 1) / 2;
    const int jgmin = (cube->size[1] - 1) / 2;
    const int igmin = (cube->size[2] - 1) / 2;
    const double r2 = disr_radius * disr_radius;
    const double inv_dx = 1.0 / dh[0][0];
    const double rx = dh[0][0];
    const double ry = dh[1][1];
    const double rz = dh[2][2];

    for (int kg = -kgmin; kg <= kgmin + 1; kg++) {
        double kr = ((double)kg) * rz;   // distance from center in a.u.
        kr *= kr;
        for (int jg = -jgmin; jg <= jgmin + 1; jg++) {
            double jr = ((double)jg) * ry;   // distance from center in a.u.
            jr *= jr;
            const double rest = r2 - kr - jr;
            if (rest < 1e-8) {
                memset(&idx3(cube[0], kg + kgmin, jg + jgmin, 0), 0, cube->size[2] * sizeof(double));
                continue;
            }
            double *__restrict dst =  &idx3(cube[0], kg + kgmin, jg + jgmin, 0);
            int rix = ceil(-1e-8 - sqrt(max(0.0, rest)) * inv_dx);
            for (int ig = -igmin; ig <= rix; ig ++) {
                dst[ig + igmin] = 0.0;
            }
            for (int ig = -rix + 1; ig < cube->size[2]; ig ++) {
                dst[ig + igmin] = 0.0;
            }
        }
    }
}


/* this function needs serious rethinking. Basically we apply a spherical mask
 * on a 3D grid and then add the result on a grid with PBC */

void apply_mapping(const double disr_radius,
                   const double dh[3][3],
                   const double dh_inv[3][3],
                   const int *map[3],
                   const int lb_cube[3],
                   tensor *cube,
                   const int cmax,
                   tensor *grid)
{
    const int kgmin = ceil(-1e-8 - disr_radius * dh_inv[2][2]);

    const double dz = dh[2][2];
    const double dy = dh[1][1];
    const double inv_dy = dh_inv[1][1];
    const double inv_dx = dh_inv[0][0];
    const int *__restrict map_x = map[0];
    const int *__restrict map_y = map[1];
    const int *__restrict map_z = map[2];

    for (int kg = kgmin; kg <= 1 - kgmin; kg++) {
        const int k = map_z[kg + cmax];   // target location on the grid
        const int kd = (2 * kg - 1) / 2;     // distance from center in grid points
        const double kr = kd * dz;   // distance from center in a.u.
        const double kremain = disr_radius * disr_radius - kr * kr;
        if (kremain < 0.0)
            continue;
        /* const int jgmin = ceil(-1e-8 - sqrt(max(0.0, kremain)) * inv_dy); */
        for (int jg = -cmax; jg <= cmax; jg++) {
            const int j = map_y[jg + cmax];  // target location on the grid
            const int jd = (2 * jg - 1) / 2;    // distance from center in grid points
            const double jr = jd * dy;  // distance from center in a.u.
            const double jremain = kremain - jr * jr;
            if (jremain < 0.0)
                continue;
            const int igmin = ceil(-1e-8 - sqrt(max(0.0, jremain)) * inv_dx);
            double *__restrict dst = &idx3(grid[0], k - 1, j - 1, 0);
            const double *__restrict src = &idx3(cube[0], kg - lb_cube[2], jg - lb_cube[1], - lb_cube[0]);
            for (int ig = igmin; ig <= 1 - igmin; ig++) {
                const int i = map_x[ig + cmax];  // target location on the grid
                dst[i - 1] += src[ig];
            }
        }
    }
}

/* It is a sub-optimal version of the mapping in case of a cubic cutoff. But is
 * very general and does not depend on the orthorombic nature of the grid. for
 * orthorombic cases, it is faster to apply PCB directly on the polynomials. */

void apply_mapping_cubic(const int *lower_boundaries_cube,
                         const int *cube_center,
                         const int *period,
                         const tensor *cube,
                         const int *lb_grid,
                         tensor *grid)
{
    const int startx = (lb_grid[0] + cube_center[0] + lower_boundaries_cube[0] + 32 * period[0]) % period[0];
    const int starty = (lb_grid[1] + cube_center[1] + lower_boundaries_cube[1] + 32 * period[1]) % period[1];
    const int startz = (lb_grid[2] + cube_center[2] + lower_boundaries_cube[2] + 32 * period[2]) % period[2];

    if ((starty + cube->size[1] <= grid->size[1]) &&
        (startx + cube->size[2] <= grid->size[2]) &&
        (startz + cube->size[0] <= grid->size[0])) {
        // it means that the cube is completely inside the grid without touching
        // the grid borders. periodic boundaries conditions are pointless here.
        // we can simply loop over all three dimensions.

        double *__restrict src = &idx3(cube[0], 0, 0, 0);
        /* double *__restrict dst = &idx3(grid[0], startz, starty, startx); */

        /* unpack_and_add(cube->size[0], cube->size[1], cube->size[2], cube->ld_, */
        /*                grid->size[0], grid->size[1], grid->size[2], grid->ld_, */
        /*                src, dst); */

        for (int z = 0; z < cube->size[0]; z++) {
            double *__restrict dst = &idx3(grid[0], startz + z, starty, startx);
            for (int y = 0; y < cube->size[1]; y++) {
#pragma GCC unroll 4
#pragma GCC ivdep
                for (int x = 0; x < cube->size[2]; x++) {
                    dst[x] += src[x];
                }
                src += cube->ld_;
                dst += grid->ld_;
            }
        }
        return;
    }

    int z1 = startz;

    const int offset = min(grid->size[2] - startx, cube->size[2]);
    const int loop_number = (cube->size[2] - offset) / period[0];
    const int remainder = min(grid->size[2], cube->size[2] - offset - loop_number * period[0]);
    for (int z = 0; (z < cube->size[0]); z++, z1++) {

        if (z1 >= period[2])
            z1 -= period[2];

        /* if (z1 >= lb_grid[2] + grid->size[0]) */
        /*     continue; */

        /* // We have a full plane. */

        int y1 = starty;
        for (int y = 0; y < cube->size[1]; y++, y1++) {

            if (y1 >= period[1])
                y1 -= period[1];

        /*     // this is needed when the grid is distributed over several ranks. */
            /* if (y1 >= lb_grid[1] + grid->size[1]) */
            /*     continue; */

            double *__restrict dst = &idx3(grid[0], z1, y1, 0);
            const double *__restrict src = &idx3(cube[0], z, y, 0);

            // the tail of the queue.
#pragma GCC unroll 4
#pragma GCC ivdep
            for (int x = 0; x < offset; x++)
                dst[x + startx] += src[x];

            int shift = offset;
            for (int l = 0; l < loop_number; l++) {
#pragma GCC unroll 4
#pragma GCC ivdep
                for (int x = 0; x < grid->size[2]; x++)
                    dst[x] += src[shift + x];
                shift += period[0];
            }
#pragma GCC unroll 4
#pragma GCC ivdep
            for (int x = 0; x < remainder; x++)
                dst[x] += src[shift + x];

            /* for (int x = offset; x < cube->size[2];) { */
            /*     const int size_x = min(grid->size[2], cube->size[2] - x); */
            /*     for (int x1 = 0; x1 < size_x; x1++) { */
            /*         dst[x1] += src[x1 + x]; */
            /*     } */
            /*     // I sum over the period not the grid size since the grid might */
            /*     // be partitioned. */
            /*     x += period[0]; */
            /* } */
        }
    }
}

void colloc_ortho(const int *non_zero_elements[3],
                  const int *number_of_non_zero_elements,
                  const tensor *co,
                  const tensor *pol,
                  tensor *dst)
{
    const int *__restrict z_int = non_zero_elements[2];
    const int *__restrict y_int = non_zero_elements[1];
    const int *__restrict x_int = non_zero_elements[0];

    if ((non_zero_elements[2] == 0) || (non_zero_elements[1] == 0) || (non_zero_elements[0] == 0))
        return;

    int z_offset = 0;

    for (int z = 0; z < dst->size[0]; z++) {

        for (;(z_int[z] == 0) && (z < dst->size[0] - 1); z++);
        const int zmin = z;

        for (;(z_int[z] == 1) && (z < dst->size[0] - 1); z++);
        const int zmax = z + z_int[z];

        if (zmax - zmin) {
            int y_offset = 0;
            for (int y = 0; y < dst->size[0]; y++) {

                /* if (y_int[y] == 0) */
                /*     continue; */
                /* continue itterate */
                for (;!y_int[y] && (y < dst->size[1] - 1); y++);

                const int ymin = y;

                /* continue itterate */
                for (;y_int[y] && (y < dst->size[1] - 1); y++);

                const int ymax = y + y_int[y];

                y = ymax;
                if (ymax - ymin) {
                    int x_offset = 0;
                    for (int x = 0; x < dst->size[2]; x++) {

                        /* if (x_int[x] == 0) */
                        /*     continue; */
                        /* continue itterate */
                        for (;!x_int[x] && (x < dst->size[1] - 1); x++);


                        const int xmin = x;


                        /* continue itterate */
                        for (;x_int[x] && (x < dst->size[1] - 1); x++);

                        const int xmax = x + x_int[x];
                        const int y_bound = ymax - ymin;
                        const int x_bound = xmax - xmin;

                        x = xmax;
                        if (xmax - xmin) {
                            collocate_core_rectangular1(NULL,
                                                        zmin,
                                                        ymin,
                                                        xmin,
                                                        zmax,
                                                        ymax,
                                                        xmax,
                                                        z_offset,
                                                        y_offset,
                                                        x_offset,
                                                        co,
                                                        pol,
                                                        dst);
                            x_offset += (xmax - xmin);
                        }
                    }
                    y_offset += (ymax - ymin);
                }
            }
            z_offset += (zmax - zmin);
        }
    }
}

void apply_mapping_ortho(const int *non_zero_elements[3],
                         const int *number_of_non_zero_elements,
                         const tensor *src,
                         tensor *dst)
{
    const int *__restrict z_int = non_zero_elements[2];
    const int *__restrict y_int = non_zero_elements[1];
    const int *__restrict x_int = non_zero_elements[0];

    if ((non_zero_elements[2] == 0) || (non_zero_elements[1] == 0) || (non_zero_elements[0] == 0))
        return;

    if ((number_of_non_zero_elements[2] == dst->size[0]) || (number_of_non_zero_elements[1] == dst->size[1]) || (number_of_non_zero_elements[0] == dst->size[2])) {
#if defined(__MKL) || defined(HAVE_CBLAS)
        cblas_daxpy(dst->alloc_size_, 1.0, src->data, 1, dst->data, 1);
#else
        double ONE = 1.0;
        int ione = 1;
        daxpy_(dst->alloc_size_, &ONE, src->data, &ione, dst->data, &ione);
#endif
        return;
    }

    int z_offset = 0;

    for (int z = 0; z < dst->size[0]; z++) {

        for (;!z_int[z] && (z < dst->size[0] - 1); z++);

        const int zmin = z;

        for (;z_int[z] && (z < dst->size[0] - 1); z++);

        const int zmax = z + z_int[z];

        if (zmax - zmin) {
            int y_offset = 0;
            for (int y = 0; y < dst->size[0]; y++) {

                /* if (y_int[y] == 0) */
                /*     continue; */
                /* continue itterate */
                for (;!y_int[y] && (y < dst->size[1] - 1); y++);

                const int ymin = y;

                /* continue itterate */
                for (;y_int[y] && (y < dst->size[1] - 1); y++);

                const int ymax = y + y_int[y];

                y = ymax;
                if (ymax - ymin) {

                    int x_offset = 0;
                    for (int x = 0; x < dst->size[2]; x++) {

                        /* if (x_int[x] == 0) */
                        /*     continue; */
                        /* continue itterate */
                        for (;!x_int[x] && (x < dst->size[1] - 1); x++);


                        const int xmin = x;

                        /* continue itterate */
                        for (;x_int[x] && (x < dst->size[1] - 1); x++);

                        const int xmax = x + x_int[x];
                        const int y_bound = ymax - ymin;
                        const int x_bound = xmax - xmin;

                        if (xmax - xmin) {
                            x = xmax;

                            for (int z1 = 0; z1 < (zmax - zmin); z1++) {
                                const double *__restrict src1 = &idx3(src[0], z_offset + z1, y_offset, x_offset);
                                const int ld = src->ld_;
                                double*__restrict dst1 = &idx3(dst[0], z1 + zmin, ymin, xmin);
                                for (int y1 = 0; y1 < y_bound; y1++) {
                                    for (int x1 = 0; x1 < x_bound; x1++) {
                                        dst1[x1] += src1[x1];
                                    }
                                    src1 += ld;
                                    dst1 += dst->ld_;
                                }
                            }
                            x_offset += (xmax - xmin);
                        }
                    }
                    y_offset += (ymax - ymin);
                }
            }
            z_offset += (zmax - zmin);
        }
    }
}

// *****************************************************************************
void grid_collocate_ortho(const double zetp,
                          const tensor *coef_xyz,
                          const double dh[3][3],
                          const double dh_inv[3][3],
                          const double rp[3],
                          const int npts[3],
                          const int lb_grid[3],
                          const bool periodic[3],
                          const double radius,
                          tensor *grid)
{

    // *** position of the gaussian product
    //
    // this is the actual definition of the position on the grid
    // i.e. a point rp(:) gets here grid coordinates
    // MODULO(rp(:)/dr(:),npts(:))+1
    // hence (0.0,0.0,0.0) in real space is rsgrid%lb on the rsgrid ((1,1,1) on grid)

    // cubecenter(:) = FLOOR(MATMUL(dh_inv, rp))
    int cubecenter[3];
    int lb_cube[3], ub_cube[3];
    double roffset[3];
    int* map[3];

    /* indicate the position of the non zero elements in all three directions.
     * Its length is max of the linear grid dimensions */
    int *non_zero_elements[3];

    /* number of non zero elements in each direction. */
    int number_of_non_zero_elements[3];

    /* cube : grid comtaining pointlike product between polynomials
     *
     * pol : grid  containing the polynomials in all three directions
     *
     * pol_folded : grid containing the polynomials after folding for periodic
     * boundaries conditions
     */

    tensor pol, cube, pol_folded;

    /* used to know how large should be the polynomial grid */
    int cmax = INT_MIN;

    /* seting up the cube parameters */

    /* cube center */
    for (int i=0; i<3; i++) {
        double dh_inv_rp = 0.0;
        for (int j=0; j<3; j++) {
            dh_inv_rp += dh_inv[j][i] * rp[j];
        }
        cubecenter[i] = floor(dh_inv_rp);
    }

    /* lower and upper bounds */

    // Historically, the radius gets discretized.
    const double drmin = min(dh[0][0], min(dh[1][1], dh[2][2]));
    const double disr_radius = drmin * max(1, ceil(radius/drmin));

    for (int i = 0; i < 3; i++) {
        lb_cube[i] = ceil(-1e-8 - disr_radius * dh_inv[i][i]);
        ub_cube[i] = 1 - lb_cube[i];
    }

    /* compute the cube size ignoring periodicity */
    const int cube_size[3] = {ub_cube[2] - lb_cube[2] + 1,
                              ub_cube[1] - lb_cube[1] + 1,
                              ub_cube[0] - lb_cube[0] + 1};



    for (int i=0; i<3; i++) {
        roffset[i] = rp[i] - ((double) cubecenter[i]) * dh[i][i];
    }

    for (int i = 0; i < 3; i++) {
        cmax = max(cmax, ub_cube[i]);
    }

    /* initialize the multidimensional array containing the polynomials */
    initialize_tensor_3(&pol, 3, coef_xyz->size[0], 2 * cmax + 1);

/* a mapping so that the ig corresponds to the right grid point */
#if defined(__LIBXSMM)
    pol.data = libxsmm_aligned_scratch(sizeof(double) * pol.alloc_size_, 0/*auto-alignment*/);

    // normally it is 2 * cmax + 1, but for alignment reason it is probably
    // better to align on 16 bytes

    //
    //  map[0] =  libxsmm_aligned_scratch(sizeof(int) * 3 * (2 * cmax + 1), 0/*auto-alignment*/);
    //  map[1] = map[0] + (2 * cmax + 1);
    //  map[2] = map[1] + (2 * cmax + 1);
    //
#else
    int map[3][2 * cmax + 1];
#endif
    /* memset(map[0], 0, sizeof(int) * 3 * (2 * cmax + 1)); */

    /* /\* compute the mapping *\/ */
    /* for (int i = 0; i < 3; i++) { */
    /*     grid_fill_map(periodic[i], */
    /*                   lb_cube[i], */
    /*                   ub_cube[i], */
    /*                   cubecenter[i], */
    /*                   lb_grid[i], */
    /*                   npts[i], */
    /*                   grid->size[i], */
    /*                   cmax, */
    /*                   map[i]); */
    /* } */


    /* compute the polynomials */
    memset(pol.data, 0, sizeof(double) * pol.alloc_size_);

    // WARNING : do not reverse the order in pol otherwise you will have to
    // reverse the order in collocate_dgemm as well.

    grid_fill_pol(dh[0][0], roffset[0], lb_cube[0], coef_xyz->size[0] - 1, cmax, zetp, &idx3(pol, 0, 0, 0)); /* i indice */
    grid_fill_pol(dh[1][1], roffset[1], lb_cube[1], coef_xyz->size[1] - 1, cmax, zetp, &idx3(pol, 1, 0, 0)); /* j indice */
    grid_fill_pol(dh[2][2], roffset[2], lb_cube[2], coef_xyz->size[2] - 1, cmax, zetp, &idx3(pol, 2, 0, 0)); /* k indice */

    if (TEST) {
        int size_max = max(grid->size[0],
                           max(grid->size[1],
                               grid->size[2]));

#if defined(__LIBXSMM)
        non_zero_elements[0] = libxsmm_aligned_scratch(sizeof(double) * 3 * size_max, 0/*auto-alignment*/);
#else
#error "Need implementation"
#endif

        non_zero_elements[1] = non_zero_elements[0] + size_max;
        non_zero_elements[2] = non_zero_elements[1] + size_max;

        /* folds the polynomials and compute the number of non zero elements at the
         * same time. It also reorders the data such that adding them to the grid is
         * simple */
        apply_pcb_orthogonal_case(cubecenter,
                                  cube_size,
                                  npts,
                                  grid->size,
                                  lb_grid,
                                  &pol,
                                  non_zero_elements,
                                  number_of_non_zero_elements,
                                  &pol_folded);

        if ((number_of_non_zero_elements[0] == 0) ||
            (number_of_non_zero_elements[1] == 0) ||
            (number_of_non_zero_elements[2] == 0)) {
            const int startx = (lb_grid[0] + cubecenter[0] + lb_cube[0] + 32 * npts[0]) % npts[0];
            const int starty = (lb_grid[1] + cubecenter[1] + lb_cube[1] + 32 * npts[1]) % npts[1];
            const int startz = (lb_grid[2] + cubecenter[2] + lb_cube[2] + 32 * npts[2]) % npts[2];
            printf("We have a problem: Info\n");
            printf("grid size                   : %d %d %d\n", grid->size[0], grid->size[1], grid->size[2]);
            printf("cube size                   : %d %d %d\n", cube_size[0], cube_size[1], cube_size[2]);
            printf("number of non zero elements : %d %d %d\n", number_of_non_zero_elements[0], number_of_non_zero_elements[1], number_of_non_zero_elements[2]);
            printf("cube center                 : %d %d %d\n", cubecenter[0], cubecenter[1], cubecenter[2]);
            printf("cube center (rel. coord)    : %d %d %d\n", startx, starty, startz);
            exit(1);
        }
        // grid[k][j][i]

        /*
         * initialize the cube containing the cartesian product of the polynomials
         * yes it is reverse order. it is natural to have xyz but we store things in
         * the format zyx. the cube containing the result is stored as C_{z,y,x}
         */

        initialize_tensor_3(&cube,
                            number_of_non_zero_elements[2],
                            number_of_non_zero_elements[1],
                            number_of_non_zero_elements[0]);
#if defined(__LIBXSMM)
        libxsmm_free(pol_folded.data);
#endif
    } else {
        initialize_tensor_3(&cube,
                            cube_size[0],
                            cube_size[1],
                            cube_size[2]);
    }

#if defined(__LIBXSMM)
        cube.data = libxsmm_aligned_scratch(sizeof(double) * cube.alloc_size_, 0/*auto-alignment*/);
#else
#error "Need implementation"
#endif


        /* memset(cube.data, 0, cube.alloc_size_ * sizeof(double)); */

    /* see comments in the function to have a idea of what is going on */

        if (TEST) {
            colloc_ortho(non_zero_elements,
                         number_of_non_zero_elements,
                         coef_xyz,
                         &pol_folded,
                         grid);

            /* collocate_core_rectangular(NULL, // will need to change that eventually. */
            /*                            // pointer to scratch memory */
            /*                            coef_xyz, */
            /*                            &pol_folded, */
            /*                            &cube); */

            //
            // Write cube back to large grid taking periodicity and radius into account.
            //

            // The cube contains an even number of grid points in each direction and
            // collocation is always performed on a pair of two opposing grid points.
            // Hence, the points with index 0 and 1 are both assigned distance zero via
            // the formular distance=(2*index-1)/2.

            /* apply_mapping(disr_radius, dh, dh_inv, map, lb_cube, &cube, cmax, grid); */

            /* apply_mapping_ortho(non_zero_elements, */
            /*                     number_of_non_zero_elements, */
            /*                     &cube, */
            /*                     grid); */
        } else {
            collocate_core_rectangular(NULL, // will need to change that eventually.
                                       // pointer to scratch memory
                                       coef_xyz,
                                       &pol,
                                       &cube);

            /* apply_mapping(disr_radius, dh, dh_inv, map, lb_cube, &cube, cmax, grid); */
            apply_mapping_cubic(lb_cube, cubecenter, npts, &cube,
                                lb_grid, grid);
        }

#if defined(__LIBXSMM)
    libxsmm_free(cube.data);
    libxsmm_free(pol.data);
    /* libxsmm_free(map[0]); */
    if (TEST)
        libxsmm_free(non_zero_elements[0]);
#endif
}


// *****************************************************************************
static void grid_collocate_general(const int lp,
                                   const double zetp,
                                   const tensor *coef_xyz,
                                   const double dh[3][3],
                                   const double dh_inv[3][3],
                                   const double rp[3],
                                   const int npts[3],
                                   const int lb_grid[3],
                                   const bool periodic[3],
                                   const double radius,
                                   const int ngrid[3],
                                   tensor *grid)
{

// Translated from collocate_general_opt()
//
// transform P_{lxp,lyp,lzp} into a P_{lip,ljp,lkp} such that
// sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-x_p)**lxp (y-y_p)**lyp (z-z_p)**lzp =
// sum_{lip,ljp,lkp} P_{lip,ljp,lkp} (i-i_p)**lip (j-j_p)**ljp (k-k_p)**lkp
//

    // aux mapping array to simplify life
    //TODO instead of this map we could use 3D arrays like coef_xyz.
    int coef_map[lp+1][lp+1][lp+1];

    //TODO really needed?
    //coef_map = HUGE(coef_map)
    for (int lzp=0; lzp<=lp; lzp++) {
        for (int lyp=0; lyp<=lp; lyp++) {
            for (int lxp=0; lxp<=lp; lxp++) {
                coef_map[lzp][lyp][lxp] = INT_MAX;
            }
        }
    }

    int lxyz = 0;
    for (int lzp=0; lzp<=lp; lzp++) {
        for (int lyp=0; lyp<=lp-lzp; lyp++) {
            for (int lxp=0; lxp<=lp-lzp-lyp; lxp++) {
                coef_map[lzp][lyp][lxp] = ++lxyz;
            }
        }
    }

    // center in grid coords
    // gp = MATMUL(dh_inv, rp)
    double gp[3];
    for (int i=0; i<3; i++) {
        gp[i] = 0.0;
        for (int j=0; j<3; j++) {
            gp[i] += dh_inv[j][i] * rp[j];
        }
    }

    // transform using multinomials
    double hmatgridp[lp+1][3][3];
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            hmatgridp[0][j][i] = 1.0;
            for (int k=1; k<=lp; k++) {
                hmatgridp[k][j][i] = hmatgridp[k-1][j][i] * dh[j][i];
            }
        }
    }

    // zero coef_ijk
    const int ncoef_ijk = ((lp+1)*(lp+2)*(lp+3))/6;
    double coef_ijk[ncoef_ijk];
    for (int i=0; i<ncoef_ijk; i++) {
        coef_ijk[i] = 0.0;
    }

    const int lpx = lp;
    for (int klx=0; klx<=lpx; klx++) {
        for (int jlx=0; jlx<=lpx-klx; jlx++) {
            for (int ilx=0; ilx<=lpx-klx-jlx; ilx++) {
                const int lx = ilx + jlx + klx;
                const int lpy = lp - lx;
                for (int kly=0; kly<=lpy; kly++) {
                    for (int jly=0; jly<=lpy-kly; jly++) {
                        for (int ily=0; ily<=lpy-kly-jly; ily++) {
                            const int ly = ily + jly + kly;
                            const int lpz = lp - lx - ly;
                            for (int klz=0; klz<=lpz; klz++) {
                                for (int jlz=0; jlz<=lpz-klz; jlz++) {
                                    for (int ilz=0; ilz<=lpz-klz-jlz; ilz++) {
                                        const int lz = ilz + jlz + klz;
                                        const int il = ilx + ily + ilz;
                                        const int jl = jlx + jly + jlz;
                                        const int kl = klx + kly + klz;
                                        const int lijk= coef_map[kl][jl][il];
                                        coef_ijk[lijk-1] += idx3(coef_xyz[0], lz, ly, lx) *
                                            hmatgridp[ilx][0][0] * hmatgridp[jlx][1][0] * hmatgridp[klx][2][0] *
                                            hmatgridp[ily][0][1] * hmatgridp[jly][1][1] * hmatgridp[kly][2][1] *
                                            hmatgridp[ilz][0][2] * hmatgridp[jlz][1][2] * hmatgridp[klz][2][2] *
                                            fac[lx] * fac[ly] * fac[lz] /
                                            (fac[ilx] * fac[ily] * fac[ilz] * fac[jlx] * fac[jly] * fac[jlz] * fac[klx] * fac[kly] * fac[klz]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // CALL return_cube_nonortho(cube_info, radius, index_min, index_max, rp)
    //
    // get the min max indices that contain at least the cube that contains a sphere around rp of radius radius
    // if the cell is very non-orthogonal this implies that many useless points are included
    // this estimate can be improved (i.e. not box but sphere should be used)
    int index_min[3], index_max[3];
    for (int idir=0; idir<3; idir++) {
        index_min[idir] = INT_MAX;
        index_max[idir] = INT_MIN;
    }
    for (int i=-1; i<=1; i++) {
        for (int j=-1; j<=1; j++) {
            for (int k=-1; k<=1; k++) {
                const double x = rp[0] + i * radius;
                const double y = rp[1] + j * radius;
                const double z = rp[2] + k * radius;
                for (int idir=0; idir<3; idir++) {
                    const double resc = dh_inv[0][idir] * x + dh_inv[1][idir] * y + dh_inv[2][idir] * z;
                    index_min[idir] = min(index_min[idir], floor(resc));
                    index_max[idir] = max(index_max[idir], ceil(resc));
                }
            }
        }
    }

    int offset[3];
    for (int idir=0; idir<3; idir++) {
        offset[idir] = mod(index_min[idir] + lb_grid[idir], npts[idir]) + 1;
    }

    // go over the grid, but cycle if the point is not within the radius
    for (int k=index_min[2]; k<=index_max[2]; k++) {
        const double dk = k - gp[2];
        int k_index;
        if (periodic[2]) {
            k_index = mod(k, npts[2]) + 1;
        } else {
            k_index = k - index_min[2] + offset[2];
        }

        // zero coef_xyt
        const int ncoef_xyt = ((lp+1)*(lp+2))/2;
        double coef_xyt[ncoef_xyt];
        for (int i=0; i<ncoef_xyt; i++) {
            coef_xyt[i] = 0.0;
        }

        int lxyz = 0;
        double dkp = 1.0;
        for (int kl=0; kl<=lp; kl++) {
            int lxy = 0;
            for (int jl=0; jl<=lp-kl; jl++) {
                for (int il=0; il<=lp-kl-jl; il++) {
                    coef_xyt[lxy++] += coef_ijk[lxyz++] * dkp;
                }
                lxy += kl;
            }
            dkp *= dk;
        }


        for (int j=index_min[1]; j<=index_max[1]; j++) {
            const double dj = j - gp[1];
            int j_index;
            if (periodic[1]) {
                j_index = mod(j, npts[1]) + 1;
            } else {
                j_index = j - index_min[1] + offset[1];
            }

            double coef_xtt[lp+1];
            for (int i=0; i<=lp; i++) {
                coef_xtt[i] = 0.0;
            }
            int lxy = 0;
            double djp = 1.0;
            for (int jl=0; jl<=lp; jl++) {
                for (int il=0; il<=lp-jl; il++) {
                    coef_xtt[il] += coef_xyt[lxy++] * djp;
                }
                djp *= dj;
            }

            // find bounds for the inner loop
            // based on a quadratic equation in i
            // a*i**2+b*i+c=radius**2

            // v = pointj-gp(1)*hmatgrid(:, 1)
            // a = DOT_PRODUCT(hmatgrid(:, 1), hmatgrid(:, 1))
            // b = 2*DOT_PRODUCT(v, hmatgrid(:, 1))
            // c = DOT_PRODUCT(v, v)
            // d = b*b-4*a*(c-radius**2)
            double a=0.0, b=0.0, c=0.0;
            for (int i=0; i<3; i++) {
                const double pointk = dh[2][i] * dk;
                const double pointj = pointk + dh[1][i] * dj;
                const double v = pointj - gp[0] * dh[0][i];
                a += dh[0][i] * dh[0][i];
                b += 2.0 * v * dh[0][i];
                c += v * v;
            }
            double d = b * b -4 * a * (c - radius * radius);
            if (d < 0.0) {
                continue;
            }

            // prepare for computing -zetp*rsq
            d = sqrt(d);
            const int ismin = ceill((-b-d)/(2.0*a));
            const int ismax = floor((-b+d)/(2.0*a));
            a *= -zetp;
            b *= -zetp;
            c *= -zetp;
            const int i = ismin - 1;

            // the recursion relation might have to be done
            // from the center of the gaussian (in both directions)
            // instead as the current implementation from an edge
            double exp2i = exp((a * i + b) * i + c);
            double exp1i = exp(2.0 * a * i + a + b);
            const double exp0i = exp(2.0 * a);

            for (int i=ismin; i<=ismax; i++) {
                const double di = i - gp[0];

                // polynomial terms
                double res = 0.0;
                double dip = 1.0;
                for (int il=0; il<=lp; il++) {
                    res += coef_xtt[il] * dip;
                    dip *= di;
                }

                // the exponential recursion
                exp2i *= exp1i;
                exp1i *= exp0i;
                res *= exp2i;

                int i_index;
                if (periodic[0]) {
                    i_index = mod(i, npts[0]) + 1;
                } else {
                    i_index = i - index_min[0] + offset[0];
                }
                idx3(grid[0], k_index - 1, j_index - 1, i_index - 1) += res;
            }
        }
    }
}

// *****************************************************************************
   void grid_collocate_internal(const bool use_ortho,
                                const int func,
                                const int *lmax,
                                const int *lmin,
                                const double zeta,
                                const double zetb,
                                const double rscale,
                                    const double dh[3][3],
                                    const double dh_inv[3][3],
                                    const double ra[3],
                                    const double rab[3],
                                    const int npts[3],
                                    const int ngrid[3],
                                    const int lb_grid[3],
                                    const bool periodic[3],
                                    const double radius,
                                    const int *offsets,
                                    const int *pab_size, // n1, n2
                                    const double pab[pab_size[1]][pab_size[0]],
                                    tensor *grid){

    const double zetp = zeta + zetb;
    const double f = zetb / zetp;
    const double rab2 = rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2];
    const double prefactor = rscale * exp(-zeta * f * rab2);
    double rp[3], rb[3];
    for (int i=0; i<3; i++) {
        rp[i] = ra[i] + f * rab[i];
        rb[i] = ra[i] + rab[i];
    }

    int lmin_diff[2], lmax_diff[2];
    grid_prepare_get_ldiffs(func, lmin_diff, lmax_diff);

    int lmin_prep[2];
    int lmax_prep[2];

    lmin_prep[0] = max(lmin[0] + lmin_diff[0], 0);
    lmin_prep[1] = max(lmin[1] + lmin_diff[1], 0);

    lmax_prep[0] = lmax[0] + lmax_diff[0];
    lmax_prep[1] = lmax[1] + lmax_diff[1];

    const int n1_prep = ncoset[lmax_prep[0]];
    const int n2_prep = ncoset[lmax_prep[1]];

    /* I really do not like this. This will disappear */

    double pab_prep[n2_prep][n1_prep];

    memset(pab_prep, 0, n2_prep * n1_prep * sizeof(double));

    grid_prepare_pab(func, offsets[0], offsets[1], lmax,
                     lmin, zeta, zetb, pab_size[0], pab_size[1], pab, n1_prep, n2_prep, pab_prep);

    //   *** initialise the coefficient matrix, we transform the sum
    //
    // sum_{lxa,lya,lza,lxb,lyb,lzb} P_{lxa,lya,lza,lxb,lyb,lzb} *
    //         (x-a_x)**lxa (y-a_y)**lya (z-a_z)**lza (x-b_x)**lxb (y-a_y)**lya (z-a_z)**lza
    //
    // into
    //
    // sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-p_x)**lxp (y-p_y)**lyp (z-p_z)**lzp
    //
    // where p is center of the product gaussian, and lp = la_max + lb_max
    // (current implementation is l**7)
    //

    tensor alpha;
    initialize_tensor_4(&alpha, 3, lmax_prep[1] + 1, lmax_prep[0] + 1, lmax_prep[0] + lmax_prep[1] + 1);
#ifdef __LIBXSMM
    alpha.data = libxsmm_aligned_scratch(sizeof(double) * alpha.alloc_size_, 0/*auto-alignment*/);
#else
    alpha.data = (double*)tmp;
    tmp += alpha.alloc_size_ * sizeof(double);
#endif


    const int lp = lmax_prep[0] + lmax_prep[1];
    tensor coef_xyz;
    initialize_tensor_3(&coef_xyz, lp + 1, lp + 1, lp + 1);
#ifdef __LIBXSMM
    coef_xyz.data = libxsmm_aligned_scratch(sizeof(double) * coef_xyz.alloc_size_, 0/*auto-alignment*/);
#else
    coef_xyz.data = (double*)tmp;
    tmp += coef_xyz.alloc_size_ * sizeof(double);
#endif

    // initialy cp2k stores coef_xyz as coef[z][y][x]. this is fine but I
    // need them to be stored as

    grid_prepare_alpha(ra,
                       rb,
                       rp,
                       lmax_prep,
                       &alpha);

    //
    //   compute P_{lxp,lyp,lzp} given P_{lxa,lya,lza,lxb,lyb,lzb} and alpha(ls,lxa,lxb,1)
    //   use a three step procedure
    //   we don't store zeros, so counting is done using lxyz,lxy in order to have
    //   contiguous memory access in collocate_fast.F
    //

    // coef[x][z][y]
    grid_prepare_coef_ortho(lmax_prep,
                            lmin_prep,
                            lp,
                            prefactor,
                            &alpha,
                            pab_prep,
                            &coef_xyz);

    if (use_ortho) {

        // coef[x][z][y]
        grid_prepare_coef_ortho(lmax_prep,
                                lmin_prep,
                                lp,
                                prefactor,
                                &alpha,
                                pab_prep,
                                &coef_xyz);

        grid_collocate_ortho(zetp,
                             &coef_xyz,
                             dh,
                             dh_inv,
                             rp,
                             npts,
                             lb_grid,
                             periodic,
                             radius,
                             grid);
    } else {

        // coef[x][z][y]
        grid_prepare_coef(lmax_prep,
                          lmin_prep,
                          lp,
                          prefactor,
                          &alpha,
                          pab_prep,
                          &coef_xyz);
// initialy cp2k stores coef_xyz as coef[z][y][x]. this is fine but I
        // need them to be stored as

        grid_collocate_general(lp,
                               zetp,
                               &coef_xyz,
                               dh,
                               dh_inv,
                               rp,
                               npts,
                               lb_grid,
                               periodic,
                               radius,
                               ngrid,
                               grid);
    }

#if defined(__LIBXSMM)
    libxsmm_free(coef_xyz.data);
    libxsmm_free(alpha.data);
#endif
}


// *****************************************************************************
void grid_collocate_pgf_product_cpu(const bool use_ortho,
                                    const int func,
                                    const int la_max,
                                    const int la_min,
                                    const int lb_max,
                                    const int lb_min,
                                    const double zeta,
                                    const double zetb,
                                    const double rscale,
                                    const double dh[3][3],
                                    const double dh_inv[3][3],
                                    const double ra[3],
                                    const double rab[3],
                                    const int npts[3],
                                    const int ngrid[3],
                                    const int lb_grid[3],
                                    const bool periodic[3],
                                    const double radius,
                                    const int o1,
                                    const int o2,
                                    const int n1,
                                    const int n2,
                                    const double pab[n2][n1],
                                    double *grid_)
{

// Uncomment this to dump all tasks to file.
// #define __GRID_DUMP_TASKS
    tensor grid;
    char *scratch = NULL;
    int offset[2] = {o1, o2};
    int pab_size[2] = {n2, n1};

    int lmax[2] = {la_max, lb_max};
    int lmin[2] = {la_min, lb_min};

    initialize_tensor_3(&grid, ngrid[2], ngrid[1], ngrid[0]);
    grid.ld_ = ngrid[0];
    grid.data = grid_;

#ifdef __GRID_DUMP_TASKS

    tensor grid_before;
    initialize_tensor_3(&grid_before, ngrid[2], ngrid[1], ngrid[0]);
#ifdef __LIBXSMM
    grid_before.data = libxsmm_aligned_scratch(sizeof(double) * coef_xyz.alloc_size_, 0/*auto-alignment*/);
#else
    // we have a buffer of 4 M
    posix_memalign((void**)&scratch, sizeof(double) * 1024 * 1024 * 4);
    grid_before.data = (double*)scratch;
    scratch += coef_xyz.alloc_size_ * sizeof(double);
#endif

    for (int i = 0; i < ngrid[2]; i++) {
        for (int j = 0; j < ngrid[1]; j++) {
            for (int k = 0; k < ngrid[0]; k++) {
                idx3(grid_before, i, j, k) = idx3(grid, i, j, k);
            }
        }
    }
    memset(grid.data, 0, sizeof(double) grid.alloc_size_);
#endif

    grid_collocate_internal(use_ortho,
                            func,
                            lmax,
                            lmin,
                            zeta,
                            zetb,
                            rscale,
                            dh,
                            dh_inv,
                            ra,
                            rab,
                            npts,
                            ngrid,
                            lb_grid,
                            periodic,
                            radius,
                            offset,
                            pab_size,
                            pab,
                            &grid);

#ifdef __GRID_DUMP_TASKS

    grid_collocate_record(use_ortho,
                          func,
                          la_max,
                          la_min,
                          lb_max,
                          lb_min,
                          zeta,
                          zetb,
                          rscale,
                          dh,
                          dh_inv,
                          ra,
                          rab,
                          npts,
                          ngrid,
                          lb_grid,
                          periodic,
                          radius,
                          o1,
                          o2,
                          n1,
                          n2,
                          pab,
                          grid);

    cblas_daxpy(grid->alloc_size_, 1.0, grid_before->data, 1, grid->data, 1);

#endif

}

//EOF
