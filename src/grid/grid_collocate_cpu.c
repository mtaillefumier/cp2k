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
#include <mkl_cblas.h>
#include <libxsmm.h>
#include "grid_collocate_replay.h"
#include "grid_collocate_cpu.h"
#include "grid_prepare_pab.h"
#include "grid_common.h"
#include "tensor_local.h"

void collocate_core_rectangular(char *scratch,
                                const struct tensor_ *co,
                                const struct tensor_ *p_alpha_beta_reduced_,
                                struct tensor_ *Vtmp);

inline int return_length_l(const int l) {
    static const int length_[] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55};
    return length_[l];
}

inline int return_offset_l(const int l) {
    static const int offset_[10] = {0, 1, 4, 10, 20, 35, 56, 84, 120, 165};
    return offset_[l];
}

inline int return_linear_index_from_exponents(const int alpha, const int beta,
                                              const int gamma) {
    const int l = alpha + beta + gamma;
    return return_offset_l(l) + (l - alpha) * (l - alpha + 1) / 2 + gamma;
}

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


// *****************************************************************************
static void grid_prepare_alpha(const double ra[3],
                               const double rb[3],
                               const double rp[3],
                               const int *lmax,
                               tensor *alpha)
{
    // Initialize with zeros.
    memset(alpha->data, 0, alpha->alloc_size_ * sizeof(double));

    //
    //   compute polynomial expansion coefs -> (x-a)**lxa (x-b)**lxb -> sum_{ls} alpha(ls,lxa,lxb,1)*(x-p)**ls
    //

    for (int iaxis=0; iaxis<3; iaxis++) {
        const double drpa = rp[iaxis] - ra[iaxis];
        const double drpb = rp[iaxis] - rb[iaxis];
        for (int lxa = 0; lxa <= lmax[0]; lxa++) {
            for (int lxb = 0; lxb <= lmax[1]; lxb++) {
                double binomial_k_lxa = 1.0;
                double a = 1.0;
                for (int k = 0; k <= lxa; k++) {
                    double binomial_l_lxb = 1.0;
                    double b = 1.0;
                    for (int l = 0; l <= lxb; l++) {
                        idx4(alpha[0], iaxis, lxb, lxa, lxa - l + lxb - k) += binomial_k_lxa * binomial_l_lxb * a * b;
                        binomial_l_lxb *= ((double)(lxb - l)) / ((double)(l + 1));
                        b *= drpb;
                    }
                    binomial_k_lxa *= ((double)(lxa-k)) / ((double)(k+1));
                    a *= drpa;
                }
            }
        }
    }
}


// *****************************************************************************
static void grid_prepare_coef(const int *lmax,
                              const int *lmin,
                              const int lp,
                              const double prefactor,
                              const tensor *alpha,
                              const double pab[ncoset[lmax[1]]][ncoset[lmax[0]]],
                              tensor *coef_xyz)
{

    memset(coef_xyz->data, 0, coef_xyz->alloc_size_ * sizeof(double));

    // we need a proper fix for that. We can use the tensor structure for this

    double coef_xyt[lp+1][lp+1];
    double coef_xtt[lp+1];

    for (int lzb = 0; lzb<=lmax[1]; lzb++) {
        for (int lza = 0; lza<=lmax[0]; lza++) {
            for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
                for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
                    coef_xyt[lyp][lxp] = 0.0;
                }
            }
            for (int lyb = 0; lyb<=lmax[1]-lzb; lyb++) {
                for (int lya = 0; lya<=lmax[0]-lza; lya++) {
                    const int lxpm = (lmax[1]-lzb-lyb) + (lmax[0]-lza-lya);
                    for (int i=0; i<=lxpm; i++) {
                        coef_xtt[i] = 0.0;
                    }
                    for (int lxb = max(lmin[1]-lzb-lyb, 0); lxb<=lmax[1]-lzb-lyb; lxb++) {
                        for (int lxa = max(lmin[0]-lza-lya, 0); lxa<=lmax[0]-lza-lya; lxa++) {
                            const int ico = coset(lxa, lya, lza);
                            const int jco = coset(lxb, lyb, lzb);
                            const double p_ele = prefactor * pab[jco][ico];
                            for (int lxp = 0; lxp<=lxa+lxb; lxp++) {
                                coef_xtt[lxp] += p_ele * idx4(alpha[0], 0, lxb, lxa, lxp);
                            }
                        }
                    }
                    for (int lyp = 0; lyp<=lya+lyb; lyp++) {
                        for (int lxp = 0; lxp<=lp-lza-lzb-lya-lyb; lxp++) {
                            coef_xyt[lyp][lxp] += idx4(alpha[0], 1, lyb, lya, lyp) * coef_xtt[lxp];
                        }
                    }
                }
            }
            for (int lzp = 0; lzp<=lza+lzb; lzp++) {
                for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
                    for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
                        idx3(coef_xyz[0], lzp, lyp, lxp) += idx4(alpha[0], 2, lzb, lza, lzp) * coef_xyt[lyp][lxp];
                    }
                }
            }
        }
    }
}

// *****************************************************************************
static void grid_prepare_coef_ortho(const int *lmax,
                                    const int *lmin,
                                    const int lp,
                                    const double prefactor,
                                    const tensor *alpha, // [3][lb_max+1][la_max+1][lp+1]
                                    const double pab[ncoset[lmax[1]]][ncoset[lmax[0]]],
                                    tensor *coef_xyz) //[lp+1][lp+1][lp+1]
{


    memset(coef_xyz->data, 0, coef_xyz->alloc_size_ * sizeof(double));

    // we need a proper fix for that. We can use the tensor structure for this

    double coef_xyt[lp+1][lp+1];
    double coef_xtt[lp+1];

    for (int lzb = 0; lzb<=lmax[1]; lzb++) {
        for (int lza = 0; lza<=lmax[0]; lza++) {
            for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
                for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
                    coef_xyt[lyp][lxp] = 0.0;
                }
            }
            for (int lyb = 0; lyb<=lmax[1]-lzb; lyb++) {
                for (int lya = 0; lya<=lmax[0]-lza; lya++) {
                    const int lxpm = (lmax[1]-lzb-lyb) + (lmax[0]-lza-lya);
                    for (int i=0; i<=lxpm; i++) {
                        coef_xtt[i] = 0.0;
                    }
                    for (int lxb = max(lmin[1]-lzb-lyb, 0); lxb<=lmax[1]-lzb-lyb; lxb++) {
                        for (int lxa = max(lmin[0]-lza-lya, 0); lxa<=lmax[0]-lza-lya; lxa++) {
                            const int ico = coset(lxa, lya, lza);
                            const int jco = coset(lxb, lyb, lzb);
                            const double p_ele = prefactor * pab[jco][ico];
                            for (int lxp = 0; lxp<=lxa+lxb; lxp++) {
                                coef_xtt[lxp] += p_ele * idx4(alpha[0], 0, lxb, lxa, lxp);
                            }
                        }
                    }
                    for (int lyp = 0; lyp<=lya+lyb; lyp++) {
                        for (int lxp = 0; lxp<=lp-lza-lzb-lya-lyb; lxp++) {
                            coef_xyt[lyp][lxp] += idx4(alpha[0], 1, lyb, lya, lyp) * coef_xtt[lxp];
                        }
                    }
                }
            }
            for (int lzp = 0; lzp<=lza+lzb; lzp++) {
                for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
                    for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
                        idx3(coef_xyz[0], lxp, lyp, lzp) += idx4(alpha[0], 2, lzb, lza, lzp) * coef_xyt[lyp][lxp];
                    }
                }
            }
        }
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
        //for (int i=0; i <= 2*cmax; i++)
        //    map[i] = mod(cubecenter + i - cmax, npts) + 1;
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
    initialize_tensor_2(&pol, lp + 1, 2 * cmax + 2);
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

#if defined(LIBXSMM)
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
        m2.a = &idx3(p_alpha_beta_reduced_[0], 0, 0, 0); // p_alpha_beta_reduced(0, gamma, j)
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
        m3.b = &idx3(p_alpha_beta_reduced_[0], 2, 0, 0); // p_alpha_beta_reduced(2, alpha, i)
        m3.ldb = p_alpha_beta_reduced_->ld_;
        m3.c = &idx3(cube[0], 0, 0, 0); // cube_{kji}
        m3.ldc = cube->ld_;


#if defined(HAVE_CBLAS) || defined(HAVE_MKL)
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
         * there is a strick (thans Hans) to get it is right with matrix-matrix
         * multiplication in row major format but calling fortran interfaces of
         * dgemm
         *
         * TA, TB, M, N, K, A, LDA, B, LDB (cblas_dgemm, row major)
         * Changes to
         * TB, TA, N, M, K, B, LDB, A, LDA (dgemm, col major)
         *
         */

// we need to replace this with libxsmm
        dgemm_("N",
               "N",
               &m1.m,
               &m1.n,
               &m1.k,
               &m1.done,
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
                   &m2.done,
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

#if defined(SCRATCH)
        libxsmm_free(W.data);
        libxsmm_free(T.data);
#endif

    } else {
        const double *__restrict pz = &idx3(p_alpha_beta_reduced_[0], 0, 0, 0); /* k indice */
        const double *__restrict py = &idx3(p_alpha_beta_reduced_[0], 1, 0, 0); /* j indice */
        const double *__restrict px = &idx3(p_alpha_beta_reduced_[0], 2, 0, 0); /* i indice */
        double *__restrict dst = &idx3(cube[0], 0, 0, 0);
        const double coo = idx3 (co[0], 0, 0, 0);
        for (int z1 = 0; z1 < cube->size[0]; z1++) {
            const double tz = coo * pz[z1];
            for (int y1 = 0; y1 < cube->size[1]; y1++) {
                const double tmp = tz * py[y1];
                for (int x1 = 0; x1 < cube->size[2]; x1++) {
                    dst[x1] = tmp * px[x1];
                }
                dst += cube->ld_;
            }
        }
    }
}

/* this function needs serious rethinking. Basically we apply a spherical mask
 * on a 3D grid and then add the result on a grid with PBC */

void apply_mapping_variant(const double disr_radius,
                           const double dh[3][3],
                           const double dh_inv[3][3],
                           const int *map[3],
                           const int lb_cube[3],
                           tensor *cube,
                           const int cmax,
                           tensor *grid)
{
    const int kgmin = ceil(-1e-8 - disr_radius * dh_inv[2][2]);
    const int jgmin = ceil(-1e-8 - disr_radius * dh_inv[1][1]);
    const int igmin = ceil(-1e-8 - disr_radius * dh_inv[0][0]);
    const double r2 = disr_radius * disr_radius;
    const double invrx = 1.0 / dh[0][0];
    const double rx = dh[0][0];
    const double ry = dh[1][1];
    const double rz = dh[2][2];

    for (int kg = kgmin; kg <= 1 - kgmin; kg++) {
        const int kd = (2 * kg - 1) / 2 + kgmin;     // distance from center in grid points
        const double kr = kd * rz;   // distance from center in a.u.
        for (int jg = jgmin; jg <= 1 - jgmin; jg++) {
            const int jd = (2 * jg - 1) / 2;     // distance from center in grid points
            const double jr = jd * ry;   // distance from center in a.u.
            const double rest = r2 - kr * kr - jr * jr;
            double *__restrict dst =  &idx3(cube[0], kg - lb_cube[2], jgmin - lb_cube[1],  -lb_cube[0]);
            for (int ig = igmin; ig <= 1 - igmin; ig ++) {
                const int id = (2 * ig - 1) / 2 + igmin;     // distance from center in grid points
                if ((rest - id * rx * id * rx) < 0.0)
                    dst[ig] = 0.0;
            }
        }
    }

    for (int kg = kgmin; kg <= 1 - kgmin; kg++) {
        const int k = map[2][kg + cmax];   // target location on the grid
        for (int jg = jgmin; jg <= 1 - jgmin; jg++) {
            const int j = map[1][jg + cmax];  // target location on the grid
            double *__restrict dst = &idx3(grid[0], k - 1, j - 1, 0);
            const double *__restrict src = &idx3(cube[0], kg - lb_cube[2], jg - lb_cube[1], - lb_cube[0]);
            for (int ig = igmin; ig <= 1 - igmin; ig++) {
                const int i = map[0][ig + cmax];  // target location on the grid
                dst[i - 1] += src[ig];
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
    const double dy = dh_inv[1][1];
    const double dx = dh_inv[0][0];
    const int *__restrict map_x = map[0];
    const int *__restrict map_y = map[1];
    const int *__restrict map_z = map[2];

    for (int kg = kgmin; kg <= 1 - kgmin; kg++) {
        const int k = map_z[kg + cmax];   // target location on the grid
        const int kd = (2 * kg - 1) / 2;     // distance from center in grid points
        const double kr = kd * dz;   // distance from center in a.u.
        const double kremain = disr_radius * disr_radius - kr * kr;
        const int jgmin = ceil(-1e-8 - sqrt(max(0.0, kremain)) * dy);
        for (int jg = jgmin; jg <= 1 - jgmin; jg++) {
            const int j = map_y[jg + cmax];  // target location on the grid
            const int jd = (2 * jg - 1) / 2;    // distance from center in grid points
            const double jr = jd * dy;  // distance from center in a.u.
            const double jremain = kremain - jr * jr;
            const int igmin = ceil(-1e-8 - sqrt(max(0.0, jremain)) * dx);
            double *__restrict dst = &idx3(grid[0], k - 1, j - 1, 0);
            const double *__restrict src = &idx3(cube[0], kg - lb_cube[2], jg - lb_cube[1], - lb_cube[0]);
            for (int ig = igmin; ig <= 1 - igmin; ig++) {
                const int i = map_x[ig + cmax];  // target location on the grid
                dst[i - 1] += src[ig];
            }
        }
    }
}

// *****************************************************************************
static void grid_collocate_core(const tensor *coef_xyz, // [lp+1][lp+1][lp+1]
                                const tensor *pol,
                                const int *map[3],
                                const int lb_cube[3],
                                const int ub_cube[3],
                                const double dh[3][3],
                                const double dh_inv[3][3],
                                const double disr_radius,
                                tensor *grid) { // [ngrid[2]][ngrid[1]][ngrid[0]]

    // Create the full cube, ignoring periodicity for now.
    const int nz = ub_cube[2] - lb_cube[2] + 1;
    const int ny = ub_cube[1] - lb_cube[1] + 1;
    const int nx = ub_cube[0] - lb_cube[0] + 1;

    // yes it is reverse order. it is natural to have xyz but we store things in
    // the format zyx. the cube containing the result is stored as C_{z,y,x}

    tensor cube;
    initialize_tensor_3(&cube, nz, ny, nx);

#if defined(LIBXSMM)
    cube.data =  libxsmm_aligned_scratch(sizeof(double) * cube.alloc_size_, 0/*auto-alignment*/);
    char *tmp = NULL;
#else
    cube.data = (double*)scratch;
    char *tmp = scratch + cube.alloc_size_ * sizeof(double);
#endif

    memset(cube.data, 0, cube.alloc_size_ * sizeof(double));

    /* see comments in the function to have a idea of is going on */

    collocate_core_rectangular(NULL, // will need to change that eventually.
                                     // pointer to scratch memory
                               coef_xyz,
                               pol,
                               &cube);

    //
    // Write cube back to large grid taking periodicity and radius into account.
    //

    // The cube contains an even number of grid points in each direction and
    // collocation is always performed on a pair of two opposing grid points.
    // Hence, the points with index 0 and 1 are both assigned distance zero via
    // the formular distance=(2*index-1)/2.
    apply_mapping(disr_radius, dh, dh_inv, map, lb_cube, &cube, (pol->size[2] - 1) / 2, grid);

#if defined(LIBXSMM)
    libxsmm_free(cube.data);
#endif
}

// *****************************************************************************
static void grid_collocate_ortho(const double zetp,
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
    for (int i=0; i<3; i++) {
        double dh_inv_rp = 0.0;
        for (int j=0; j<3; j++) {
            dh_inv_rp += dh_inv[j][i] * rp[j];
        }
        cubecenter[i] = floor(dh_inv_rp);
    }

    double roffset[3];
    for (int i=0; i<3; i++) {
        roffset[i] = rp[i] - ((double) cubecenter[i]) * dh[i][i];
    }

    // Historically, the radius gets discretized.
    const double drmin = min(dh[0][0], min(dh[1][1], dh[2][2]));
    const double disr_radius = drmin * max(1, ceil(radius/drmin));

    int lb_cube[3], ub_cube[3];
    for (int i = 0; i < 3; i++) {
        lb_cube[i] = ceil(-1e-8 - disr_radius * dh_inv[i][i]);
        ub_cube[i] = 1 - lb_cube[i];
    }

    //cmax = MAXVAL(ub_cube)
    int cmax = INT_MIN;
    for (int i = 0; i < 3; i++) {
        cmax = max(cmax, ub_cube[i]);
    }

    // a mapping so that the ig corresponds to the right grid point
#if defined(LIBXSMM)
    int* map[3];
    // normally it is 2 * cmax + 1, but for alignment reason it is probably
    // better to align on 16 bytes

    map[0] =  libxsmm_aligned_scratch(sizeof(int) * 3 * (2 * cmax + 1), 0/*auto-alignment*/);
    map[1] = map[0] + (2 * cmax + 1);
    map[2] = map[1] + (2 * cmax + 1);
#else
    int map[3][2 * cmax + 1];
#endif
    memset(map[0], 0, sizeof(int) * 3 * (2 * cmax + 1));

    for (int i = 0; i < 3; i++) {
        grid_fill_map(periodic[i],
                      lb_cube[i],
                      ub_cube[i],
                      cubecenter[i],
                      lb_grid[i],
                      npts[i],
                      grid->size[3 - i - 1],
                      cmax,
                      map[i]);
    }

    /* double pol[3][lp+1][2*cmax+1]; */
    tensor pol;
    initialize_tensor_3(&pol, 3, coef_xyz->size[0], 2 * cmax + 1);

#if defined(LIBXSMM)
    pol.data =  libxsmm_aligned_scratch(sizeof(double) * pol.alloc_size_, 0/*auto-alignment*/);
#else
    pol.data = (double*)tmp;
    tmp += poly.alloc_size_ * sizeof(double);
#endif

    memset(pol.data, 0, sizeof(double) * pol.alloc_size_);

    // WARNING : do not reverse the order in pol otherwise you will have to
    // reverse the order in collocate_dgemm as well.

    grid_fill_pol(dh[0][0], roffset[0], lb_cube[0], coef_xyz->size[0] - 1, cmax, zetp, &idx3(pol, 2, 0, 0)); /* i indice */
    grid_fill_pol(dh[1][1], roffset[1], lb_cube[1], coef_xyz->size[1] - 1, cmax, zetp, &idx3(pol, 1, 0, 0)); /* j indice */
    grid_fill_pol(dh[2][2], roffset[2], lb_cube[2], coef_xyz->size[2] - 1, cmax, zetp, &idx3(pol, 0, 0, 0)); /* k indice */

    // grid[k][j][i]

    grid_collocate_core(coef_xyz, &pol, map, lb_cube, ub_cube, dh, dh_inv, disr_radius, grid);

#if defined(LIBXSMM)
    libxsmm_free(pol.data);
    libxsmm_free(map[0]);
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
static void grid_collocate_internal(const bool use_ortho,
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
#ifdef LIBXSMM
    alpha.data = libxsmm_aligned_scratch(sizeof(double) * alpha.alloc_size_, 0/*auto-alignment*/);
#else
    alpha.data = (double*)tmp;
    tmp += alpha.alloc_size_ * sizeof(double);
#endif


    const int lp = lmax_prep[0] + lmax_prep[1];
    tensor coef_xyz;
    initialize_tensor_3(&coef_xyz, lp + 1, lp + 1, lp + 1);
#ifdef LIBXSMM
    coef_xyz.data = libxsmm_aligned_scratch(sizeof(double) * coef_xyz.alloc_size_, 0/*auto-alignment*/);
#else
    coef_xyz.data = (double*)tmp;
    tmp += coef_xyz.alloc_size_ * sizeof(double);
#endif

    if (use_ortho) {
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
// initialy cp2k stores coef_xyz as coef[z][y][x]. this is fine but I
        // need them to be stored as

        grid_prepare_alpha(ra,
                           rb,
                           rp,
                           lmax_prep,
                           &alpha);

        grid_prepare_coef(lmax_prep,
                          lmin_prep,
                          lp,
                          prefactor,
                          &alpha,
                          pab_prep,
                          &coef_xyz);

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

#if defined(LIBXSMM)
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
#ifdef LIBXSMM
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
