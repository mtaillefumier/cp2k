#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__LIBXSMM)
#include <libxsmm.h>
#endif

#include "grid_common.h"
#include "utils.h"
#include "tensor_local.h"
#include "coefficients.h"

// binomial coefficients n = 0 ... 20
const static int binomial[21][21] = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 6, 15, 20, 15, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 7, 21, 35, 35, 21, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 8, 28, 56, 70, 56, 28, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 9, 36,  84, 126, 126, 84, 36, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0},
                                     {1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, 78, 13, 1, 0, 0, 0, 0, 0, 0, 0},
                                     {1, 14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1, 0, 0, 0, 0, 0, 0},
                                     {1, 15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455, 105, 15, 1, 0, 0, 0, 0, 0},
                                     {1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1, 0, 0, 0, 0},
                                     {1, 17, 136, 680, 2380, 6188, 12376, 19448, 24310, 24310, 19448, 12376, 6188, 2380, 680, 136, 17, 1, 0, 0, 0},
                                     {1, 18, 153, 816, 3060, 8568, 18564, 31824, 43758, 48620, 43758, 31824, 18564, 8568, 3060, 816, 153, 18, 1, 0, 0},
                                     {1, 19, 171, 969, 3876, 11628, 27132, 50388, 75582, 92378, 92378, 75582, 50388,  27132, 11628, 3876, 969, 171, 19, 1, 0},
                                     {1, 20, 190, 1140, 4845, 15504, 38760, 77520, 125970, 167960, 184756, 167960, 125970, 77520, 38760, 15504, 4845, 1140, 190, 20, 1}};


extern void collocate_core_rectangular(double *scratch,
                                       const double prefactor,
                                       const struct tensor_ *co,
                                       const struct tensor_ *p_alpha_beta_reduced_,
                                       struct tensor_ *cube);

// *****************************************************************************
void grid_prepare_coef(const bool ortho,
                       const int *lmax,
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
            if (ortho) {
                for (int lzp = 0; lzp<=lza+lzb; lzp++) {
                    for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
                        for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
                            idx3(coef_xyz[0], lxp, lzp, lyp) += idx4(alpha[0], 2, lzb, lza, lzp) * coef_xyt[lyp][lxp];
                        }
                    }
                }
            } else {
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
}

// *****************************************************************************
void grid_prepare_alpha(const double ra[3],
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



/* this function will replace the function grid_prepare_coef when I get the
 * tensor coef right. This need change elsewhere in cp2k, something I do not
 * want to do right now. I need to rethink the order of the parameters */

void compute_compact_polynomial_coefficients(const tensor *coef,
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

    collocate_core_rectangular(NULL, prefactor, &coef_tmp, &px, co);
}

inline double return_multimonial_prefactor(const int l, const int m, int *const alpha, int *const gamma, int *const beta, const tensor *const power, const int dir)
{
    const int expo1 = return_exponents(return_offset_l(l) + m);
    *alpha = (expo1 & 0xff0000) >> 16;
    *gamma = (expo1 & 0xff);
    *beta = (expo1 & 0xff00) >> 8;
    return multinomial3(*alpha, *gamma, *beta) *
        idx3(power[0], *alpha, dir, 0) *
        idx3(power[0], *gamma, dir, 1) *
        idx3(power[0], *beta, dir, 2);
}

/* // ***************************************************************************** */
void grid_prepare_coef_generic(const int *lmax,
                               const int *lmin,
                               const int lp,
                               const double prefactor,
                               const double dh[3][3],
                               const tensor *coef_xyz,
                               const tensor *coef_non_ortho) //[lp+1][lp+1][lp+1]
{

    tensor power;


    initialize_tensor_3(&power, coef_xyz->size[0], 3, 3);

    posix_memalign(&power.data, 32, power.alloc_size_ * sizeof(double));

    idx3(power, 0, 0, 0) = 1.0;
    idx3(power, 0, 0, 1) = 1.0;
    idx3(power, 0, 0, 2) = 1.0;
    idx3(power, 0, 1, 0) = 1.0;
    idx3(power, 0, 1, 1) = 1.0;
    idx3(power, 0, 1, 2) = 1.0;
    idx3(power, 0, 2, 0) = 1.0;
    idx3(power, 0, 2, 1) = 1.0;
    idx3(power, 0, 2, 2) = 1.0;

    idx3(power, 1, 0, 0) = dh[2][0];
    idx3(power, 1, 0, 1) = dh[2][1];
    idx3(power, 1, 0, 2) = dh[2][2];
    idx3(power, 1, 1, 0) = dh[1][0];
    idx3(power, 1, 1, 1) = dh[1][1];
    idx3(power, 1, 1, 2) = dh[1][2];
    idx3(power, 1, 2, 0) = dh[0][0];
    idx3(power, 1, 2, 1) = dh[0][1];
    idx3(power, 1, 2, 2) = dh[0][2];

    for (int l = 2; l < coef_xyz->size[0]; l++) {
        idx3(power, l, 0, 0) = dh[2][0] * idx3(power, l - 1, 0, 0);
        idx3(power, l, 0, 1) = dh[2][1] * idx3(power, l - 1, 0, 1);
        idx3(power, l, 0, 2) = dh[2][2] * idx3(power, l - 1, 0, 2);
        idx3(power, l, 1, 0) = dh[1][0] * idx3(power, l - 1, 1, 0);
        idx3(power, l, 1, 1) = dh[1][1] * idx3(power, l - 1, 1, 1);
        idx3(power, l, 1, 2) = dh[1][2] * idx3(power, l - 1, 1, 2);
        idx3(power, l, 2, 0) = dh[0][0] * idx3(power, l - 1, 2, 0);
        idx3(power, l, 2, 1) = dh[0][1] * idx3(power, l - 1, 2, 1);
        idx3(power, l, 2, 2) = dh[0][2] * idx3(power, l - 1, 2, 2);
    }

    for (int a1 = 0; a1 < coef_xyz->size[0]; a1++) {
        for (int l1 = 0; l1 < return_length_l(a1); l1++) {
            int alpha_part1, gamma_part1, beta_part1;
            const double multinomial1 = return_multimonial_prefactor(a1, l1, &alpha_part1, &gamma_part1, &beta_part1, &power, 0);
            for (int g1 = 0; g1 < coef_xyz->size[1]; g1++) {
                for (int l2 = 0; l2 < return_length_l(g1); l2++) {
                    int alpha_part2, gamma_part2, beta_part2;
                    const double multinomial2 = return_multimonial_prefactor(g1, l2, &alpha_part2, &gamma_part2, &beta_part2, &power, 2);
                    for (int b1 = 0; b1 < coef_xyz->size[2]; b1++) {
                        for (int l3 = 0; l3 < return_length_l(b1); l3++) {
                            int alpha_part3, gamma_part3, beta_part3;
                            const double multinomial3 = return_multimonial_prefactor(b1, l3, &alpha_part3, &gamma_part3, &beta_part3, &power, 1);
                            idx3(coef_non_ortho[0], a1, g1, b1) += multinomial3 *
                                multinomial2 *
                                multinomial1 *
                                idx3(coef_xyz[0],
                                     alpha_part1 + alpha_part2 + alpha_part3,
                                     gamma_part1 + gamma_part2 + gamma_part3,
                                     beta_part1 + beta_part2 + beta_part3);
                        }
                    }
                }
            }
        }
    }
    free(power.data);
}


void compute_two_gaussian_coefficients(const tensor *const coef,
                                       const int *const lmin, const int *const lmax,
                                       const double *rab, const double *ra, const double *rb,
                                       tensor *const co)
{
    tensor power;

    initialize_tensor_4(&power, 2, 3, lmax[0] + lmax[1] + 1, lmax[0] + lmax[1] + 1);
    posix_memalign(power.data, 32, sizeof(double) * power.alloc_size_);

    for (int dir = 0; dir < 3; dir++) {
        double tmp = rab[dir] - ra[dir];
        idx4(power, 0, dir, 0, 0) = 1.0;
        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            idx4(power, 0, dir, 0, k) = tmp * idx4(power, 0, dir, 0, k - 1);

        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            memcpy(&idx4(power, 0, dir, k ,0),
                   &idx4(power, 0, dir, k - 1, 0),
                   sizeof(double) * power.size[3]);

        tmp = rab[dir] - rb[dir];

        idx4(power, 1, dir, 0, 0) = 1.0;
        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            idx4(power, 1, dir, 0, k) = tmp * idx4(power, 1, dir, 0, k - 1);

        for (int k = 1; k < lmax[0] + lmax[1] + 1; k++)
            memcpy(&idx4(power, 1, dir, k ,0),
                   &idx4(power,1, dir, k - 1, 0),
                   sizeof(double) * power.size[3]);

        for (int a1 = 0; a1 < lmax[0] + lmax[1] + 1; a1++)
            for (int k = 0; k < lmax[0] + lmax[1] + 1; k++)
                idx4(power, 0, dir, a1, k) *= binomial[a1][k];

        for (int a1 = 0; a1 < lmax[0] + lmax[1] + 1; a1++)
            for (int k = 0; k < lmax[0] + lmax[1] + 1; k++)
                idx4(power, 1, dir, a1, k) *= binomial[a1][k];

    }

    /* We can also do that dgemm actually. Use of collocate seems possible */


    for (int l1 = lmin[0]; l1 <= lmax[0]; l1++) {
        for (int l2 = lmin[1]; l2 <= lmax[1]; l2++) {
            for (int alpha1 = 0; alpha1 <= l1; alpha1++) {
                for (int alpha2 = 0; alpha2 <= l2; alpha2++) {
                    for (int beta1 = 0; beta1 <= (l1 - alpha1); beta1++) {
                        const int gamma1 = l1 - alpha1 - beta1;
                        for (int beta2 = 0; beta2 <= (l2 - alpha2); beta2++) {
                            const int gamma2 = l2 - alpha2 - beta2;
                            double tmp = 0.0;
                            for (int k1 = 0; k1 <= alpha1; k1++) {
                                /*
                                  WARNING : We may have to permute the indices
                                */

                                const double c1 = idx4(power, 2, 0, alpha1, k1);
                                for (int k2 = 0; k2 <= alpha2; k2++) {
                                    const double c2 = c1 * idx4(power, 2, 1, alpha2, k2);
                                    for (int k3 = 0; k3 <= beta1; k3++) {
                                        const double c3 = c2 * idx4(power, 1, 0, beta1, k3);
                                        for (int k4 = 0; k4 <= beta2; k4++) {
                                            const double c4 = c3 * idx4(power, 1, 1, beta2, k4);
                                            for (int k5 = 0; k5 <= gamma1; k5++) {
                                                const double c5 = c4 * idx4(power, 0, 0, gamma1, k5);
                                                const double *__restrict__ src = &idx3(coef[0], k3 + k4, k1 + k2, k5);
                                                for (int k6 = 0; k6 <= gamma2; k6++) {
                                                    tmp += c5 * src[k6] * idx4(power, 0, 1, gamma2, k6);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            idx2(co[0], return_linear_index_from_exponents(gamma1, beta1, alpha1),
                                 return_linear_index_from_exponents(gamma2, beta2, alpha2)) = tmp;
                        }
                    }
                }
            }
        }
    }
}
