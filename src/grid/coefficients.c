#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <assert.h>
#if defined(__LIBXSMM)
#include <libxsmm.h>
#endif

#include "grid_common.h"
#include "utils.h"
#include "tensor_local.h"
#include "coefficients.h"

void transform_xyz_to_triangular(const tensor *const coef, double  *const  coef_xyz)
{
    assert(coef != NULL);
    assert(coef_xyz != NULL);

    int lxyz = 0;
    const int lp = (coef->size[0] - 1);
    for (int lzp = 0; lzp <= lp; lzp++) {
        for (int lyp = 0; lyp <= lp - lzp; lyp++) {
            for (int lxp = 0; lxp <= lp - lzp - lyp; lxp++, lxyz++) {
                coef_xyz[lxyz] = idx3(coef[0], lzp, lyp, lxp);
            }
        }
    }
}


void transform_yxz_to_triangular(const tensor *const coef, double  *const coef_xyz)
{
    assert(coef != NULL);
    assert(coef_xyz != NULL);
    int lxyz = 0;
    const int lp = (coef->size[0] - 1);
    for (int lzp = 0; lzp <= lp; lzp++) {
        for (int lyp = 0; lyp <= lp - lzp; lyp++) {
            for (int lxp = 0; lxp <= lp - lzp - lyp; lxp++, lxyz++) {
                coef_xyz[lxyz] = idx3(coef[0], lyp, lxp, lzp);
            }
        }
    }
}


void transform_triangular_to_xyz(const double  *const coef_xyz, tensor *const coef)
{
    assert(coef != NULL);
    assert(coef_xyz != NULL);
    int lxyz = 0;
    const int lp = coef->size[0] - 1;
    for (int lzp = 0; lzp <= lp; lzp++) {
        for (int lyp = 0; lyp <= lp - lzp; lyp++) {
            for (int lxp = 0; lxp <= lp - lzp - lyp; lxp++, lxyz++) {
                idx3(coef[0], lzp, lyp, lxp) = coef_xyz[lxyz];
            }
            /* initialize the remaining coefficients to zero */
            for (int lxp = lp - lzp - lyp + 1; lxp <= lp; lxp++)
                idx3(coef[0], lzp, lyp, lxp) = 0.0;
        }
    }
}


// *****************************************************************************
void grid_prepare_coef(const int *lmin,
                       const int *lmax,
                       const int lp,
                       const double prefactor,
                       const tensor *alpha, // [3][lb_max+1][la_max+1][lp+1]
                       const double pab[ncoset[lmax[1]]][ncoset[lmax[0]]],
                       tensor *coef_xyz) //[lp+1][lp+1][lp+1]
{
    assert(alpha != NULL);
    assert(coef_xyz != NULL);
    assert(coef_xyz->data != NULL);
    memset(coef_xyz->data, 0, coef_xyz->alloc_size_ * sizeof(double));
    // we need a proper fix for that. We can use the tensor structure for this

    double coef_xyt[lp + 1][lp + 1];
    double coef_xtt[lp + 1];

    for (int lzb = 0; lzb<=lmax[1]; lzb++) {
        for (int lza = 0; lza<=lmax[0]; lza++) {
            memset(coef_xyt, 0, sizeof(double) * (lp + 1) * (lp + 1));
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
            /* I need to permute two fo the indices for the orthogonal case */
            for (int lzp = 0; lzp<=lza+lzb; lzp++) {
                for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
                    for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
                        idx3(coef_xyz[0], lxp, lzp, lyp) += idx4(alpha[0], 2, lzb, lza, lzp) * coef_xyt[lyp][lxp];
                    }
                }
            }
        }
    }
}

// *****************************************************************************
// for gpu, it is better to store only the relevant matrix elements instead of
// the full matrix. This reduces transfer between CPU and GPU

void grid_prepare_coef_gpu(const int *lmin,
                           const int *lmax,
                           const int lp,
                           const double prefactor,
                           const tensor *alpha, // [3][lb_max+1][la_max+1][lp+1]
                           const double pab[ncoset[lmax[1]]][ncoset[lmax[0]]],
                           double *coef_xyz) //[lp+1][lp+1][lp+1]
{
    assert(alpha != NULL);
    assert(coef_xyz != NULL);
    // we need a proper fix for that. We can use the tensor structure for this

    double coef_xyt[lp+1][lp+1];
    double coef_xtt[lp+1];

    for (int lzb = 0; lzb<=lmax[1]; lzb++) {
        for (int lza = 0; lza<=lmax[0]; lza++) {
            memset(coef_xyt, 0, sizeof(double) * (lp + 1)* (lp + 1));
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
            /* I need to permute two fo the indices for the orthogonal case */
            for (int lzp = 0; lzp<=lza+lzb; lzp++) {
                for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
                    for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
                        coef_xyz[coset_without_offset(lxp, lzp, lyp)] += idx4(alpha[0], 2, lzb, lza, lzp) * coef_xyt[lyp][lxp];
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
    assert(alpha != NULL);
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


/* this function computes the coefficients initially expressed in the cartesian
 * space to the grid space. It is inplane and can also be done with
 * matrix-matrix multiplication. It is in fact a tensor reduction. */

void grid_transform_coef_xzy_to_ikj(const double dh[3][3],
                                    const tensor *coef_xyz)
{
    assert(coef_xyz != NULL);
    const int lp = coef_xyz->size[0] - 1;
    tensor coef_ijk;

    /* this tensor corresponds to the term
     * $v_{11}^{k_{11}}v_{12}^{k_{12}}v_{13}^{k_{13}}
     * v_{21}^{k_{21}}v_{22}^{k_{22}}v_{23}^{k_{23}}
     * v_{31}^{k_{31}}v_{32}^{k_{32}} v_{33}^{k_{33}}$ in Eq.26 found section
     * III.A of the notes */
    tensor hmatgridp;

    initialize_tensor_3(&coef_ijk, coef_xyz->size[0], coef_xyz->size[1], coef_xyz->size[2]);

    coef_ijk.data = memalign(64, sizeof(double) * coef_ijk.alloc_size_);

    if(coef_ijk.data == NULL)
        abort();

    memset(coef_ijk.data, 0, sizeof(double) * coef_ijk.alloc_size_);
    initialize_tensor_3(&hmatgridp, coef_xyz->size[0], 3, 3);

    hmatgridp.data = memalign(64, sizeof(double) * hmatgridp.alloc_size_);

    // transform using multinomials
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            idx3(hmatgridp, 0, j, i) = 1.0;
            for (int k = 1; k <= lp; k++) {
                idx3(hmatgridp, k, j, i) = idx3(hmatgridp, k - 1, j, i) * dh[j][i];
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
                                        //const int lijk= coef_map[kl][jl][il];
                                        /* the fac table is the factorial. It
                                         * would be better to use the
                                         * multinomials. */
                                        idx3(coef_ijk, il, kl, jl) += idx3(coef_xyz[0], lx, lz, ly) *
                                            idx3(hmatgridp, ilx, 0, 0) * idx3(hmatgridp, jlx, 1, 0) * idx3(hmatgridp, klx, 2, 0) *
                                            idx3(hmatgridp, ily, 0, 1) * idx3(hmatgridp, jly, 1, 1) * idx3(hmatgridp, kly, 2, 1) *
                                            idx3(hmatgridp, ilz, 0, 2) * idx3(hmatgridp, jlz, 1, 2) * idx3(hmatgridp, klz, 2, 2) *
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

    memcpy(coef_xyz->data, coef_ijk.data, sizeof(double) * coef_ijk.alloc_size_);
    free(coef_ijk.data);
    free(hmatgridp.data);
}

/* Rotate the coefficients computed in the local grid coordinates to the
 * cartesians coorinates. The order of the indices indicates how the
 * coefficients are stored */
void grid_transform_coef_jik_to_yxz(const double dh[3][3],
                                    const tensor *coef_xyz)
{
    assert(coef_xyz);
    const int lp = coef_xyz->size[0] - 1;
    tensor coef_ijk;

    /* this tensor corresponds to the term
     * $v_{11}^{k_{11}}v_{12}^{k_{12}}v_{13}^{k_{13}}
     * v_{21}^{k_{21}}v_{22}^{k_{22}}v_{23}^{k_{23}}
     * v_{31}^{k_{31}}v_{32}^{k_{32}} v_{33}^{k_{33}}$ in Eq.26 found section
     * III.A of the notes */
    tensor hmatgridp;

    initialize_tensor_3(&coef_ijk, coef_xyz->size[0], coef_xyz->size[1], coef_xyz->size[2]);

    coef_ijk.data = memalign(64, sizeof(double) * coef_ijk.alloc_size_);
    if(coef_ijk.data == NULL)
        abort();

    memset(coef_ijk.data, 0, sizeof(double) * coef_ijk.alloc_size_);
    initialize_tensor_3(&hmatgridp, coef_xyz->size[0], 3, 3);

    hmatgridp.data = memalign(64, sizeof(double) * hmatgridp.alloc_size_);

    // transform using multinomials
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            idx3(hmatgridp, 0, j, i) = 1.0;
            for (int k = 1; k <= lp; k++) {
                idx3(hmatgridp, k, j, i) = idx3(hmatgridp, k - 1, j, i) * dh[j][i];
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
                                        //const int lijk= coef_map[kl][jl][il];
                                        /* the fac table is the factorial. It
                                         * would be better to use the
                                         * multinomials. */
                                        idx3(coef_ijk, ly, lx, lz) += idx3(coef_xyz[0], jl, il, kl) *
                                            idx3(hmatgridp, ilx, 0, 0) * idx3(hmatgridp, jlx, 1, 0) * idx3(hmatgridp, klx, 2, 0) *
                                            idx3(hmatgridp, ily, 0, 1) * idx3(hmatgridp, jly, 1, 1) * idx3(hmatgridp, kly, 2, 1) *
                                            idx3(hmatgridp, ilz, 0, 2) * idx3(hmatgridp, jlz, 1, 2) * idx3(hmatgridp, klz, 2, 2) *
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
    memcpy(coef_xyz->data, coef_ijk.data, sizeof(double) * coef_ijk.alloc_size_);
    free(coef_ijk.data);
    free(hmatgridp.data);
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
    assert(coef != NULL);
    assert(co != NULL);
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
    power.data = memalign(64, sizeof(double) * power.alloc_size_);
    px.data = memalign(64, sizeof(double) * px.alloc_size_);
    coef_tmp.data = memalign(64, sizeof(double) * coef_tmp.alloc_size_);
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
                                    const int i2 = coef_offset_[1] + return_linear_index_from_exponents(a2, b2, g2);
                                    idx3(coef_tmp, i_b, i_a, i_g) += src[i2];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

#if defined(__LIBXSMM)
    libxsmm_free(power.data);
    libxsmm_free(px.data);
    libxsmm_free(coef_tmp.data);
#else
    free(power.data);
    free(px.data);
    free(coef_tmp.data);
#endif
}
