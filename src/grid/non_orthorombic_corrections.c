#include "utils.h"
#include "non_orthorombic_corrections.h"

double exp_recursive(const double c_exp, const double c_exp_minus_1, const int index)
{

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

void exp_i(const double alpha, const int imin, const int imax, double *__restrict__ const res)
{
    const double c_exp_co = exp(alpha);
    /* const double c_exp_minus_1 = 1/ c_exp; */
    res[0] = exp(imin * alpha);
    for (int i = 1; i < (imax - imin); i++) {
        res[i] = res[i - 1] * c_exp_co;//exp_recursive(c_exp_co, 1.0 / c_exp_co, i + imin);
    }
}

void exp_ij(const double alpha, const int imin, const int imax, const int jmin, const int jmax, tensor *exp_ij_)
{
    double c_exp = exp(alpha * imin);
    const double c_exp_co = exp(alpha);

    for (int i = 0; i < (imax - imin); i++) {
        double *__restrict dst = &idx2(exp_ij_[0], i, 0);
        double ctmp = exp_recursive(c_exp, 1.0 / c_exp, jmin);

#pragma GCC ivdep
        for (int j = 0; j < (jmax - jmin); j++) {
            dst[j] *= ctmp;
            ctmp *= c_exp;
        }
        c_exp *= c_exp_co;
    }
}

void calculate_non_orthorombic_corrections_tensor(const double mu_mean,
                                                  const double *r_ab,
                                                  const double basis[3][3],
                                                  const int *const xmin,
                                                  const int *const xmax,
                                                  bool *plane,
                                                  tensor *const Exp)
{
    // zx, zy, yx
    const int n[3][2] = {{0, 2},
                         {0, 1},
                         {1, 2}};

    // need to review this
    const double c[3] = {
        /* alpha gamma */
        -2.0 * mu_mean * (basis[0][0] * basis[2][0] + basis[0][1] * basis[2][1] + basis[0][2] * basis[2][2]),
        /* beta gamma */
        -2.0 * mu_mean * (basis[1][0] * basis[2][0] + basis[1][1] * basis[2][1] + basis[1][2] * basis[2][2]),
        /* alpha beta */
        -2.0 * mu_mean * (basis[0][0] * basis[1][0] + basis[0][1] * basis[1][1] + basis[0][2] * basis[1][2])};


    /* Check if some vectors are orthogonal */
    // xz
    plane[0] = (fabs(c[0]) < 1e-12);
    // yz
    plane[1] = (fabs(c[1]) < 1e-12);
    // xy
    plane[2] = (fabs(c[2]) < 1e-12);

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
    if (plane[0] && plane[1] && plane[2])
        return;

    tensor exp_tmp;
    double *x1, *x2;

    initialize_tensor_2(&exp_tmp, Exp->size[1], Exp->size[2]);
    const int max_elem = max(max(xmax[0] - xmin[0], xmax[1] - xmin[1]), xmax[2] - xmin[2]) + 1;
    posix_memalign((void **)&x1, 64, sizeof(double) * max_elem);
    posix_memalign((void **)&x2, 64, sizeof(double) * max_elem);

    for (int dir = 0; dir < 3; dir++) {
        int d1 = n[dir][0];
        int d2 = n[dir][1];

        if (fabs(c[dir]) > 1e-12) {
            memset(&idx3(Exp[0], dir, 0, 0), 0, sizeof(double) * Exp->ld_ * (xmax[d1] - xmin[d1] + 1));
            const double c_exp_const = exp(c[dir] * r_ab[d1] * r_ab[d2]);

            exp_i(-r_ab[d2] * c[dir], xmin[d1], xmax[d1] + 1, x1);
            exp_i(-r_ab[d1] * c[dir], xmin[d2], xmax[d2] + 1, x2);

            exp_tmp.data = &idx3(Exp[0], dir, 0, 0);
            cblas_dger(CblasRowMajor,
                       xmax[d1] - xmin[d1] + 1,
                       xmax[d2] - xmin[d2] + 1,
                       c_exp_const,
                       x1, 1,
                       x2, 1,
                       exp_tmp.data, exp_tmp.ld_);
            exp_ij(c[dir], xmin[d1], xmax[d1] + 1, xmin[d2], xmax[d2] + 1, &exp_tmp);
        }// else {
        /*     // it means that the two directions are orthogonal to each other. The exponential is equal to one */
        /*     for(int x = 0; x < (xmax[d1] - xmin[d1] + 1); x++) { */
        /*         for(int y = 0; y < (xmax[d2] - xmin[d2] + 1); y++) { */
        /*             idx3(Exp[0], dir, x, y) = 1.0; */
        /*         } */
        /*     } */
        /* } */
    }
    free(x1);
    free(x2);
}

void apply_non_orthorombic_corrections(const bool *__restrict plane, const tensor *const Exp, tensor *const cube)
{
    // Well we should never call non orthorombic corrections if everything is orthorombic
    if (plane[0] && plane[1] && plane[2])
        return;

/*k and i are orthogonal, k and j as well */
    if (plane[0] && plane[1]) {
        for (int z = 0; z < cube->size[0]; z++) {
            for (int y = 0; y < cube->size[1]; y++) {
                const double *__restrict__ yx = &idx3(Exp[0], 2, y, 0);
                LIBXSMM_PRAGMA_SIMD
                    for (int x = 0; x < cube->size[2]; x++) {
                        idx3(cube[0], z, y, x) *= yx[x];
                    }
            }
        }
        return;
    }

    /* k and i are orhogonal, i and j as well */
    if (plane[0] && plane[2]) {
        for (int z = 0; z < cube->size[0]; z++) {
            for (int y = 0; y < cube->size[1]; y++) {
                const double zy = idx3(Exp[0], 1, z, y);
                LIBXSMM_PRAGMA_SIMD
                    for (int x = 0; x < cube->size[2]; x++) {
                        idx3(cube[0], z, y, x) *= zy;
                    }
            }
        }
        return;
    }

    /* j, k are orthognal, i and j are orthognal */
    if (plane[1] && plane[2]) {
        for (int z = 0; z < cube->size[0]; z++) {
            double *__restrict__ zx = &idx3(Exp[0], 0, z, 0);
            for (int y = 0; y < cube->size[1]; y++) {
                LIBXSMM_PRAGMA_SIMD
                    for (int x = 0; x < cube->size[2]; x++) {
                        idx3(cube[0], z, y, x) *= zx[x];
                    }
            }
        }
        return;
    }

    if (plane[0]) {
        // z perpendicular to x. but y non perpendicular to any
        for (int z = 0; z < cube->size[0]; z++) {
            for (int y = 0; y < cube->size[1]; y++) {
                const double zy = idx3(Exp[0], 1, z, y);
                const double *__restrict__ yx = &idx3(Exp[0], 2, y, 0);
                LIBXSMM_PRAGMA_SIMD
                    for (int x = 0; x < cube->size[2]; x++) {
                        idx3(cube[0], z, y, x) *= zy * yx[x];
                    }
            }
        }
        return;
    }

    if (plane[1]) {
        // z perpendicular to y, but x and z are not and y and x neither
        for (int z = 0; z < cube->size[0]; z++) {
            double *__restrict__ zx = &idx3(Exp[0], 0, z, 0);
            for (int y = 0; y < cube->size[1]; y++) {
                const double *__restrict__ yx = &idx3(Exp[0], 2, y, 0);
                LIBXSMM_PRAGMA_SIMD
                    for (int x = 0; x < cube->size[2]; x++) {
                        idx3(cube[0], z, y, x) *= zx[x] * yx[x];
                    }
            }
        }
        return;
    }


    if (plane[2]) {
// x perpendicular to y, but x and z are not and y and z neither
        for (int z = 0; z < cube->size[0]; z++) {
            double *__restrict__ zx = &idx3(Exp[0], 0, z, 0);
            for (int y = 0; y < cube->size[1]; y++) {
                const double zy = idx3(Exp[0], 1, z, y);
                LIBXSMM_PRAGMA_SIMD
                    for (int x = 0; x < cube->size[2]; x++) {
                        idx3(cube[0], z, y, x) *= zx[x] * zy;
                    }
            }
        }
        return;
    }

/* generic  case */

    for (int z = 0; z < cube->size[0]; z++) {
        double *__restrict__ zx = &idx3(Exp[0], 0, z, 0);
        for (int y = 0; y < cube->size[1]; y++) {
            const double zy = idx3(Exp[0], 1, z, y);
            const double *__restrict__ yx = &idx3(Exp[0], 2, y, 0);
            LIBXSMM_PRAGMA_SIMD
                for (int x = 0; x < cube->size[2]; x++) {
                    idx3(cube[0], z, y, x) *= zx[x] * zy * yx[x];
                }
        }
    }
    return;
}
