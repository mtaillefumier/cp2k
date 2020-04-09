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
#include <stdlib.h>
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
#include "collocation_integration.h"
#include "utils.h"
#include "coefficients.h"

extern void apply_mapping(const double disr_radius,
                          const double dh[3][3],
                          const double dh_inv[3][3],
                          const int *map[3],
                          const int lb_cube[3],
                          tensor *cube,
                          const int cmax,
                          tensor *grid);

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

void collocate_l0(const struct tensor_ *co,
                  const struct tensor_ *p_alpha_beta_reduced_,
                  struct tensor_ *cube);

void collocate_core_rectangular(double *scratch,
                                const double prefactor,
                                const struct tensor_ *co,
                                const struct tensor_ *p_alpha_beta_reduced_,
                                struct tensor_ *cube);


void *print_collocation_list(void *const handler);
collocation_list *create_collocation_list(const int num_elem, const int integration);
void destroy_collocation_list(struct collocation_list_ *list_collocation);
void print_collocation_block(const struct collocation_block_ *const list);
void add_collocation_block(struct collocation_integration_ *const handle,
                           const int *period,
                           const int *cube_size,
                           const int *position,
                           const int *lower_corner,
                           const int *upper_corner,
                           const tensor *Exp,
                           tensor *grid);
void release_collocation_block_memory(struct collocation_list_ *const list_collocation);
void calculate_collocation(void *const list_);


void collocate_l0(const struct tensor_ *co,
                  const struct tensor_ *p_alpha_beta_reduced_,
                  struct tensor_ *cube)
{
    const double *__restrict pz = &idx3(p_alpha_beta_reduced_[0], 0, 0, 0); /* k indice */
    const double *__restrict py = &idx3(p_alpha_beta_reduced_[0], 1, 0, 0); /* j indice */
    const double *__restrict px = &idx3(p_alpha_beta_reduced_[0], 2, 0, 0); /* i indice */
    const double coo = idx3 (co[0], 0, 0, 0);

    memset(&idx3(cube[0], 0, 0, 0), 0, sizeof(double) * cube->alloc_size_);

    cblas_dger (CblasRowMajor,
                cube->size[1],
                cube->size[2],
                idx3(co[0], 0, 0, 0),
                py,
                1,
                px,
                1,
                &idx3(cube[0], 0, 0, 0),
                cube->ld_);

    for (int z1 = 1; z1 < cube->size[0]; z1++) {
        const double tz = pz[z1];
        cblas_daxpy(cube->size[1] * cube->ld_,
                    pz[z1],
                    &idx3(cube[0], 0, 0, 0),
                    1,
                    &idx3(cube[0], z1, 0, 0),
                    1);
    }

    const double tz = pz[0];
    cblas_dscal(cube->size[1] * cube->ld_,
                tz,
                &idx3(cube[0], 0, 0, 0),
                1);
}

/* compute the functions (x - x_i)^l exp (-eta (x - x_i)^2) for l = 0..lp using
 * a recursive relation to avoid computing the exponential on each grid point. I
 * think it is not really necessary anymore since it is *not* the dominating
 * contribution to computation of collocate and integrate */


void grid_fill_pol(const bool transpose,
                   const double dr,
                   const double roffset,
                   const int lb_cube,
                   const int lp,
                   const int cmax,
                   const double zetp,
                   double *pol_)
{
    tensor pol;
    const double t_exp_1 = exp(-zetp * dr * dr);
    const double t_exp_2 = t_exp_1 * t_exp_1;

    double t_exp_min_1 = exp(-zetp * (dr - roffset) * (dr - roffset));
    double t_exp_min_2 = exp(2.0 * zetp * (dr - roffset) * dr);

    double t_exp_plus_1 = exp(-zetp * roffset * roffset);
    double t_exp_plus_2 = exp(2.0 * zetp * roffset * dr);

    if (transpose) {
        initialize_tensor_2(&pol, 2 * cmax + 1, lp + 1);
        pol.data = pol_;

        /* It is original Ole code. I need to transpose the polynomials for the
         * integration routine and Ole code already does it. */
        for (int ig=0; ig >= lb_cube; ig--) {
            const double rpg = ig * dr - roffset;
            t_exp_min_1 *= t_exp_min_2 * t_exp_1;
            t_exp_min_2 *= t_exp_2;
            double pg = t_exp_min_1;
            // pg  = EXP(-zetp*rpg**2)
            for (int icoef=0; icoef<=lp; icoef++) {
                idx2(pol, ig - lb_cube, icoef) = pg;
                pg *= rpg;
            }
        }

        double t_exp_plus_1 = exp(-zetp * roffset * roffset);
        double t_exp_plus_2 = exp(2 * zetp * roffset * dr);
        for (int ig=0; ig >= lb_cube; ig--) {
            const double rpg = (1 - ig) * dr - roffset;
            t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
            t_exp_plus_2 *= t_exp_2;
            double pg = t_exp_plus_1;
            // pg  = EXP(-zetp*rpg**2)
            for (int icoef = 0; icoef <= lp; icoef++) {
                idx2(pol, 1 - ig - lb_cube, icoef) = pg;
                pg *= rpg;
            }
        }

    } else {
        initialize_tensor_2(&pol, lp + 1, 2 * cmax + 1);
        pol.data = pol_;
        /*
         *   compute the values of all (x-xp)**lp*exp(..)
         *
         *  still requires the old trick:
         *  new trick to avoid to many exps (reuse the result from the previous gridpoint):
         *  exp( -a*(x+d)**2)=exp(-a*x**2)*exp(-2*a*x*d)*exp(-a*d**2)
         *  exp(-2*a*(x+d)*d)=exp(-2*a*x*d)*exp(-2*a*d**2)
         */

        /* compute the exponential recursively and store the polynomial prefactors as well */
        for (int ig = 0; ig >= lb_cube; ig--) {
            const double rpg = ig * dr - roffset;
            t_exp_min_1 *= t_exp_min_2 * t_exp_1;
            t_exp_min_2 *= t_exp_2;
            double pg = t_exp_min_1;
            // pg  = EXP(-zetp*rpg**2)
            /* for (int icoef=0; icoef <= lp; icoef++) { */
            idx2(pol, 0, ig - lb_cube) = pg;
            if (lp > 0)
                idx2(pol, 1, ig - lb_cube) = rpg;
            /* pg *= rpg; */
            /* } */
        }

        for (int ig = 0; ig >= lb_cube; ig--) {
            const double rpg = (1 - ig) * dr - roffset;
            t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
            t_exp_plus_2 *= t_exp_2;
            double pg = t_exp_plus_1;
            // pg  = EXP(-zetp*rpg**2)
            /* for (int icoef = 0; icoef <= lp; icoef++) { */
            idx2(pol, 0, 1 - ig - lb_cube) = pg;
            if (lp > 0)
                idx2(pol, 1, 1 - ig - lb_cube) = rpg;
            /* pg *= rpg; */
        /* } */
        }

        /* compute the remaining powers using previously computed stuff */
        if (lp >= 2) {
            double *__restrict__ poly = &idx2(pol, 1, 0);
            double *__restrict__ src1 = &idx2(pol, 0, 0);
            double *__restrict__ dst = &idx2(pol, 2, 0);
//#pragma omp simd
#pragma GCC ivdep
            for (int ig = 0; ig <= 1 - 2 * lb_cube; ig++)
                dst[ig] = src1[ig] * poly[ig] * poly[ig];
        }

        for (int icoef = 3; icoef <= lp; icoef++) {
            const double *__restrict__ poly = &idx2(pol, 1, 0);
            const double *__restrict__ src1 = &idx2(pol, icoef - 1, 0);
            double *__restrict__ dst = &idx2(pol, icoef, 0);
//#pragma omp simd
#pragma GCC ivdep
            for (int ig = 0; ig <= 1 - 2 * lb_cube; ig++) {
                dst[ig] = src1[ig] * poly[ig];
            }
        }

        //
        if (lp > 0) {
            double *__restrict__ dst = &idx2(pol, 1, 0);
            const double *__restrict__ src = &idx2(pol, 0, 0);
#pragma GCC ivdep
            for (int ig = 0; ig <= 1 - 2 * lb_cube; ig++) {
                dst[ig] *= src[ig];
            }
        }
    }
}


/* compute the following operation (variant)

   V_{kji} = \sum_{\alpha\beta\gamma} C_{\alpha\gamma\beta} T_{0,\alpha,i} T_{1,\beta,j} T_{2,\gamma,k}

*/
void collocate_core_rectangular(double *scratch,
                                const double prefactor,
                                const struct tensor_ *co,
                                const struct tensor_ *p_alpha_beta_reduced_,
                                struct tensor_ *cube)
{
    if (co->size[0] > 1) {
        struct {
#if defined(__LIBXSMM)
            libxsmm_dmmfunction kernel;
            int prefetch;
            int flags;
#endif
            double alpha;
            double beta;
            double *a, *b, *c;
            int m, n, k, lda, ldb, ldc;
        } m1, m2, m3;

        tensor T;
        tensor W;

        double *__restrict const pz = &idx3(p_alpha_beta_reduced_[0], 0, 0, 0); /* k indice */
        double *__restrict const py = &idx3(p_alpha_beta_reduced_[0], 1, 0, 0); /* j indice */
        double *__restrict const px = &idx3(p_alpha_beta_reduced_[0], 2, 0, 0); /* i indice */

        initialize_tensor_3(&T, co->size[0] /* alpha */, co->size[1] /* gamma */, cube->size[1] /* j */);
        initialize_tensor_3(&W, co->size[1] /* gamma */ , cube->size[1] /* j */, cube->size[2] /* i */);

        T.data = scratch;
        W.data = scratch + T.alloc_size_;

        /* WARNING we are in row major layout. cblas allows it and it is more
         * natural to read left to right than top to bottom
         *
         * we do first T_{\alpha,\gamma,j} = \sum_beta C_{alpha\gamma\beta} Y_{\beta, j}
         *
         * keep in mind that Y_{\beta, j} = p_alpha_beta_reduced(1, \beta, j)
         * and the order of indices is also important. the last indice is the
         * fastest one. it can be done with one dgemm.
         */

        m1.alpha = prefactor;
        m1.beta = 0.0;
        m1.m = co->size[0] * co->size[1]; /* alpha gamma */
        m1.n = cube->size[1]; /* j */
        m1.k = co->size[2]; /* beta */
        m1.a = co->data; // Coef_{alpha,gamma,beta} Coef_xzy
        m1.lda = co->ld_;
        m1.b = py; // Y_{beta, j} = p_alpha_beta_reduced(1, beta, j)
        m1.ldb = p_alpha_beta_reduced_->ld_;
        m1.c = T.data; // T_{\alpha, \gamma, j} = T(alpha, gamma, j)
        m1.ldc = T.ld_;

        /*
         * the next step is a reduction along the alpha index.
         *
         * We compute then
         *
         * W_{alpha, k, j} = sum_{\gamma} T_{\gamma, j, alpha} X_{\alpha, i}
         *
         * which means we need to transpose T_{\alpha, \gamma, j} to get
         * T_{\gamma, j, \alpha}. Fortunately we can do it while doing the
         * matrix - matrix multiplication
         */

        m2.alpha = 1.0;
        m2.beta = 0.0;
        m2.m = cube->size[1] * co->size[1]; // \gamma j direction
        m2.n = cube->size[2]; // i
        m2.k = co->size[2]; // alpha
        m2.a = T.data; // T_{\alpha, \gamma, j}
        m2.lda = T.ld_ * co->size[2];
        m2.b = px; // X_{alpha, i}  = p_alpha_beta_reduced(0, alpha, i)
        m2.ldb = p_alpha_beta_reduced_->ld_;
        m2.c = W.data; // W_{\gamma, j, i}
        m2.ldc = W.ld_;

        /* the final step is again a reduction along the alpha indice. It can
         * again be done with one dgemm. The operation is simply
         *
         * Cube_{k, j, i} = \sum_{alpha} Z_{k, \gamma} W_{\gamma, j, i}
         *
         * which means we need to permute W_{\alpha, k, j} in to W_{k, j,
         * \alpha} which can be done with one transposition if we consider (k,j)
         * as a composite index.
         */

        m3.alpha = 1.0;
        m3.beta = 0.0;
        m3.m = cube->size[0]; // Z_{k \gamma}
        m3.n = cube->size[1] * cube->size[2]; // (ji) direction
        m3.k = co->size[2]; // \gamma
        m3.a = pz; // p_alpha_beta_reduced(0, gamma, i)
        m3.lda = p_alpha_beta_reduced_->ld_;
        m3.b = &idx3(W, 0, 0, 0); // W_{\gamma, j, i}
        m3.ldb = W.size[1] * W.ld_;
        m3.c = &idx3(cube[0], 0, 0, 0); // cube_{kji}
        m3.ldc = cube->ld_ * cube->size[1];

#if defined(__LIBXSMM)
        m1.prefetch = LIBXSMM_PREFETCH_AUTO;
        m1.flags = LIBXSMM_GEMM_FLAG_NONE;
        m1.kernel = libxsmm_dmmdispatch(m1.n,
                                        m1.m,
                                        m1.k,
                                        &m1.ldb,
                                        &m1.lda,
                                        &m1.ldc,
                                        &m1.alpha,
                                        &m1.beta,
                                        &m1.flags,
                                        &m1.prefetch);

        m1.kernel(m1.b, m1.a, m1.c);

        /* libxsmm_dgemm("N", */
        /*               "N", */
        /*               &m1.n, */
        /*               &m1.m, */
        /*               &m1.k, */
        /*               &m1.alpha, */
        /*               m1.b, */
        /*               &m1.ldb, */
        /*               m1.a, */
        /*               &m1.lda, */
        /*               &m1.beta, */
        /*               m1.c, // tmp_{alpha, gamma, j} */
        /*               &m1.ldc); */

        m2.prefetch = LIBXSMM_PREFETCH_AUTO;
        m2.flags = LIBXSMM_GEMM_FLAG_TRANS_B;
        m2.kernel = libxsmm_dmmdispatch(m2.n,
                                        m2.m,
                                        m2.k,
                                        &m2.ldb,
                                        &m2.lda,
                                        &m2.ldc,
                                        &m2.alpha,
                                        &m2.beta,
                                        &m2.flags,
                                        &m2.prefetch);

        m2.kernel(m2.b, m2.a, m2.c);

        /* libxsmm_dgemm("N", */
        /*               "T", */
        /*               &m2.n, */
        /*               &m2.m, */
        /*               &m2.k, */
        /*               &m2.alpha, */
        /*               m2.b, // X_{alpha, i} */
        /*               &m2.ldb, */
        /*               m2.a, // T_{\alpha, \gamma, j} -> transposed such that T_{\gamma, j, \alpha} */
        /*               &m2.lda, */
        /*               &m2.beta, */
        /*               m2.c, // W_{\gamma, j, i} */
        /*               &m2.ldc); */

        m3.prefetch = LIBXSMM_PREFETCH_AUTO;
        m3.flags = LIBXSMM_GEMM_FLAG_TRANS_B;
        m3.kernel = libxsmm_dmmdispatch(m3.n,
                                        m3.m,
                                        m3.k,
                                        &m3.ldb,
                                        &m3.lda,
                                        &m3.ldc,
                                        &m3.alpha,
                                        &m3.beta,
                                        &m3.flags,
                                        &m3.prefetch);

        m3.kernel(m3.b, m3.a, m3.c);

        /* libxsmm_dgemm("N", */
        /*               "T", */
        /*               &m3.n, */
        /*               &m3.m, */
        /*               &m3.k, */
        /*               &m3.alpha, */
        /*               m3.b, // W_{gamma, j, i} */
        /*               &m3.ldb, */
        /*               m3.a, // Z_{\gamma, k} -> Transposed Z_{k, \gamma} */
        /*               &m3.lda, */
        /*               &m3.beta, */
        /*               m3.c, // cube_{kji} */
        /*               &m3.ldc); */


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

        cblas_dgemm(CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m2.m,
                    m2.n,
                    m2.k,
                    1.0,
                    m2.a, // T_{\alpha, \gamma, j} -> transposed such that T_{\gamma, j, \alpha}
                    m2.lda,
                    m2.b, // X_{alpha, i}
                    m2.ldb,
                    0.0,
                    m2.c, // W_{\gamma, j, i}
                    m2.ldc);

        cblas_dgemm(CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m3.m,
                    m3.n,
                    m3.k,
                    1.0,
                    m3.a, // Z_{\gamma, k} -> Transposed Z_{k, \gamma}
                    m3.lda,
                    m3.b, // W_{gamma, j, i}
                    m3.ldb,
                    0.0,
                    m3.c, // cube_{kji}
                    m3.ldc);

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

        dgemm_("N",
               "T",
               &m2.n,
               &m2.m,
               &m2.k,
               &m2.alpha,
               m2.b, // X_{alpha, i}
               &m2.ldb,
               m2.a, // T_{\alpha, \gamma, j} -> transposed such that T_{\gamma, j, \alpha}
               &m2.lda,
               &m2.beta,
               m2.c, // W_{\gamma, j, i}
               &m2.ldc);

        dgemm_("N",
               "T",
               &m3.n,
               &m3.m,
               &m3.k,
               &m3.alpha,
               m3.b, // W_{gamma, j, i}
               &m3.ldb,
               m3.a, // Z_{\gamma, k} -> Transposed Z_{k, \gamma}
               &m3.lda,
               &m3.beta,
               m3.c, // cube_{kji}
               &m3.ldc);
#endif
        return;
    }
    collocate_l0(co,
                 p_alpha_beta_reduced_,
                 cube);
    return;
}

/* It is a sub-optimal version of the mapping in case of a cubic cutoff. But is
 * very general and does not depend on the orthorombic nature of the grid. for
 * orthorombic cases, it is faster to apply PCB directly on the polynomials. */

void compute_blocks(collocation_integration *const handler,
                    const int *lower_boundaries_cube,
                    const int *cube_size,
                    const int *cube_center,
                    const int *period,
                    const tensor *Exp,
                    const int *lb_grid,
                    tensor *grid)
{
    int position[3];
    return_cube_position(grid->size, lb_grid, cube_center, lower_boundaries_cube, period, position);
    mark_collocation_new_pair(handler);
    if ((position[1] + cube_size[1] <= grid->size[1]) &&
        (position[2] + cube_size[2] <= grid->size[2]) &&
        (position[0] + cube_size[0] <= grid->size[0])) {
        // it means that the cube is completely inside the grid without touching
        // the grid borders. periodic boundaries conditions are pointless here.
        // we can simply loop over all three dimensions.

        // it also consider the case where they are open boundaries

        const int upper_corner[3] = {position[0] + cube_size[0],
                                     position[1] + cube_size[1],
                                     position[2] + cube_size[2]};

        add_collocation_block(handler,
                              period,
                              cube_size,
                              NULL,
                              position,
                              upper_corner,
                              Exp,
                              grid);
        return;
    }

    int z1 = position[0];
    int z_offset = 0;
    /* We actually split the cube into smaller parts such that we do not have
     * to apply pcb as a last stage. The blocking takes care of it */

    int lower_corner[3];
    int upper_corner[3];

    for (int z = 0; (z < (cube_size[0] - 1)); z++, z1++) {
        lower_corner[0] = z1;
        upper_corner[0] = compute_next_boundaries(&z1, z, grid->size[0], period[0], cube_size[0]);

        /* // We have a full plane. */

        if (upper_corner[0] - lower_corner[0] > 0) {
            int y1 = position[1];
            int y_offset = 0;
            for (int y = 0; y < cube_size[1]; y++, y1++) {
                lower_corner[1] = y1;
                upper_corner[1] = compute_next_boundaries(&y1, y, grid->size[1], period[1], cube_size[1]);

                /*     // this is needed when the grid is distributed over several ranks. */
                /* if (y1 >= lb_grid[1] + grid->size[1]) */
                /*     continue; */

                if (upper_corner[1] - lower_corner[1] > 0) {
                    int x1 = position[2];
                    int x_offset = 0;

                    for (int x = 0; x < cube_size[2]; x++, x1++) {
                        lower_corner[2] = x1;
                        upper_corner[2] = compute_next_boundaries(&x1, x, grid->size[2], period[2], cube_size[2]);
                        if (upper_corner[2] - lower_corner[2] > 0) {

                            int position2[3]= {z_offset, y_offset, x_offset};

                            add_collocation_block(handler,
                                                  period,
                                                  cube_size,
                                                  position2, // starting position in the subgrid
                                                  lower_corner,
                                                  upper_corner,
                                                  Exp,
                                                  grid);

                        }

                        update_loop_index(lower_corner[2], upper_corner[2], grid->size[2], period[2], &x_offset, &x, &x1);
                    }
                    /* this dimension of the grid is divided over several ranks */
                }
                update_loop_index(lower_corner[1], upper_corner[1], grid->size[1], period[1], &y_offset, &y, &y1);
            }
        }
        update_loop_index(lower_corner[0], upper_corner[0], grid->size[0], period[0], &z_offset, &z, &z1);
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
    int position[3];
    return_cube_position(grid->size,
                         lb_grid,
                         cube_center,
                         lower_boundaries_cube,
                         period,
                         position);

    if ((position[1] + cube->size[1] <= grid->size[1]) &&
        (position[2] + cube->size[2] <= grid->size[2]) &&
        (position[0] + cube->size[0] <= grid->size[0])) {
        // it means that the cube is completely inside the grid without touching
        // the grid borders. periodic boundaries conditions are pointless here.
        // we can simply loop over all three dimensions.

        // it also consider the case where they are open boundaries

        const int lower_corner[3] = {position[0], position[1], position[2]};

        const int upper_corner[3] = {lower_corner[0] + cube->size[0],
                                     lower_corner[1] + cube->size[1],
                                     lower_corner[2] + cube->size[2]};
        const int position1[3] = {0, 0, 0};

        add_sub_grid(lower_corner, // lower corner position where the subgrid should placed
                     upper_corner, // upper boundary
                     position1, // starting position of in the subgrid
                     cube, // subgrid
                     grid);

        return;
    }


    const int offset_x = min(grid->size[2] - position[2], cube->size[2]);;
    const int loop_number_x = (cube->size[2] - offset_x) / period[2];
    const int reminder_x = min(grid->size[2], cube->size[2] - offset_x - loop_number_x * period[2]);

    int z1 = position[0];
    int z_offset = 0;

    int lower_corner[2];
    int upper_corner[2];
    for (int z = 0; (z < cube->size[0]); z++, z1++) {
        lower_corner[0] = z1;
        upper_corner[0] = compute_next_boundaries(&z1, z, grid->size[0], period[0], cube->size[0]);

        /* // We have a full plane. */
        if (upper_corner[0] - lower_corner[0]) {
            int y1 = position[1];
            int y_offset = 0;
            for (int y = 0; y < cube->size[1]; y1++, y++) {
                lower_corner[1] = y1;
                upper_corner[1] = compute_next_boundaries(&y1, y, grid->size[1], period[1], cube->size[1]);

                if (upper_corner[1] - lower_corner[1]) {

                    if ((lower_corner[0] > grid->size[0]) ||
                        (upper_corner[0] > grid->size[0]) ||
                        (lower_corner[1] > grid->size[1]) ||
                        (upper_corner[1] > grid->size[1])) {
                        printf("Problem with the subblock boundaries. Some of them are outside the grid\n");
                        printf("Grid size     : %d %d %d\n", grid->size[0], grid->size[1], grid->size[2]);
                        printf("Grid lb_grid  : %d %d %d\n", lb_grid[0], lb_grid[1], lb_grid[2]);
                        printf("zmin - zmax   : %d %d\n", lower_corner[0], upper_corner[0]);
                        printf("ymin - ymax   : %d %d\n", lower_corner[1], upper_corner[1]);
                        printf("Cube position : %d %d %d\n", position[0], position[1], position[2]);
                        abort();
                    }
                    // we take periodicity into account
                    for (int z2 = lower_corner[0]; z2 < upper_corner[0]; z2++) {
                        double *__restrict dst = &idx3(grid[0], z2, lower_corner[1], 0);
                        const double *__restrict src = &idx3(cube[0], z_offset + z2 - lower_corner[0], y_offset, 0);
                        for (int y2 = 0; y2 < upper_corner[1] - lower_corner[1]; y2++) {

                            // the tail of the queue.
                            LIBXSMM_PRAGMA_SIMD
                            for (int x = 0; x < offset_x; x++)
                                dst[x + position[2]] += src[x];

                            int shift = offset_x;

                            if (loop_number_x) {
                                for (int l = 0; l < loop_number_x; l++) {
                                    LIBXSMM_PRAGMA_SIMD
                                    for (int x = 0; x < grid->size[2]; x++)
                                        dst[x] += src[shift + x];

                                    shift = offset_x + (l + 1)  * period[2];
                                }
                            }
                            LIBXSMM_PRAGMA_SIMD
                            for (int x = 0; x < reminder_x; x++)
                                dst[x] += src[shift + x];

                            dst += grid->ld_;
                            src += cube->ld_;
                        }
                    }
                    update_loop_index(lower_corner[1], upper_corner[1], grid->size[1], period[1], &y_offset, &y, &y1);
                }
            }
            update_loop_index(lower_corner[0], upper_corner[0], grid->size[0], period[0], &z_offset, &z, &z1);
        }
    }
}

// *****************************************************************************
void grid_collocate_ortho(collocation_integration *const handler,
                          const double zetp,
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
    int cube_size[3];
    int lb_cube[3], ub_cube[3];
    double roffset[3];
    double disr_radius;
    /* cube : grid comtaining pointlike product between polynomials
     *
     * pol : grid  containing the polynomials in all three directions
     *
     * pol_folded : grid containing the polynomials after folding for periodic
     * boundaries conditions
     */

    /* seting up the cube parameters */
    int cmax = compute_cube_properties(true,
                                       radius,
                                       dh,
                                       dh_inv,
                                       rp,
                                       &disr_radius,
                                       roffset,
                                       cubecenter,
                                       lb_cube,
                                       ub_cube,
                                       cube_size);

    /* initialize the multidimensional array containing the polynomials */
    initialize_tensor_3(&handler->pol, 3, handler->coef.size[0], 2 * cmax + 1);
    handler->pol_alloc_size = realloc_tensor(handler->pol.data, handler->pol_alloc_size, handler->pol.alloc_size_,  (void **)&handler->pol.data);

    /* allocate memory for the polynomial and the cube */

    if (handler->sequential_mode) {
        initialize_tensor_3(&handler->cube,
                            cube_size[0],
                            cube_size[1],
                            cube_size[2]);

        handler->cube_alloc_size = realloc_tensor(handler->cube.data, handler->cube_alloc_size, handler->cube.alloc_size_, (void **)&handler->cube.data);

        size_t tmp1 = max(handler->T_alloc_size,
                                 compute_memory_space_tensor_3(handler->coef.size[0] /* alpha */,
                                                               handler->coef.size[1] /* gamma */,
                                                               cube_size[1] /* j */));
        size_t tmp2 = max(handler->W_alloc_size,
                          compute_memory_space_tensor_3(handler->coef.size[1] /* gamma */ ,
                                                        cube_size[1] /* j */,
                                                        cube_size[2] /* i */));
        if (((tmp1 + tmp2) > (handler->T_alloc_size + handler->W_alloc_size)) ||
            (handler->scratch == NULL)) {
            handler->T_alloc_size = tmp1;
            handler->W_alloc_size = tmp2;
            if (handler->scratch)
                free(handler->scratch);
            if (posix_memalign(&handler->scratch, 64, sizeof(double) * (tmp1 + tmp2)) != 0)
                abort();
        }
    }


    /* compute the polynomials */

    // WARNING : do not reverse the order in pol otherwise you will have to
    // reverse the order in collocate_dgemm as well.

    grid_fill_pol(false, dh[0][0], roffset[2], lb_cube[2], handler->coef.size[2] - 1, cmax, zetp, &idx3(handler->pol, 2, 0, 0)); /* i indice */
    grid_fill_pol(false, dh[1][1], roffset[1], lb_cube[1], handler->coef.size[1] - 1, cmax, zetp, &idx3(handler->pol, 1, 0, 0)); /* j indice */
    grid_fill_pol(false, dh[2][2], roffset[0], lb_cube[0], handler->coef.size[0] - 1, cmax, zetp, &idx3(handler->pol, 0, 0, 0)); /* k indice */

    if (handler->sequential_mode) {
        collocate_core_rectangular(handler->scratch,
                                   // pointer to scratch memory
                                   1.0,
                                   &handler->coef,
                                   &handler->pol,
                                   &handler->cube);

        /* apply_mapping(disr_radius, dh, dh_inv, map, lb_cube, &handler->cube, cmax, grid); */

        apply_mapping_cubic(lb_cube, cubecenter, npts, &handler->cube, lb_grid, grid);
    } else {
        compute_blocks(handler,
                       lb_cube,
                       cube_size,
                       cubecenter,
                       npts,
                       NULL,
                       lb_grid,
                       grid);
    }
}

// *****************************************************************************
void grid_collocate_generic(collocation_integration *const handler,
                            const double zetp,
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
    int cube_size[3];
    int lb_cube[3], ub_cube[3];
    double roffset[3];
    double disr_radius;
    /* cube : grid comtaining pointlike product between polynomials
     *
     * pol : grid  containing the polynomials in all three directions
     *
     * pol_folded : grid containing the polynomials after folding for periodic
     * boundaries conditions
     */

    /* seting up the cube parameters */
    int cmax = compute_cube_properties(false,
                                       radius,
                                       dh,
                                       dh_inv,
                                       rp,
                                       &disr_radius,
                                       roffset,
                                       cubecenter,
                                       lb_cube,
                                       ub_cube,
                                       cube_size);

    /* initialize the multidimensional array containing the polynomials */

    initialize_tensor_3(&handler->pol, 3, handler->coef.size[0], 2 * cmax + 1);
    handler->pol_alloc_size = realloc_tensor(handler->pol.data, handler->pol_alloc_size, handler->pol.alloc_size_,  (void **)&handler->pol.data);

    initialize_tensor_3(&handler->Exp, 3, max(cube_size[0], cube_size[1]), max(cube_size[1], cube_size[2]));
    handler->Exp_alloc_size = realloc_tensor(handler->Exp.data, handler->Exp_alloc_size, handler->Exp.alloc_size_, (void **)&handler->Exp.data);

    /* allocate memory for the polynomial and the cube */

    if (handler->sequential_mode) {
        initialize_tensor_3(&handler->cube,
                            cube_size[0],
                            cube_size[1],
                            cube_size[2]);

        handler->cube_alloc_size = realloc_tensor(handler->cube.data, handler->cube_alloc_size, handler->cube.alloc_size_, (void **)&handler->cube.data);

        size_t tmp1 = max(handler->T_alloc_size,
                                 compute_memory_space_tensor_3(handler->coef.size[0] /* alpha */,
                                                               handler->coef.size[1] /* gamma */,
                                                               cube_size[1] /* j */));
        size_t tmp2 = max(handler->W_alloc_size,
                          compute_memory_space_tensor_3(handler->coef.size[1] /* gamma */ ,
                                                        cube_size[1] /* j */,
                                                        cube_size[2] /* i */));
        if (((tmp1 + tmp2) > (handler->T_alloc_size + handler->W_alloc_size)) ||
            (handler->scratch == NULL)) {
            handler->T_alloc_size = tmp1;
            handler->W_alloc_size = tmp2;
            if (handler->scratch)
                free(handler->scratch);
            if (posix_memalign(&handler->scratch, 64, sizeof(double) * (tmp1 + tmp2)) != 0)
                abort();
        }
    }

    /* compute the polynomials */

    // WARNING : do not reverse the order in pol otherwise you will have to
    // reverse the order in collocate_dgemm as well.

    double dx[3];

    dx[2] = sqrt(dh[0][0] * dh[0][0] + dh[0][1] * dh[0][1] + dh[0][2] * dh[0][2]);
    dx[1] = sqrt(dh[1][0] * dh[1][0] + dh[1][1] * dh[1][1] + dh[1][2] * dh[1][2]);
    dx[0] = sqrt(dh[2][0] * dh[2][0] + dh[2][1] * dh[2][1] + dh[2][2] * dh[2][2]);

    grid_fill_pol(false, dx[0], roffset[0], lb_cube[0], handler->coef.size[0] - 1, cmax, zetp, &idx3(handler->pol, 0, 0, 0)); /* k indice */
    grid_fill_pol(false, dx[1], roffset[1], lb_cube[1], handler->coef.size[1] - 1, cmax, zetp, &idx3(handler->pol, 1, 0, 0)); /* j indice */
    grid_fill_pol(false, dx[2], roffset[2], lb_cube[2], handler->coef.size[2] - 1, cmax, zetp, &idx3(handler->pol, 2, 0, 0)); /* i indice */

    calculate_non_orthorombic_corrections_tensor(zetp,
                                                 roffset,
                                                 dh,
                                                 handler->cube.size,
                                                 &handler->Exp);

    /* Use a slightly modified version of Ole code */
    grid_transform_coef_xyz_to_ijk(dh, dh_inv, &handler->coef);

    if (handler->sequential_mode) {
        collocate_core_rectangular(handler->scratch,
                                   // pointer to scratch memory
                                   1.0,
                                   &handler->coef,
                                   &handler->pol,
                                   &handler->cube);

        apply_non_orthorombic_corrections(&handler->Exp, &handler->cube);
        /* apply_mapping(disr_radius, dh, dh_inv, map, lb_cube, &handler->cube, cmax, grid); */

        apply_mapping_cubic(lb_cube, cubecenter, npts, &handler->cube, lb_grid, grid);
    } else {
        compute_blocks(handler,
                       lb_cube,
                       cube_size,
                       cubecenter,
                       npts,
                       NULL,
                       lb_grid,
                       grid);
    }
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
    /* for (int lzp=0; lzp<=lp; lzp++) { */
    /*     for (int lyp=0; lyp<=lp; lyp++) { */
    /*         for (int lxp=0; lxp<=lp; lxp++) { */
    /*             coef_map[lzp][lyp][lxp] = INT_MAX; */
    /*         } */
    /*     } */
    /* } */

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
void grid_collocate_internal(collocation_integration *const handler,
                             const bool use_ortho,
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

    /* precautionary tail since I probably intitialize data to NULL when I
     * initialize a new tensor. I want to keep the memory (I put a ridiculous
     * amount already) */

    void *tmp = handler->alpha.data;
    initialize_tensor_4(&handler->alpha, 3, lmax_prep[1] + 1, lmax_prep[0] + 1, lmax_prep[0] + lmax_prep[1] + 1);

    handler->alpha_alloc_size = realloc_tensor(tmp,
                                                handler->alpha_alloc_size,
                                                handler->alpha.alloc_size_,
                                               (void **)&(handler->alpha.data));

    const int lp = lmax_prep[0] + lmax_prep[1];

    tmp = handler->coef.data;
    initialize_tensor_3(&(handler->coef), lp + 1, lp + 1, lp + 1);

    handler->coef_alloc_size = realloc_tensor(tmp,
                                              handler->coef_alloc_size,
                                              handler->coef.alloc_size_,
                                              (void **)&(handler->coef.data));

    // initialy cp2k stores coef_xyz as coef[z][y][x]. this is fine but I
    // need them to be stored as

    grid_prepare_alpha(ra,
                       rb,
                       rp,
                       lmax_prep,
                       &handler->alpha);

    //
    //   compute P_{lxp,lyp,lzp} given P_{lxa,lya,lza,lxb,lyb,lzb} and alpha(ls,lxa,lxb,1)
    //   use a three step procedure
    //   we don't store zeros, so counting is done using lxyz,lxy in order to have
    //   contiguous memory access in collocate_fast.F
    //

    // coef[x][z][y]
    grid_prepare_coef(use_ortho,
                      lmax_prep,
                      lmin_prep,
                      lp,
                      prefactor,
                      &handler->alpha,
                      pab_prep,
                      &handler->coef);

    if (use_ortho) {
        const int period[3] = {npts[2], npts[1], npts[0]};
        const int lb_grid_bis[3] = {lb_grid[2], lb_grid[1], lb_grid[0]};
        const bool periodic_bis[3] = {periodic[2], periodic[1], periodic[0]};

        grid_collocate_ortho(handler,
                             zetp,
                             dh,
                             dh_inv,
                             rp,
                             period,
                             lb_grid_bis,
                             periodic_bis,
                             radius,
                             grid);
    } else {

// initialy cp2k stores coef_xyz as coef[z][y][x]. this is fine but I
        // need them to be stored as
        const int period[3] = {npts[2], npts[1], npts[0]};
        const int lb_grid_bis[3] = {lb_grid[2], lb_grid[1], lb_grid[0]};
        const bool periodic_bis[3] = {periodic[2], periodic[1], periodic[0]};
        grid_collocate_generic(handler,
                               zetp,
                               dh,
                               dh_inv,
                               rp,
                               period,
                               lb_grid_bis,
                               periodic_bis,
                               radius,
                               grid);
        /* grid_collocate_general(lp, */
        /*                        zetp, */
        /*                        &handler->coef, */
        /*                        dh, */
        /*                        dh_inv, */
        /*                        rp, */
        /*                        npts, */
        /*                        lb_grid, */
        /*                        periodic, */
        /*                        radius, */
        /*                        ngrid, */
        /*                        grid); */

    }
}


// *****************************************************************************
void grid_collocate_pgf_product_cpu(void *const handle,
                                    const bool use_ortho,
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
    posix_memalign((void**)&grid_before.data, 64, sizeof(double) * grid_before.alloc_size_);
    // we have a buffer of 4 M

    for (int i = 0; i < ngrid[2]; i++) {
        for (int j = 0; j < ngrid[1]; j++) {
            for (int k = 0; k < ngrid[0]; k++) {
                idx3(grid_before, i, j, k) = idx3(grid, i, j, k);
            }
        }
    }
    memset(grid.data, 0, sizeof(double) grid.alloc_size_);
#endif

    grid_collocate_internal(handle,
                            use_ortho,
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
    free(grid_before.data);
#endif

}

//EOF
