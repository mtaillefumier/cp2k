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
#include <stdbool.h>
#if defined(__MKL) || defined(HAVE_MKL)
#include <mkl.h>
#include <mkl_cblas.h>
#endif

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

#include "grid_collocate_cpu.h"
#include "grid_prepare_pab.h"
#include "grid_common.h"
#include "collocation_integration.h"
#include "tensor_local.h"
#include "utils.h"
#include "coefficients.h"
#include "thpool.h"

extern void grid_fill_pol(const bool transpose,
                          const double dr,
                          const double roffset,
                          const int xmin,
                          const int xmax,
                          const int lp,
                          const int cmax,
                          const double zetp,
                          double *pol_);

void extract_cube(const int *lower_boundaries_cube,
                  const int *cube_center,
                  const int *period,
                  const tensor *grid,
                  const int *lb_grid,
                  tensor *cube);

extern void compute_blocks(collocation_integration *const handler,
                           const int *lower_boundaries_cube,
                           const int *cube_size,
                           const int *cube_center,
                           const int *period,
                           const tensor *Exp,
                           const int *lb_grid,
                           tensor *grid);

extern void collocate_core_rectangular(double *scratch,
                                       const double prefactor,
                                       const struct tensor_ *co,
                                       const struct tensor_ *p_alpha_beta_reduced_,
                                       struct tensor_ *cube);

/* compute the following operation (variant) it is a tensor contraction

   V_{kji} = \sum_{\alpha\beta\gamma} C_{\alpha\gamma\beta} T_{2,\alpha,i} T_{1,\beta,j} T_{0,\gamma,k}

*/
void integrate_core_rectangular(double *scratch,
                                const double prefactor,
                                const struct tensor_ *cube,
                                const struct tensor_ *p_alpha_beta_reduced_,
                                struct tensor_ *co)
{

    if (co->size[0] == 1) {
        /* it is very specific to integrate because we might end up with a single
         * element after the tensor product. In that case, I call the specific case
         * with l = 0 and then do a scalar product between the two. We can not get
         * one of the dimensions at 1 and all the other above. It is physics non
         * sense */
        tensor cube_tmp;
        initialize_tensor_3(&cube_tmp, cube->size[0], cube->size[1], cube->size[2]);
        posix_memalign((void **)&cube_tmp.data, 32, sizeof(double) * cube_tmp.alloc_size_);
        collocate_l0(prefactor,
                     p_alpha_beta_reduced_,
                     &cube_tmp);
        co->data[0] = cblas_ddot(cube_tmp.alloc_size_, cube_tmp.data, 1, cube->data, 1);
        free(cube_tmp.data);
        return;
    }

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

        initialize_tensor_3(&T, cube->size[0] /* k */, cube->size[1] /* j */, co->size[1] /* alpha */);
        initialize_tensor_3(&W, cube->size[1] /* j */ , co->size[1] /* alpha */, co->size[2] /* gamma */);

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
        m1.m = cube->size[0] * cube->size[1]; /* z y */
        m1.n = co->size[1]; /* alpha */
        m1.k = cube->size[2]; /* i */
        m1.a = cube->data; // V_{kji}
        m1.lda = cube->ld_;
        m1.b = px; // X_{i, alpha} = p_alpha_beta_reduced(2, i, alpha)
        m1.ldb = p_alpha_beta_reduced_->ld_;
        m1.c = T.data; // T_{k, j, alpha} = T(k, j, alpha)
        m1.ldc = T.ld_;

        /*
         * the next step is a reduction along the alpha index.
         *
         * We compute then
         *
         * W_{j, alpha, gamma} = sum_{k} T_{j, alpha, k} Z_{k, \gamma}
         *
         * which means we need to transpose T_{\alpha, \gamma, j} to get
         * T_{\gamma, j, \alpha}. Fortunately we can do it while doing the
         * matrix - matrix multiplication
         */

        m2.alpha = 1.0;
        m2.beta = 0.0;
        m2.m = cube->size[1] * co->size[1]; // j \alpha direction
        m2.n = co->size[2]; // z
        m2.k = cube->size[0]; // z
        m2.a = T.data; // T_{j, alpha, gamma}
        m2.lda = T.ld_ * cube->size[1];
        m2.b = pz; // Z_{k, gamma}  = p_alpha_beta_reduced(0, k, gamma)
        m2.ldb = p_alpha_beta_reduced_->ld_;
        m2.c = W.data; // W_{j, \alpha, \gamma}
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
        m3.m = co->size[0]; // Y_{j, beta}
        m3.n = co->size[1] * co->size[2]; // (alpha,gamma) direction
        m3.k = cube->size[1]; // y
        m3.a = py; // p_alpha_beta_reduced(1, j, beta)
        m3.lda = p_alpha_beta_reduced_->ld_;
        m3.b = &idx3(W, 0, 0, 0); // W_{j, alpha, gamma}
        m3.ldb = W.size[1] * W.ld_;
        m3.c = &idx3(co[0], 0, 0, 0); // co_{beta,alpha,gamma}
        m3.ldc = co->ld_ * co->size[1];

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
        if (!m1.kernel)
            abort();

        m1.kernel(m1.b, m1.a, m1.c);

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
        if (!m2.kernel)
        {
            printf("matrix size m = %d, n = %d, k = %d\n", m2.m, m2.n, m2.k);
            printf("leading dimensions lda = %d, ldb = %d, ldc = %d\n", m2.lda, m2.ldb, m2.ldc);

            // fall back to mkl
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
        } else {
            m2.kernel(m2.b, m2.a, m2.c);
        }

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

        if (!m3.kernel)
            abort();

        m3.kernel(m3.b, m3.a, m3.c);

#elif defined(__MKL)
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m1.m,
                    m1.n,
                    m1.k,
                    m1.alpha,
                    m1.a,
                    m1.lda,
                    m1.b,
                    m1.ldb,
                    m1.beta,
                    m1.c, tmp_{alpha, gamma, j}
                    m1.ldc);

        cblas_dgemm(CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m2.m,
                    m2.n,
                    m2.k,
                    m2.alpha,
                    m2.a, T_{\alpha, \gamma, j} -> transposed such that T_{\gamma, j, \alpha}
                    m2.lda,
                    m2.b, X_{alpha, i}
                    m2.ldb,
                    m2.beta,
                    m2.c, W_{\gamma, j, i}
                    m2.ldc);

        cblas_dgemm(CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    m3.m,
                    m3.n,
                    m3.k,
                    m3.alpha,
                    m3.a, Z_{\gamma, k} -> Transposed Z_{k, \gamma}
                    m3.lda,
                    m3.b, W_{gamma, j, i}
                    m3.ldb,
                    m3.beta,
                    m3.c, cube_{kji}
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
    collocate_l0(co->data[0],
                 p_alpha_beta_reduced_,
                 cube);
    return;
}


void extract_cube(const int *lower_boundaries_cube,
                  const int *cube_center,
                  const int *period,
                  const tensor *grid,
                  const int *lb_grid,
                  tensor *cube)
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

        extract_sub_grid(lower_corner, // lower corner position where the subgrid should placed
                         upper_corner, // upper boundary
                         position1, // starting position of in the subgrid
                         grid,
                         cube);

        return;
    }


    int z1 = position[0];
    int z_offset = 0;
    int lower_corner[3];
    int upper_corner[3];
    for (int z = 0; (z < cube->size[0]); z++, z1++) {
        /* see utils.h */
        lower_corner[0] = z1;
        upper_corner[0] = compute_next_boundaries(&z1, z, grid->size[0], period[0], cube->size[0]);

        /* // We have a full plane. */
        if (upper_corner[0] - lower_corner[0]) {
            int y1 = position[1];
            int y_offset = 0;
            for (int y = 0; y < cube->size[1]; y1++, y++) {
                /* see utils.h */
                lower_corner[1] = y1;
                upper_corner[1] = compute_next_boundaries(&y1, y, grid->size[1], period[1], cube->size[1]);

                if (upper_corner[1] - lower_corner[1]) {

                    if ((upper_corner[0] > grid->size[0]) ||
                        (upper_corner[0] > grid->size[0]) ||
                        (lower_corner[1] > grid->size[1]) ||
                        (upper_corner[1] > grid->size[1])) {
                        printf("Problem with the subblock boundaries. Some of them are outside the grid\n");
                        printf("Grid size     : %d %d %d\n", grid->size[0], grid->size[1], grid->size[2]);
                        printf("Grid lb_grid  : %d %d %d\n", lb_grid[0], lb_grid[1], lb_grid[2]);
                        printf("zmin-zmax     : %d %d\n", lower_corner[0], upper_corner[0]);
                        printf("ymin-ymax     : %d %d\n", lower_corner[1], upper_corner[1]);
                        printf("Cube position : %d %d %d\n", position[0], position[1], position[2]);
                        abort();
                    }

                    int x1 = position[2];
                    int x_offset = 0;
                    for (int x = 0; x < cube->size[2]; x++, x1++) {
                        /* see utils.h */
                        lower_corner[2] = x1;
                        upper_corner[2] = compute_next_boundaries(&x1, x, grid->size[2], period[2], cube->size[2]);
                        if (upper_corner[2] - lower_corner[2]) {
                            int position2[3]= {z_offset, y_offset, x_offset};

                            extract_sub_grid(lower_corner,
                                             upper_corner,
                                             position2, // starting position in the subgrid
                                             grid,
                                             cube);

                        }
                        update_loop_index(lower_corner[2], upper_corner[2], grid->size[2], period[2], &x_offset, &x, &x1);
                    }
                }
                update_loop_index(lower_corner[1], upper_corner[1], grid->size[1], period[1], &y_offset, &y, &y1);
            }
        }
        update_loop_index(lower_corner[0], upper_corner[0], grid->size[0], period[0], &z_offset, &z, &z1);
    }
}

void grid_integrate(collocation_integration *const handler,
                    const bool use_ortho,
                    const int lp,
                    const double zetp,
                    const double dh[3][3],
                    const double dh_inv[3][3],
                    const double rp[3],
                    const int npts[3],
                    const int lb_grid[3],
                    const bool periodic[3],
                    const double radius,
                    const tensor *grid)
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
    int cmax = compute_cube_properties(use_ortho,
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

    /* printf("lower boundaries %d %d %d\n", lb_cube[0], lb_cube[1], lb_cube[2]); */
    /* printf("upper boundaries %d %d %d\n", ub_cube[0], ub_cube[1], ub_cube[2]); */
    /* printf("offset %.15lf %.15lf %.15lf\n", roffset[0], roffset[1], roffset[2]); */
    /* printf("cube center %d %d %d\n", cubecenter[0], cubecenter[1], cubecenter[2]); */
    /* printf("gauss center %.15lf %.15lf %.15lf\n", rp[0], rp[1], rp[2]); */
    /* printf("gaussian factor %.15lf\n", zetp); */
    /*Initialize the table for the coefficients */
    initialize_tensor_3(&handler->coef, lp + 1, lp + 1, lp + 1);
    handler->coef_alloc_size = realloc_tensor(&handler->coef);

    /* initialize the multidimensional array containing the polynomials */
    if (lp != 0) {
        initialize_tensor_3(&handler->pol, 3, 2 * cmax + 1, handler->coef.size[0]);
    } else {
        initialize_tensor_3(&handler->pol, 3, handler->coef.size[0], 2 * cmax + 1);
    }
    handler->pol_alloc_size = realloc_tensor(&handler->pol);

    /* allocate memory for the polynomial and the cube */

    if (handler->sequential_mode) {
        initialize_tensor_3(&handler->cube,
                            cube_size[0],
                            cube_size[1],
                            cube_size[2]);

        handler->cube_alloc_size = realloc_tensor(&handler->cube);

        initialize_W_and_T(handler, &handler->coef, &handler->cube);
    }

    /* compute the polynomials */

    // WARNING : do not reverse the order in pol otherwise you will have to
    // reverse the order in collocate_dgemm as well.

    /* the tensor contraction is done for a given order so I have to be careful
     * how the tensors X, Y, Z are stored. In collocate, they are stored
     * normally 0 (Z), (1) Y, (2) X in the table pol. but in integrate (which
     * uses the same tensor reduction), I have a special treatment for l = 0. In
     * that case the order *should* be the same than for collocate. For l > 0,
     * we need a different storage order which is (X) 2, (Y) 0, and (Z) 1. */

    int perm[3] = {2, 0, 1};

    if (lp == 0) {
        /* I need to restore the same order than for collocate */
        perm[0] = 0;
        perm[1] = 1;
        perm[2] = 2;
    }

    if (use_ortho) {
        grid_fill_pol((lp != 0), dh[0][0], roffset[2], lb_cube[2], ub_cube[2], lp, cmax, zetp, &idx3(handler->pol, perm[2], 0, 0)); /* i indice */
        grid_fill_pol((lp != 0), dh[1][1], roffset[1], lb_cube[1], ub_cube[1], lp, cmax, zetp, &idx3(handler->pol, perm[1], 0, 0)); /* j indice */
        grid_fill_pol((lp != 0), dh[2][2], roffset[0], lb_cube[0], ub_cube[0], lp, cmax, zetp, &idx3(handler->pol, perm[0], 0, 0)); /* k indice */
    } else {
        initialize_tensor_3(&handler->Exp, 3, max(cube_size[0], cube_size[1]), max(cube_size[1], cube_size[2]));
        handler->Exp_alloc_size = realloc_tensor(&handler->Exp);

        double dx[3];
        dx[2] = dh[0][0] * dh[0][0] + dh[0][1] * dh[0][1] + dh[0][2] * dh[0][2];
        dx[1] = dh[1][0] * dh[1][0] + dh[1][1] * dh[1][1] + dh[1][2] * dh[1][2];
        dx[0] = dh[2][0] * dh[2][0] + dh[2][1] * dh[2][1] + dh[2][2] * dh[2][2];

        grid_fill_pol((lp != 0), 1.0, roffset[2], lb_cube[2], ub_cube[2], lp, cmax, zetp * dx[2], &idx3(handler->pol, perm[2], 0, 0)); /* i indice */
        grid_fill_pol((lp != 0), 1.0, roffset[1], lb_cube[1], ub_cube[1], lp, cmax, zetp * dx[1], &idx3(handler->pol, perm[1], 0, 0)); /* j indice */
        grid_fill_pol((lp != 0), 1.0, roffset[0], lb_cube[0], ub_cube[0], lp, cmax, zetp * dx[0], &idx3(handler->pol, perm[0], 0, 0)); /* k indice */

        calculate_non_orthorombic_corrections_tensor(zetp,
                                                     roffset,
                                                     dh,
                                                     lb_cube,
                                                     ub_cube,
                                                     &handler->Exp);
    }

    if (handler->sequential_mode) {
        extract_cube(lb_cube,
                     cubecenter,
                     npts,
                     grid,
                     lb_grid,
                     &handler->cube);

        if (!use_ortho)
            apply_non_orthorombic_corrections(&handler->Exp, &handler->cube);

        collocate_core_rectangular(handler->scratch,
                                   // pointer to scratch memory
                                   1.0,
                                   &handler->cube,
                                   &handler->pol,
                                   &handler->coef);

        if (!use_ortho) {
            /* It is actually the same multinomial transformation than
             * grid_transform_coef_xyz_to_ijk with dh and dh_inv permuted. For
             * clarity I created a function to indicate the which transformation
             * is done. */
            grid_transform_coef_jik_to_yxz(dh, &handler->coef);
        }
    } else {
        compute_blocks(handler,
                       lb_cube,
                       cube_size,
                       cubecenter,
                       npts,
                       use_ortho ? NULL : &handler->Exp,
                       lb_grid,
                       grid);
    }
}


// *****************************************************************************
void grid_integrate_pgf_product_cpu(void *const handle,
                                    const bool use_ortho,
                                    const int lp,
                                    const double zeta,
                                    const double zetb,
                                    const double dh[3][3],
                                    const double dh_inv[3][3],
                                    const double ra[3],
                                    const double rab[3],
                                    const int npts[3],
                                    const int ngrid[3],
                                    const int lb_grid[3],
                                    const bool periodic[3],
                                    const double radius,
                                    double *const grid_,
                                    double *const coef)
{

    if (!handle) {
        abort();
    }

    collocation_integration *handler = (collocation_integration *)handle;
    const double zetp = zeta + zetb;
    const double f = zetb / zetp;
    const double rab2 = rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2];
    const double prefactor = exp(-zeta * f * rab2);
    const int period[3] = {npts[2], npts[1], npts[0]};
    const int lb_grid_bis[3] = {lb_grid[2], lb_grid[1], lb_grid[0]};
    const bool periodic_bis[3] = {periodic[2], periodic[1], periodic[0]};

    tensor grid;
    initialize_tensor_3(&grid, ngrid[2], ngrid[1], ngrid[0]);
    grid.ld_ = ngrid[0];
    grid.data = grid_;
    double rp[3], rb[3];
    for (int i=0; i<3; i++) {
        rp[i] = ra[i] + f * rab[i];
        rb[i] = ra[i] + rab[i];
    }

    grid_integrate(handler,
                   use_ortho,
                   lp,
                   zetp,
                   dh,
                   dh_inv,
                   rp,
                   period,
                   lb_grid_bis,
                   periodic_bis,
                   radius,
                   &grid);

    /* I need to transpose the coefficients because they are computed as ly, lx,
     * lz while we want them in the format lz, ly, lx. Fortunately it is a
     * single transpose. So either I include it in the next tranformation or I
     * do it separately. */

    transform_yxz_to_triangular(&handler->coef, coef);
    /* Return the result to cp2k for now */
}
