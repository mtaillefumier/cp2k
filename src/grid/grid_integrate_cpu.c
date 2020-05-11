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
                          const int pol_offset,
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

extern void tensor_reduction_for_collocate_integrate(double *scratch,
                                                     const double alpha,
                                                     const double beta,
                                                     const struct tensor_ *co,
                                                     const struct tensor_ *p_alpha_beta_reduced_,
                                                     struct tensor_ *cube);

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

double integrate_l0_blocked_z(double *scratch,
                              const int z0,
                              const struct tensor_ *const p_alpha_beta_reduced_,
                              struct tensor_ *const grid)
{
    const double *__restrict const pz = &idx3(p_alpha_beta_reduced_[0], 0, 0, z0); /* k indice */
    double *__restrict__ dst = &idx3(grid[0], 0, 0, 0);
    LIBXSMM_PRAGMA_SIMD
        for (int s = 0; s < grid->size[1] * grid->ld_; s++) {
            dst[s] *= pz[0];
        }

    for (int z1 = 1; z1 < grid->size[0]; z1++) {
        double *__restrict__ src = &idx3(grid[0], z1, 0, 0);
        const double pzz = pz[z1];
        cblas_daxpy(grid->size[1] * grid->ld_, pzz, src, 1, dst, 1);
    }

    return cblas_ddot(grid->size[1] * grid->ld_, &idx3(grid[0], 0, 0, 0), 1, scratch, 1);
}

/*
 * It is from an algebraic point of view the same operation than the tensor
 * reduction for the collocate routine. In practice however we must be careful
 * when we compute only one element (l_1 + l_2 = 0)
 * compute the following operation (variant) it is a tensor contraction

   V_{kji} = \sum_{\alpha\beta\gamma} C_{\alpha\gamma\beta} T_{2,\alpha,i} T_{1,\beta,j} T_{0,\gamma,k}

*/
void tensor_reduction_for_integrate_blocked(double *scratch,
                                            const double alpha,
                                            const double beta,
                                            const int *lower_block,
                                            const int *upper_block,
                                            const bool *orthogonal,
                                            const struct tensor_ *Exp,
                                            const struct tensor_ *gr,
                                            const struct tensor_ *p_alpha_beta_reduced_,
                                            struct tensor_ *coef)
{
    assert(scratch != NULL);
    assert(p_alpha_beta_reduced_);
    assert(gr != NULL);
    assert(gr->data != NULL);

    assert((lower_block[0] >= 0) && (lower_block[0] < gr->size[0]));
    assert((lower_block[1] >= 0) && (lower_block[1] < gr->size[1]));
    assert((lower_block[2] >= 0) && (lower_block[2] < gr->size[2]));

    assert(upper_block[0] - lower_block[0]);
    assert(upper_block[1] - lower_block[1]);
    assert(upper_block[2] - lower_block[2]);

    assert(gr->block->size[0] == gr->blockDim[0]);
    assert(gr->block->size[1] == gr->blockDim[1]);
    assert(gr->block->size[2] == gr->blockDim[2]);

    if (!gr->blocked_decomposition)
    {
        printf("Erro : You are using the wrong version of the tensor reduction.\n");
        printf("You should use tensor_reduction_for_collocate_integrate instead\n");
        printf("or initialize the gr tensor as blocked\\n");
        abort();
    }

    // build a list of matrix-matrix multiplications
    tensor T, W;
    dgemm_params *m1, *m2, *m3;


    if (coef->size[0] > 1) {
        const int list_length = (upper_block[0] - lower_block[0]) *
            (upper_block[1] - lower_block[1]) *
            (upper_block[2] - lower_block[2]);
        m1 = (dgemm_params *)malloc(sizeof(dgemm_params) *
                                    list_length);
        m2 = (dgemm_params *)malloc(sizeof(dgemm_params) *
                                    list_length);
        m3 = (dgemm_params *)malloc(sizeof(dgemm_params) *
                                    list_length);

        memset(m3, 0, sizeof(dgemm_params) * list_length);
        memset(m2, 0, sizeof(dgemm_params) * list_length);
        memset(m1, 0, sizeof(dgemm_params) * list_length);

        initialize_tensor_4(&T,
                            list_length,
                            gr->blockDim[0] /* k */,
                            gr->blockDim[1] /* j */,
                            coef->size[2]/* alpha */);


        initialize_tensor_4(&W,
                            list_length,
                            gr->blockDim[1] /* j */ ,
                            coef->size[1] /* alpha */,
                            coef->size[2] /* gamma */);

        T.data = scratch;
        W.data = scratch + T.alloc_size_;

        /* WARNING we are in row major layout. cblas allows it and it is more
         * natural to read left to right than top to bottom
         *
         * we do first T_{k,j,\alpha} = \sum_beta V_{kji} X_{i, \alpha}
         *
         * keep in mind that X_{i, alpha} = p_alpha_beta_reduced(2, i, alpha)
         * and the order of indices is also important. the last indice is the
         * fastest one. it can be done with one dgemm.
         */
    }

    int indz = 0;

    coef->data[0] = 0.0;

    for (int z = 0; z < (upper_block[0] - lower_block[0]); z++) {
        int z1 = (z + lower_block[0]) % gr->size[0];
        for (int y = 0; y < (upper_block[1] - lower_block[1]); y++) {
            int y1 = (y + lower_block[1]) % gr->size[1];
            for (int x = 0; x < (upper_block[2] - lower_block[2]); x++) {
                int x1 = (x + lower_block[2]) % gr->size[2];
                double *__restrict const px = &idx3(p_alpha_beta_reduced_[0],
                                                    2,
                                                    x * gr->blockDim[2],
                                                    0); /* i indice */

                /* if (!orthogonal[0] || !orthogonal[1] || !orthogonal[2]) { */
                /*     int position[3] = {z * gr->blockDim[0], y * gr->blockDim[1], x * gr->blockDim[2]}; */
                /*     gr->block->data = scratch + indz * gr->block->alloc_size_; */
                /*     memcpy(gr->block->data, &idx4(gr[0], z1, y1, x1, 0), sizeof(double) * gr->block->alloc_size_); */
                /*     apply_non_orthorombic_corrections_subblock(orthogonal, */
                /*                                                position, */
                /*                                                Exp, */
                /*                                                gr->block); */
                /* } else { */
                gr->block->data = &idx4(gr[0], z1, y1, x1, 0);
                /* } */

                /* Maybe write a kernel with ispc since we do this all the time. Right now rely on batched dgemm */
                /* will need to go from array of structure to structure of array. What a pain */
                if (coef->size[0] > 1) {
                    m1[indz].op1 = 'N';
                    m1[indz].op2 = 'N';
                    m1[indz].alpha = alpha;
                    m1[indz].beta = 0.0;
                    m1[indz].m = gr->blockDim[0] * gr->blockDim[1]; /* k j */
                    m1[indz].n = coef->size[1]; /* alpha */
                    m1[indz].k = gr->blockDim[2]; /* i */
                    m1[indz].a = gr->block->data; // gr_{kji}
                    m1[indz].lda = gr->block->ld_;
                    m1[indz].b = px; // X_{j, alpha} = p_alpha_beta_reduced(2, j, alpha)
                    m1[indz].ldb = p_alpha_beta_reduced_->ld_;
                    m1[indz].c = &idx4(T, indz, 0, 0, 0); // T_{k, j, alpha}
                    m1[indz].ldc = T.ld_;

                    /*
                     * the next step is a reduction along the alpha index.
                     *
                     * We compute then
                     *
                     * W_{j, alpha, gamma} = sum_{k} T_{k, j, alpha} Z_{k, gamma}
                     *
                     * which means we need to transpose T_{k, j, alpha} to get
                     * T_{j, \alpha, k}. Fortunately we can do it while doing the
                     * matrix - matrix multiplication
                     */


                    m2[indz].op1='T';
                    m2[indz].op2='N';
                    m2[indz].alpha = 1.0;
                    m2[indz].beta = 0.0;
                    m2[indz].m = coef->size[1] * gr->blockDim[1]; // (j, alpha)
                    m2[indz].n = coef->size[2]; // gamma
                    m2[indz].k = gr->blockDim[0]; // k
                    m2[indz].a = &idx4(T, indz, 0, 0, 0); // T_{k, j, \alpha} -> need to transpose this
                    m2[indz].lda = T.ld_ * gr->blockDim[1];
                    m2[indz].b = &idx3(p_alpha_beta_reduced_[0], 0, z * gr->blockDim[0], 0); // X_{alpha, i}  = p_alpha_beta_reduced(0, alpha, i)
                    m2[indz].ldb = p_alpha_beta_reduced_->ld_;
                    m2[indz].c = &idx4(W, indz, 0, 0, 0); // W_{j, alpha, gamma}
                    m2[indz].ldc = W.ld_;

                    /* the final step is again a reduction along the gamma indice. It can
                     * again be done with one dgemm. The operation is simply
                     *
                     * coef_{beta,alpha,gamma} = \sum_{j} Y_{\beta, j} W_{j, alpha, gamma}
                     *
                     * which means we need to transpose Y_{k, beta}.
                     */
                    m3[indz].op1 = 'T';
                    m3[indz].op2 = 'N';
                    m3[indz].alpha = alpha;
                    m3[indz].m = coef->size[1]; // Y_{j, beta}
                    m3[indz].n = coef->size[1] * coef->size[2]; // (alpha gamma) direction
                    m3[indz].k = gr->blockDim[1]; // j
                    m3[indz].a = &idx3(p_alpha_beta_reduced_[0], 1, y * gr->blockDim[1], 0); // p_alpha_beta_reduced(0, j, beta)
                    m3[indz].lda = p_alpha_beta_reduced_->ld_;
                    m3[indz].b = &idx3(W, 0, 0, 0); // W_{j, alpha, gamma}
                    m3[indz].ldb = coef->size[1] * W.ld_;
                    m3[indz].c = &idx3(coef[0], 0, 0, 0); // coef_{beta,alpha,gamma}
                    m3[indz].ldc = coef->ld_ * coef->size[1];
                    indz++;
                }  else {
                    collocate_l0_blocked_xy(scratch,
                                            1.0,
                                            gr->blockDim[2],
                                            gr->blockDim[1],
                                            x * gr->blockDim[2],
                                            y * gr->blockDim[1],
                                            gr->block->ld_,
                                            p_alpha_beta_reduced_);
                    // it is just a scalar product
                    coef->data[0] += integrate_l0_blocked_z(scratch,
                                                            z * gr->blockDim[0],
                                                            p_alpha_beta_reduced_,
                                                            gr->block);
                }
            }
        }
    }

    if (coef->size[0] > 1) {
        batched_dgemm_simplified(m1,
                                 indz,
                                 true);
        batched_dgemm_simplified(m2,
                                 indz,
                                 true);
        batched_dgemm_simplified(m3,
                                 indz,
                                 true);

        free(m1);
        free(m2);
        free(m3);
    }

    return;
}

void grid_integrate(collocation_integration *const handler,
                    const bool use_ortho,
                    const int lp,
                    const double zetp,
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
    int pol_offset[3] = {0, 0, 0};
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
                                       handler->dh,
                                       handler->dh_inv,
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
        initialize_tensor_3(&handler->pol, 3, cmax, handler->coef.size[0]);
    } else {
        initialize_tensor_3(&handler->pol, 3, handler->coef.size[0], cmax);
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

    bool use_ortho_forced = handler->orthogonal[0] && handler->orthogonal[1] && handler->orthogonal[2];
    if (use_ortho && use_ortho_forced) {
        grid_fill_pol((lp != 0), handler->dh[0][0], roffset[2], pol_offset[2], lb_cube[2], ub_cube[2], lp, cmax, zetp, &idx3(handler->pol, perm[2], 0, 0)); /* i indice */
        grid_fill_pol((lp != 0), handler->dh[1][1], roffset[1], pol_offset[1], lb_cube[1], ub_cube[1], lp, cmax, zetp, &idx3(handler->pol, perm[1], 0, 0)); /* j indice */
        grid_fill_pol((lp != 0), handler->dh[2][2], roffset[0], pol_offset[0], lb_cube[0], ub_cube[0], lp, cmax, zetp, &idx3(handler->pol, perm[0], 0, 0)); /* k indice */
    } else {
        initialize_tensor_3(&handler->Exp, 3, max(cube_size[0], cube_size[1]), max(cube_size[1], cube_size[2]));
        handler->Exp_alloc_size = realloc_tensor(&handler->Exp);

        double dx[3];

        dx[2] = handler->dh[0][0] * handler->dh[0][0] +
            handler->dh[0][1] * handler->dh[0][1] +
            handler->dh[0][2] * handler->dh[0][2];

        dx[1] = handler->dh[1][0] * handler->dh[1][0] +
            handler->dh[1][1] * handler->dh[1][1] +
            handler->dh[1][2] * handler->dh[1][2];

        dx[0] = handler->dh[2][0] * handler->dh[2][0] +
            handler->dh[2][1] * handler->dh[2][1] +
            handler->dh[2][2] * handler->dh[2][2];

        grid_fill_pol((lp != 0), 1.0, roffset[2], pol_offset[2], lb_cube[2], ub_cube[2], lp, cmax, zetp * dx[2], &idx3(handler->pol, perm[2], 0, 0)); /* i indice */
        grid_fill_pol((lp != 0), 1.0, roffset[1], pol_offset[1], lb_cube[1], ub_cube[1], lp, cmax, zetp * dx[1], &idx3(handler->pol, perm[1], 0, 0)); /* j indice */
        grid_fill_pol((lp != 0), 1.0, roffset[0], pol_offset[0], lb_cube[0], ub_cube[0], lp, cmax, zetp * dx[0], &idx3(handler->pol, perm[0], 0, 0)); /* k indice */

        calculate_non_orthorombic_corrections_tensor(zetp,
                                                     roffset,
                                                     handler->dh,
                                                     lb_cube,
                                                     ub_cube,
                                                     handler->orthogonal,
                                                     &handler->Exp);
    }

    extract_cube(lb_cube,
                 cubecenter,
                 npts,
                 grid,
                 lb_grid,
                 &handler->cube);

    if (!use_ortho && !use_ortho_forced)
        apply_non_orthorombic_corrections(handler->orthogonal,
                                          &handler->Exp,
                                          &handler->cube);

    if (lp != 0) {
        tensor_reduction_for_collocate_integrate(handler->scratch,
                                                 // pointer to scratch memory
                                                 1.0,
                                                 0.0,
                                                 &handler->cube,
                                                 &handler->pol,
                                                 &handler->coef);
    } else {
        collocate_l0_blocked_xy(handler->scratch,
                                1.0,
                                handler->cube.size[1],
                                handler->cube.size[2],
                                0,
                                0,
                                handler->cube.ld_,
                                &handler->pol);
        // it is just a scalar product
        handler->coef.data[0] = integrate_l0_blocked_z(handler->scratch,
                                                       0,
                                                       &handler->pol,
                                                       &handler->cube);
    }

/* go from ijk -> xyz */
    if (!use_ortho && !use_ortho_forced)
        grid_transform_coef_jik_to_yxz(handler->dh, &handler->coef);
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
    /* const double rab2 = rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2]; */
    /* const double prefactor = exp(-zeta * f * rab2); */
    const int period[3] = {npts[2], npts[1], npts[0]};
    const int lb_grid_bis[3] = {lb_grid[2], lb_grid[1], lb_grid[0]};
    const bool periodic_bis[3] = {periodic[2], periodic[1], periodic[0]};

    tensor grid;
    initialize_tensor_3(&grid, ngrid[2], ngrid[1], ngrid[0]);
    grid.ld_ = ngrid[0];
    grid.data = grid_;
    double rp[3]/* , rb[3] */;
    for (int i=0; i<3; i++) {
        rp[i] = ra[i] + f * rab[i];
        /* rb[i] = ra[i] + rab[i]; */
    }

    /* initialize_basis_vectors(handler, */
    /*                          dh, */
    /*                          dh_inv); */

    /* verify_orthogonality(dh, handler->orthogonal); */

    initialize_grid(handler,
                    use_ortho,
                    true,
                    dh,
                    dh_inv,
                    ngrid,
                    grid_);

    grid_integrate(handler,
                   use_ortho,
                   lp,
                   zetp,
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
