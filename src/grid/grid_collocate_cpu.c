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
#else
#include "openblas.h"
#endif

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

#include "tensor_local.h"
#include "grid_collocate_replay.h"
#include "grid_collocate_cpu.h"
#include "grid_prepare_pab.h"
#include "grid_common.h"
#include "collocation_integration.h"
#include "utils.h"
#include "coefficients.h"
#include "non_orthorombic_corrections.h"

void collocate_l0(double *scratch,
                  const double alpha,
                  const double beta,
                  const struct tensor_ *p_alpha_beta_reduced_,
                  struct tensor_ *cube);

void tensor_reduction_for_collocate_integrate(double *scratch,
                                              /* const int *pos, */
                                              const double alpha,
                                              const double beta,
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


void collocate_l0(double *scratch,
                  const double alpha,
                  const double beta,
                  const struct tensor_ *p_alpha_beta_reduced_,
                  struct tensor_ *cube)
{
    const double *__restrict pz = &idx3(p_alpha_beta_reduced_[0], 0, 0, 0); /* k indice */
    const double *__restrict py = &idx3(p_alpha_beta_reduced_[0], 1, 0, 0); /* j indice */
    const double *__restrict px = &idx3(p_alpha_beta_reduced_[0], 2, 0, 0); /* i indice */

    if (fabs(beta) < 1e-15)
        memset(&idx3(cube[0], 0, 0, 0), 0, sizeof(double) * cube->alloc_size_);

    for (int y = 0; y < cube->size[1]; y++) {
        const double coef = alpha * py[y];
        for (int x = 0; x < cube->size[2]; x++) {
            scratch[y * cube->ld_ + x] = coef * px[x];
        }
    }

    /* cblas_dger (CblasRowMajor, */
    /*             cube->size[1], */
    /*             cube->size[2], */
    /*             idx3(co[0], 0, 0, 0), */
    /*             py, */
    /*             1, */
    /*             px, */
    /*             1, */
    /*             &idx3(cube[0], 0, 0, 0), */
    /*             cube->ld_); */

    for (int z1 = 1; z1 < cube->size[0]; z1++) {
        cblas_daxpy(cube->size[1] * cube->ld_,
                    pz[z1],
                    scratch,
                    1,
                    &idx3(cube[0], z1, 0, 0),
                    1);
    }

    /* NOTE: DO NOT PERMUTE THIS */
    if (scratch == cube->data) {
        cblas_dscal(cube->size[1] * cube->ld_, pz[0], &idx3(cube[0], 0, 0, 0), 1);
    } else {
        cblas_daxpy(cube->size[1] * cube->ld_,
                    pz[0],
                    scratch,
                    1,
                    &idx3(cube[0], 0, 0, 0),
                    1);
    }
}

void collocate_l0_blocked_xy(double *scratch,
                             const double alpha,
                             const int sizex,
                             const int sizey,
                             const int x0,
                             const int y0,
                             const int ld,
                             const struct tensor_ *p_alpha_beta_reduced_)
{
    const double *__restrict py = &idx3(p_alpha_beta_reduced_[0], 1, 0, y0); /* j indice */
    const double *__restrict px = &idx3(p_alpha_beta_reduced_[0], 2, 0, x0); /* i indice */

    for (int y = 0; y < sizey; y++) {
        const double coef = alpha * py[y];
        for (int x = 0; x < sizex; x++) {
            scratch[y * ld + x] = coef * px[x];
        }
    }
}


void collocate_l0_blocked_z(double *scratch,
                            const double alpha,
                            const double beta,
                            const int z0,
                            const struct tensor_ *p_alpha_beta_reduced_,
                            struct tensor_ *cube)
{
    const double *__restrict pz = &idx3(p_alpha_beta_reduced_[0], 0, 0, z0); /* k indice */
    if (fabs(beta) < 1e-15)
        memset(&idx3(cube[0], 0, 0, 0), 0, sizeof(double) * cube->alloc_size_);

    for (int z1 = 0; z1 < cube->size[0]; z1++) {
        double *__restrict__ dst = &idx3(cube[0], z1, 0, 0);
        for (int s = 0; s < cube->size[1] * cube->ld_; s++)
            dst[s] += pz[z1] * scratch[s];

        /* cblas_daxpy(cube->size[1] * cube->ld_, */
        /*             pz[z1], */
        /*             scratch, */
        /*             1, */
        /*             &idx3(cube[0], z1, 0, 0), */
        /*             1); */
    }
}


void collocate_l0_blocked(double *scratch,
                          const double alpha,
                          const double beta,
                          const int *const position,
                          const struct tensor_ *p_alpha_beta_reduced_,
                          struct tensor_ *cube)
{
    assert(scratch != NULL);
    assert(p_alpha_beta_reduced_);
    assert(cube != NULL);
    assert(position[0] >= 0);
    assert(position[1] >= 0);
    assert(position[2] >= 0);
    const double *__restrict pz = &idx3(p_alpha_beta_reduced_[0], 0, 0, position[0]); /* k indice */
    const double *__restrict py = &idx3(p_alpha_beta_reduced_[0], 1, 0, position[1]); /* j indice */
    const double *__restrict px = &idx3(p_alpha_beta_reduced_[0], 2, 0, position[2]); /* i indice */

    /* if (fabs(beta) < 1e-15) */
    /*     memset(&idx3(cube[0], 0, 0, 0), 0, sizeof(double) * cube->alloc_size_); */

    for (int y = 0; y < cube->size[1]; y++) {
        const double coef = alpha * py[y];
        for (int x = 0; x < cube->size[2]; x++) {
            scratch[y * cube->ld_ + x] = coef * px[x];
        }
    }

    if (fabs(beta) <= 1e-15) {
        for (int z1 = 0; z1 < cube->size[0]; z1++) {
            double *__restrict__ dst = &idx3(cube[0], z1, 0, 0);
#pragma GCC ivdep
            for (int s = 0; s < cube->size[1] * cube->ld_; s++)
                dst[s] = pz[z1] * scratch[s];
        }
    } else {
        for (int z1 = 0; z1 < cube->size[0]; z1++) {
            double *__restrict__ dst = &idx3(cube[0], z1, 0, 0);
#pragma GCC ivdep
            for (int s = 0; s < cube->size[1] * cube->ld_; s++)
                dst[s] = beta * dst[s] + pz[z1] * scratch[s];
        }
    }
}

/* compute the following operation (variant) it is a tensor contraction

   V_{kji} = \sum_{\alpha\beta\gamma} C_{\alpha\gamma\beta} T_{2,\alpha,i} T_{1,\beta,j} T_{0,\gamma,k}

*/
void tensor_reduction_for_collocate_integrate_blocked(double *scratch,
                                                      const double alpha,
                                                      const double beta,
                                                      const int *lower_block,
                                                      const int *upper_block,
                                                      const struct tensor_ *co,
                                                      const struct tensor_ *p_alpha_beta_reduced_,
                                                      struct tensor_ *gr)
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


    if (!gr->blocked_decomposition)
    {
        printf("Erro : You are using the wrong version of the tensor reduction.\n");
        printf("You should use tensor_reduction_for_collocate_integrate instead\n");
        printf("or initialize the gr tensor as blocked\\n");
        abort();
    }

//    co->size[0] > 1
    if (co->size[0] > 1) {


// build a list of matrix-matrix multiplications
        tensor T, W;
        dgemm_params m1, m2, *m3;

        //m1 = (dgemm_params *)malloc(sizeof(dgemm_params) * (upper_block[1] - lower_block[1]));
        m3 = (dgemm_params *)malloc(sizeof(dgemm_params) *
                                    (upper_block[0] - lower_block[0]));

        memset(&m1, 0, sizeof(dgemm_params));
        memset(&m2, 0, sizeof(dgemm_params));
        memset(m3, 0, sizeof(dgemm_params) * (upper_block[0] - lower_block[0]));

        initialize_tensor_3(&T,
                            co->size[0] /* alpha */,
                            co->size[1] /* gamma */,
                            gr->blockDim[1]/* j */);

        initialize_tensor_3(&W,
                            co->size[1] /* gamma */ ,
                            gr->blockDim[1] /* j */,
                            gr->blockDim[2] /* i */);

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



        for (int y = 0; y < (upper_block[1] - lower_block[1]); y++) {
            int y1 = (y + lower_block[1]) % gr->size[1];
            double *__restrict const py = &idx3(p_alpha_beta_reduced_[0], 1, 0, y * gr->blockDim[1]); /* j indice */

            m1.op1 = 'N';
            m1.op2 = 'N';
            m1.alpha = alpha;
            m1.beta = 0.0;
            m1.m = co->size[0] * co->size[1]; /* alpha gamma */
            m1.n = gr->blockDim[1]; /* j */
            m1.k = co->size[2]; /* beta */
            m1.a = co->data; // Coef_{alpha,gamma,beta} Coef_xzy
            m1.lda = co->ld_;
            m1.b = py; // Y_{beta, j} = p_alpha_beta_reduced(1, beta, j)
            m1.ldb = p_alpha_beta_reduced_->ld_;
            m1.c = T.data; // T_{\alpha, \gamma, j} = T(alpha, gamma, j)
            m1.ldc = T.ld_;

            dgemm_simplified(&m1, true);

            /*
             * the next step is a reduction along the alpha index.
             *
             * We compute then
             *
             * W_{gamma, j, i} = sum_{\alpha} T_{\gamma, j, alpha} X_{\alpha, i}
             *
             * which means we need to transpose T_{\alpha, \gamma, j} to get
             * T_{\gamma, j, \alpha}. Fortunately we can do it while doing the
             * matrix - matrix multiplication
             */

            for (int x = 0; x < (upper_block[2] - lower_block[2]); x++) {
                int x1 = (x + lower_block[2]) % gr->size[2];

                m2.op1='T';
                m2.op2='N';
                m2.alpha = 1.0;
                m2.beta = 0.0;
                m2.m = gr->blockDim[1] * co->size[1]; // (\gamma j) direction
                m2.n = gr->blockDim[2]; // i
                m2.k = co->size[0]; // alpha
                m2.a = &idx3(T, 0, 0, 0); // T_{\alpha, \gamma, j}
                m2.lda = T.ld_ * co->size[1];
                m2.b = &idx3(p_alpha_beta_reduced_[0], 2, 0, x * gr->blockDim[2]); // X_{alpha, i}  = p_alpha_beta_reduced(0, alpha, i)
                m2.ldb = p_alpha_beta_reduced_->ld_;
                m2.c = &idx3(W, 0, 0, 0); // W_{\gamma, j, i}
                m2.ldc = W.ld_;

                dgemm_simplified(&m2, true);

                /* the final step is again a reduction along the gamma indice. It can
                 * again be done with one dgemm. The operation is simply
                 *
                 * Cube_{k, j, i} = \sum_{alpha} Z_{k, \gamma} W_{\gamma, j, i}
                 *
                 * which means we need to transpose Z_{\gamma, k}.
                 */
                int indz = 0;
                for (int z = 0; z < (upper_block[0] - lower_block[0]); z++) {
                    int z1 = (z + lower_block[0]) % gr->size[0];
                    m3[indz].op1 = 'T';
                    m3[indz].op2 = 'N';
                    m3[indz].alpha = alpha;
                    m3[indz].beta = beta;
                    m3[indz].m = gr->blockDim[0]; // Z_{k \gamma}
                    m3[indz].n = gr->blockDim[1] * gr->blockDim[2]; // (ji) direction
                    m3[indz].k = co->size[1]; // \gamma
                    m3[indz].a = &idx3(p_alpha_beta_reduced_[0], 0, 0, z * gr->blockDim[0]); // p_alpha_beta_reduced(0, gamma, i)
                    m3[indz].lda = p_alpha_beta_reduced_->ld_;
                    m3[indz].b = &idx3(W, 0, 0, 0); // W_{\gamma, j, i}
                    m3[indz].ldb = gr->blockDim[1] * W.ld_;
                    m3[indz].c = &idx4(gr[0], z1, y1, x1, 0); // cube_{kji}
                    m3[indz].ldc = gr->block->ld_ * gr->block->size[1];
                    indz++;
                }

                batched_dgemm_simplified(m3,
                                         (upper_block[0] - lower_block[0]),
                                         true);
            }
        }

        free(m3);
        return;
    }

    assert(gr->block->size[0] == gr->blockDim[0]);
    assert(gr->block->size[1] == gr->blockDim[1]);
    assert(gr->block->size[2] == gr->blockDim[2]);

    for (int y = 0; y < (upper_block[1] - lower_block[1]); y++) {
        int y1 = (y + lower_block[1]) % gr->size[1];
        for (int x = 0; x < (upper_block[2] - lower_block[2]); x++) {
            int x1 = (x + lower_block[2]) % gr->size[2];
            collocate_l0_blocked_xy(scratch,
                                    co->data[0] * alpha,
                                    gr->blockDim[2],
                                    gr->blockDim[1],
                                    x * gr->blockDim[2],
                                    y * gr->blockDim[1],
                                    gr->block->ld_,
                                    p_alpha_beta_reduced_);
            for (int z = 0; z < (upper_block[0] - lower_block[0]); z++) {
                int z1 = (z + lower_block[0]) % gr->size[0];

                /* const int position[3] = {z * gr->blockDim[0], y * gr->blockDim[1], x * gr->blockDim[2]}; */
                gr->block->data = &idx4(gr[0], z1, y1, x1, 0);
                collocate_l0_blocked_z(scratch,
                                       co->data[0] * alpha,
                                       beta,
                                       z * gr->blockDim[0],
                                       p_alpha_beta_reduced_,
                                       gr->block);

                /* collocate_l0_blocked(scratch, */
                /*                      co->data[0] * alpha, */
                /*                      beta, */
                /*                      position, */
                /*                      p_alpha_beta_reduced_, */
                /*                      gr->block); */
            }
        }
    }

    return;
}



/* compute the following operation (variant) it is a tensor contraction

   V_{kji} = \sum_{\alpha\beta\gamma} C_{\alpha\gamma\beta} T_{2,\alpha,i} T_{1,\beta,j} T_{0,\gamma,k}

*/
void tensor_reduction_for_collocate_integrate(double *scratch,
//                                              const int *position,
                                              const double alpha,
                                              const double beta,
                                              const struct tensor_ *co,
                                              const struct tensor_ *p_alpha_beta_reduced_,
                                              struct tensor_ *cube)
{

    if (cube->size[0] == 1) {
        /* it is very specific to integrate because we might end up with a single
         * element after the tensor product. In that case, I call the specific case
         * with l = 0 and then do a scalar product between the two. We can not get
         * one of the dimensions at 1 and all the other above. It is physics non
         * sense */
        tensor cube_tmp;
        initialize_tensor_3(&cube_tmp, co->size[0], co->size[1], co->size[2]);
        cube_tmp.data = scratch;
        collocate_l0(scratch,
                     alpha,
                     0.0,
                     p_alpha_beta_reduced_,
                     &cube_tmp);
        cube->data[0] = cube->data[0] * beta + cblas_ddot(cube_tmp.alloc_size_, cube_tmp.data, 1, co->data, 1);
        return;
    }

    if (co->size[0] > 1) {
        dgemm_params m1, m2, m3;

        memset(&m1, 0, sizeof(dgemm_params));
        memset(&m2, 0, sizeof(dgemm_params));
        memset(&m3, 0, sizeof(dgemm_params));
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

        m1.op1 = 'N';
        m1.op2 = 'N';
        m1.alpha = alpha;
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
         * W_{gamma, j, i} = sum_{\alpha} T_{\gamma, j, alpha} X_{\alpha, i}
         *
         * which means we need to transpose T_{\alpha, \gamma, j} to get
         * T_{\gamma, j, \alpha}. Fortunately we can do it while doing the
         * matrix - matrix multiplication
         */

        m2.op1='T';
        m2.op2='N';
        m2.alpha = 1.0;
        m2.beta = 0.0;
        m2.m = cube->size[1] * co->size[1]; // (\gamma j) direction
        m2.n = cube->size[2]; // i
        m2.k = co->size[0]; // alpha
        m2.a = T.data; // T_{\alpha, \gamma, j}
        m2.lda = T.ld_ * co->size[1];
        m2.b = px; // X_{alpha, i}  = p_alpha_beta_reduced(0, alpha, i)
        m2.ldb = p_alpha_beta_reduced_->ld_;
        m2.c = W.data; // W_{\gamma, j, i}
        m2.ldc = W.ld_;

        /* the final step is again a reduction along the gamma indice. It can
         * again be done with one dgemm. The operation is simply
         *
         * Cube_{k, j, i} = \sum_{alpha} Z_{k, \gamma} W_{\gamma, j, i}
         *
         * which means we need to transpose Z_{\gamma, k}.
         */

        m3.op1 = 'T';
        m3.op2 = 'N';
        m3.alpha = alpha;
        m3.beta = beta;
        m3.m = cube->size[0]; // Z_{k \gamma}
        m3.n = cube->size[1] * cube->size[2]; // (ji) direction
        m3.k = co->size[1]; // \gamma
        m3.a = pz; // p_alpha_beta_reduced(0, gamma, i)
        m3.lda = p_alpha_beta_reduced_->ld_;
        m3.b = &idx3(W, 0, 0, 0); // W_{\gamma, j, i}
        m3.ldb = W.size[1] * W.ld_;
        m3.c = &idx3(cube[0], 0, 0, 0); // cube_{kji}
        m3.ldc = cube->ld_ * cube->size[1];

        dgemm_simplified(&m1, true);
        dgemm_simplified(&m2, true);
        dgemm_simplified(&m3, true);

        return;
    }

    int position[3] = {0, 0, 0};
    collocate_l0_blocked(scratch,
                         co->data[0] * alpha,
                         beta,
                         position,
                         p_alpha_beta_reduced_,
                         cube);
    return;
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
                                    /* LIBXSMM_PRAGMA_SIMD */
#pragma unroll(8)
                                    for (int x = 0; x < grid->size[2]; x++)
                                        dst[x] += src[shift + x];

                                    shift = offset_x + (l + 1)  * period[2];
                                }
                            }
//                            LIBXSMM_PRAGMA_SIMD
#pragma unroll(8)
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
void grid_collocate(collocation_integration *const handler,
                    const bool use_ortho,
                    const double zetp,
                    const double rp[3],
                    const int npts[3],
                    const int lb_grid[3],
                    const bool periodic[3],
                    const double radius)
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
    int pivot[3] = {0, 0, 0};
    int pol_offset[3] = {0, 0, 0};
    int lower_block_corner[3] = {0, 0, 0};
    int upper_block_corner[3] = {0, 0, 0};
    bool fold[3] = {false, false, false};
    double roffset[3];
    double disr_radius;
    /* cube : grid containing pointlike product between polynomials
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

    compute_block_boundaries(handler->blockDim,
                             lb_grid,
                             handler->grid.size,
                             handler->blocked_grid.size,
                             npts,
                             cubecenter,
                             cube_size,
                             lb_cube,
                             lower_block_corner,
                             upper_block_corner,
                             pol_offset,
                             fold);

    cmax = max(max(max((upper_block_corner[0] - lower_block_corner[0]) * handler->blockDim[0],
                   (upper_block_corner[1] - lower_block_corner[1]) * handler->blockDim[1]) ,
               (upper_block_corner[2] - lower_block_corner[2])  * handler->blockDim[2]), cmax);

    /* initialize the multidimensional array containing the polynomials */
    initialize_tensor_3(&handler->pol, 3, handler->coef.size[0], cmax);
    handler->pol_alloc_size = realloc_tensor(&handler->pol);

    /* compute the polynomials */

    // WARNING : do not reverse the order in pol otherwise you will have to
    // reverse the order in collocate_dgemm as well.

    if (use_ortho) {
        grid_fill_pol(false, handler->dh[0][0], roffset[2], pol_offset[2], lb_cube[2], ub_cube[2], handler->coef.size[2] - 1, cmax, zetp, &idx3(handler->pol, 2, 0, 0)); /* i indice */
        grid_fill_pol(false, handler->dh[1][1], roffset[1], pol_offset[1], lb_cube[1], ub_cube[1], handler->coef.size[1] - 1, cmax, zetp, &idx3(handler->pol, 1, 0, 0)); /* j indice */
        grid_fill_pol(false, handler->dh[2][2], roffset[0], pol_offset[0], lb_cube[0], ub_cube[0], handler->coef.size[0] - 1, cmax, zetp, &idx3(handler->pol, 0, 0, 0)); /* k indice */

        /* for (int i = 0; i < 3; i++) { */
        /*     /\* I can fold the polynomials before doing the collocation. it saves both compute and bandwidth  *\/ */
        /*     lb_cube[i] -= pol_offset[i]; */
        /*     fold[i] = fold_polynomial(handler->scratch, */
        /*                               &handler->pol, */
        /*                               i, */
        /*                               cubecenter[i], */
        /*                               cube_size[i], */
        /*                               lb_cube[i], */
        /*                               lb_grid[i], */
        /*                               handler->grid.size[i], */
        /*                               npts[i], */
        /*                               &pivot[i]); */

        /*     if (fold[i]) { */
        /*         /\* folding also mean that the origin of the collocated result is */
        /*          * now the the origin of the grid and the cube center in that */
        /*          * direction is meaningless *\/ */
        /*         cube_size[i] = handler->grid.size[i]; */
        /*         lb_cube[i] = 0; */
        /*         cubecenter[i] = 0; */
        /*     } */
        /* } */
    } else {
        pol_offset[0] = 0;
        pol_offset[1] = 0;
        pol_offset[2] = 0;

        initialize_tensor_3(&handler->Exp, 3, max(cube_size[0], cube_size[1]), max(cube_size[1], cube_size[2]));
        handler->Exp_alloc_size = realloc_tensor(&handler->Exp);

        double dx[3];
        dx[2] = handler->dh[0][0] * handler->dh[0][0] + handler->dh[0][1] * handler->dh[0][1] + handler->dh[0][2] * handler->dh[0][2];
        dx[1] = handler->dh[1][0] * handler->dh[1][0] + handler->dh[1][1] * handler->dh[1][1] + handler->dh[1][2] * handler->dh[1][2];
        dx[0] = handler->dh[2][0] * handler->dh[2][0] + handler->dh[2][1] * handler->dh[2][1] + handler->dh[2][2] * handler->dh[2][2];

        grid_fill_pol(false, 1.0, roffset[0], pol_offset[0], lb_cube[0], ub_cube[0], handler->coef.size[0] - 1, cmax, zetp * dx[0], &idx3(handler->pol, 0, 0, 0)); /* k indice */
        grid_fill_pol(false, 1.0, roffset[1], pol_offset[1], lb_cube[1], ub_cube[1], handler->coef.size[1] - 1, cmax, zetp * dx[1], &idx3(handler->pol, 1, 0, 0)); /* j indice */
        grid_fill_pol(false, 1.0, roffset[2], pol_offset[2], lb_cube[2], ub_cube[2], handler->coef.size[2] - 1, cmax, zetp * dx[2], &idx3(handler->pol, 2, 0, 0)); /* i indice */

        calculate_non_orthorombic_corrections_tensor(zetp,
                                                     roffset,
                                                     handler->dh,
                                                     lb_cube,
                                                     ub_cube,
                                                     handler->orthogonal,
                                                     &handler->Exp);

        /* Use a slightly modified version of Ole code */
        grid_transform_coef_xzy_to_ikj(handler->dh, &handler->coef);
    }


    /* allocate memory for the polynomial and the cube */

    initialize_tensor_3(&handler->cube,
                        cube_size[0],
                        cube_size[1],
                        cube_size[2]);

    handler->cube_alloc_size = realloc_tensor(&handler->cube);

    initialize_W_and_T(handler, &handler->cube, &handler->coef);


    if (use_ortho) {
        tensor_reduction_for_collocate_integrate_blocked(handler->scratch,
                                                         // pointer to scratch memory
                                                         1.0,
                                                         1.0,
                                                         lower_block_corner,
                                                         upper_block_corner,
                                                         &handler->coef,
                                                         &handler->pol,
                                                         &handler->blocked_grid);
    } else {
        tensor_reduction_for_collocate_integrate(handler->scratch, // pointer to scratch memory
                                                 1.0,
                                                 0.0,
                                                 &handler->coef,
                                                 &handler->pol,
                                                 &handler->cube);

        apply_non_orthorombic_corrections(handler->orthogonal,
                                          &handler->Exp,
                                          &handler->cube);

        apply_mapping_cubic(lb_cube, cubecenter, npts, &handler->cube, lb_grid, &handler->grid);
    }

    /* I apply the mapping if I can not do a collocate directly in the grid */
    //if (!fold[0] || !fold[1] || !fold[2])
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

    assert(handle != NULL);
    collocation_integration *handler = (collocation_integration *)handle;

// Uncomment this to dump all tasks to file.
// #define __GRID_DUMP_TASKS
    tensor grid;
    int offset[2] = {o1, o2};
    int pab_size[2] = {n2, n1};

    int lmax[2] = {la_max, lb_max};
    int lmin[2] = {la_min, lb_min};

    initialize_grid(handler,
                    use_ortho,
                    false,
                    dh,
                    dh_inv,
                    ngrid,
                    grid_);

    const double zetp = zeta + zetb;
    const double f = zetb / zetp;
    const double rab2 = rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2];
    const double prefactor = rscale * exp(-zeta * f * rab2);
    const int period[3] = {npts[2], npts[1], npts[0]};
    const int lb_grid_bis[3] = {lb_grid[2], lb_grid[1], lb_grid[0]};
    const bool periodic_bis[3] = {periodic[2], periodic[1], periodic[0]};

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

    grid_prepare_pab(func, offset[0], offset[1], lmax,
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

    initialize_tensor_4(&handler->alpha, 3, lmax_prep[1] + 1, lmax_prep[0] + 1, lmax_prep[0] + lmax_prep[1] + 1);
    handler->alpha_alloc_size = realloc_tensor(&handler->alpha);

    const int lp = lmax_prep[0] + lmax_prep[1];

    initialize_tensor_3(&(handler->coef), lp + 1, lp + 1, lp + 1);
    handler->coef_alloc_size = realloc_tensor(&handler->coef);

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
    grid_prepare_coef(lmin_prep,
                      lmax_prep,
                      lp,
                      prefactor,
                      &handler->alpha,
                      pab_prep,
                      &handler->coef);


    grid_collocate(handler,
                   use_ortho,
                   zetp,
                   rp,
                   period,
                   lb_grid_bis,
                   periodic_bis,
                   radius);
}
