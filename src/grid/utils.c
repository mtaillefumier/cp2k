#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "utils.h"
#include "tensor_local.h"

void compute_block_dimensions(const int *const grid_size, int *const blockDim)
{
    int block_size_test[8] = {2, 3, 4, 5, 6, 7, 8, 10};
    bool block_divided[8];
    for (int d = 0; d < 3; d++) {
        for (int s = 0; s < 8; s++)
            block_divided[s] = (grid_size[d] % block_size_test[s] == 0);

        if (block_divided[7])
        {
            blockDim[d] = 10;
            continue;
        }

        if (block_divided[6])
        {
            blockDim[d] = 8;
            continue;
        }

        if (block_divided[5]) {
            blockDim[d] = 7;
            continue;
        }

        if (block_divided[4]) {
            blockDim[d] = 6;
            continue;
        }

        if (block_divided[3]) {
            blockDim[d] = 5;
            continue;
        }

        if (block_divided[2]) {
            blockDim[d] = 4;
            continue;
        }

        if (block_divided[1]) {
            blockDim[d] = 3;
            continue;
        }

        if (block_divided[0]) {
            blockDim[d] = 2;
            continue;
        }
    }
}

/* compute the functions (x - x_i)^l exp (-eta (x - x_i)^2) for l = 0..lp using
 * a recursive relation to avoid computing the exponential on each grid point. I
 * think it is not really necessary anymore since it is *not* the dominating
 * contribution to computation of collocate and integrate */

void grid_fill_pol(const bool transpose,
                   const double dr,
                   const double roffset,
                   const int pol_offset,
                   const int xmin,
                   const int xmax,
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
        initialize_tensor_2(&pol, cmax, lp + 1);
        pol.data = pol_;
        /* It is original Ole code. I need to transpose the polynomials for the
         * integration routine and Ole code already does it. */
        for (int ig = 0; ig >= xmin; ig--) {
            const double rpg = ig * dr - roffset;
            t_exp_min_1 *= t_exp_min_2 * t_exp_1;
            t_exp_min_2 *= t_exp_2;
            double pg = t_exp_min_1;
            for (int icoef = 0; icoef <= lp; icoef++) {
                idx2(pol, pol_offset + ig - xmin, icoef) = pg;
                pg *= rpg;
            }
        }

        double t_exp_plus_1 = exp(-zetp * roffset * roffset);
        double t_exp_plus_2 = exp(2 * zetp * roffset * dr);
        for (int ig = 1; ig <= xmax; ig++) {
            const double rpg = ig * dr - roffset;
            t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
            t_exp_plus_2 *= t_exp_2;
            double pg = t_exp_plus_1;
            for (int icoef = 0; icoef <= lp; icoef++) {
                idx2(pol, pol_offset + ig - xmin, icoef) = pg;
                pg *= rpg;
            }
        }

    } else {
        initialize_tensor_2(&pol, lp + 1, cmax);
        pol.data = pol_;
        /* memset(pol.data, 0, sizeof(double) * pol.alloc_size_); */
        /*
         *   compute the values of all (x-xp)**lp*exp(..)
         *
         *  still requires the old trick:
         *  new trick to avoid to many exps (reuse the result from the previous gridpoint):
         *  exp( -a*(x+d)**2)=exp(-a*x**2)*exp(-2*a*x*d)*exp(-a*d**2)
         *  exp(-2*a*(x+d)*d)=exp(-2*a*x*d)*exp(-2*a*d**2)
         */

        /* compute the exponential recursively and store the polynomial prefactors as well */
        for (int ig = 0; ig >= xmin; ig--) {
            const double rpg = ig * dr - roffset;
            t_exp_min_1 *= t_exp_min_2 * t_exp_1;
            t_exp_min_2 *= t_exp_2;
            double pg = t_exp_min_1;
            idx2(pol, 0, pol_offset + ig - xmin) = pg;
            if (lp > 0)
                idx2(pol, 1, pol_offset + ig - xmin) = rpg;
        }

        for (int ig = 1; ig <= xmax; ig++) {
            const double rpg = ig * dr - roffset;
            t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
            t_exp_plus_2 *= t_exp_2;
            double pg = t_exp_plus_1;
            idx2(pol, 0, pol_offset + ig - xmin) = pg;
            if (lp > 0)
                idx2(pol, 1, pol_offset + ig - xmin) = rpg;
        }

        /* compute the remaining powers using previously computed stuff */
        if (lp >= 2) {
            double *__restrict__ poly = &idx2(pol, 1, 0);
            double *__restrict__ src1 = &idx2(pol, 0, 0);
            double *__restrict__ dst = &idx2(pol, 2, 0);
//#pragma omp simd
#pragma GCC ivdep
            for (int ig = 0; ig < (xmax - xmin + 1 + pol_offset); ig++)
                dst[ig] = src1[ig] * poly[ig] * poly[ig];
        }

        for (int icoef = 3; icoef <= lp; icoef++) {
            const double *__restrict__ poly = &idx2(pol, 1, 0);
            const double *__restrict__ src1 = &idx2(pol, icoef - 1, 0);
            double *__restrict__ dst = &idx2(pol, icoef, 0);
//#pragma omp simd
#pragma GCC ivdep
            for (int ig = 0; ig <  (xmax - xmin + 1 + pol_offset); ig++) {
                dst[ig] = src1[ig] * poly[ig];
            }
        }

        //
        if (lp > 0) {
            double *__restrict__ dst = &idx2(pol, 1, 0);
            const double *__restrict__ src = &idx2(pol, 0, 0);
#pragma GCC ivdep
            for (int ig = 0; ig <  (xmax - xmin + 1 + pol_offset); ig++) {
                dst[ig] *= src[ig];
            }
        }
    }
}


bool fold_polynomial(double *scratch,
                     tensor *pol,
                     const int axis,
                     const int center,
                     const int cube_size,
                     const int lb_cube,
                     const int lb_grid,
                     const int grid_size,
                     const int period,
                     int *const pivot)
{
    const int position = (lb_grid + center + lb_cube + 32 * period) % period;
    const int offset_x = min(grid_size - position, cube_size);
    const int loop_number_x = (cube_size - offset_x) / period;
    const int reminder_x = min(grid_size, cube_size - offset_x - loop_number_x * period);
    *pivot = 0;
    if (grid_size < cube_size) {
        for (int l = 0; l < pol->size[1]; l++) {
            memset(scratch, 0, sizeof(double) * grid_size);
            const double *__restrict src = &idx3(pol[0], axis, l, 0);
            // the tail of the queue.
            LIBXSMM_PRAGMA_SIMD
                for (int x = 0; x < offset_x; x++)
                    scratch[x + position] = src[x];

            int shift = offset_x;

            if (loop_number_x) {
                for (int li = 0; li < loop_number_x; li++) {
                    LIBXSMM_PRAGMA_SIMD
                        for (int x = 0; x < grid_size; x++)
                        scratch[x] += src[shift + x];

                    shift = offset_x + (li + 1)  * period;
                }
            }
            LIBXSMM_PRAGMA_SIMD
                for (int x = 0; x < reminder_x; x++)
                    scratch[x] += src[shift + x];
            memcpy(&idx3(pol[0], axis, l, 0), scratch, sizeof(double) * grid_size);
        }
        return true;
    }
    return false;
}

void decompose_grid_to_blocked_grid(const tensor *gr, struct tensor_ *block_grid)
{
    if ((gr == NULL) || (block_grid == NULL)) {
        abort();
    }

    tensor tmp;
    initialize_tensor_3(&tmp,
                        block_grid->blockDim[0],
                        block_grid->blockDim[1],
                        block_grid->blockDim[2]);

    int lower_corner[3], upper_corner[3];

    for (int z = 0; z < block_grid->size[0]; z++) {
        lower_corner[0] = z * block_grid->blockDim[0];
        upper_corner[0] = lower_corner[0] + min(gr->size[0] - lower_corner[0], block_grid->blockDim[0]);
        for (int y = 0; y < block_grid->size[1]; y++) {
            lower_corner[1] = y * block_grid->blockDim[1];
            upper_corner[1] = lower_corner[1] + min(gr->size[1] - lower_corner[1], block_grid->blockDim[1]);
            for (int x = 0; x < block_grid->size[2]; x++) {
                tmp.data = &idx4(block_grid[0], z, y, x, 0);
                lower_corner[2] = x * block_grid->blockDim[2];
                upper_corner[2] = lower_corner[2] + min(gr->size[2] - lower_corner[2], block_grid->blockDim[2]);

                extract_sub_grid(lower_corner,
                                 upper_corner,
                                 NULL,
                                 gr, // original grid
                                 &tmp);
            }
        }
    }
}

/* recompose the natural grid from the block decomposed grid. Result is copied to the grid gr */

void recompose_grid_from_blocked_grid(const struct tensor_ *block_grid, tensor *gr)
{
    tensor tmp;
    initialize_tensor_3(&tmp,
                        block_grid->blockDim[0],
                        block_grid->blockDim[1],
                        block_grid->blockDim[2]);
    int lower_corner[3], upper_corner[3];
    for (int z = 0; z < block_grid->size[0]; z++) {
        lower_corner[0] = z * block_grid->blockDim[0];
        upper_corner[0] = lower_corner[0] + min(gr->size[0] - lower_corner[0], block_grid->blockDim[0]);
        for (int y = 0; y < block_grid->size[1]; y++) {
            lower_corner[1] = y * block_grid->blockDim[1];
            upper_corner[1] = lower_corner[1] + min(gr->size[1] - lower_corner[1], block_grid->blockDim[1]);
            for (int x = 0; x < block_grid->size[2]; x++) {
                tmp.data = &idx4(block_grid[0], z, y, x, 0);
                lower_corner[2] = x * block_grid->blockDim[2];
                upper_corner[2] = lower_corner[2] + min(gr->size[2] - lower_corner[2], block_grid->blockDim[2]);

                const int sizex = upper_corner[2] - lower_corner[2];
                const int sizey = upper_corner[1] - lower_corner[1];
                const int sizez = upper_corner[0] - lower_corner[0];

                for (int z = 0; z < sizez; z++) {
                    for (int y = 0; y < sizey; y++) {
                        double *__restrict__ dst = &idx3(gr[0], lower_corner[0] + z, lower_corner[1] + y, lower_corner[2]);
                        double *__restrict__ src = &idx3(tmp, z, y, 0);
                        for (int x = 0; x < sizex; x++) {
                            dst[x] = src[x];
                        }
                    }
                }
            }
        }
    }
}

/* recompose the natural grid from the block decomposed grid and add the result
 * to the grid gr */
void add_blocked_tensor_to_tensor(const struct tensor_ *block_grid, tensor *gr)
{
    tensor tmp;
    initialize_tensor_3(&tmp, block_grid->blockDim[0], block_grid->blockDim[1], block_grid->blockDim[2]);
    int lower_corner[3], upper_corner[3];

    for (int z = 0; z < block_grid->size[0]; z++) {
        lower_corner[0] = z * block_grid->blockDim[0];
        upper_corner[0] = lower_corner[0] + min(gr->size[0] - lower_corner[0], block_grid->blockDim[0]);
        for (int y = 0; y < block_grid->size[1]; y++) {
            lower_corner[1] = y * block_grid->blockDim[1];
            upper_corner[1] = lower_corner[1] + min(gr->size[1] - lower_corner[1], block_grid->blockDim[1]);
            for (int x = 0; x < block_grid->size[2]; x++) {
                tmp.data = &idx4(block_grid[0], z, y, x, 0);
                lower_corner[2] = x * block_grid->blockDim[2];
                upper_corner[2] = lower_corner[2] + min(gr->size[2] - lower_corner[2], block_grid->blockDim[2]);

                const int sizex = upper_corner[2] - lower_corner[2];
                const int sizey = upper_corner[1] - lower_corner[1];
                const int sizez = upper_corner[0] - lower_corner[0];

                for (int z1 = 0; z1 < sizez; z1++) {
                    for (int y1 = 0; y1 < sizey; y1++) {
                        double *__restrict__ dst = &idx3(gr[0], lower_corner[0] + z1, lower_corner[1] + y1, lower_corner[2]);
                        double *__restrict__ src = &idx3(tmp, z1, y1, 0);
                        for (int x1 = 0; x1 < sizex; x1++) {
                            dst[x1] += src[x1];
                        }
                    }
                }
            }
        }
    }
}

/* recompose the natural grid from the block decomposed grid and add the result
 * to the grid gr. The blocked tensor coordinates are in the yxz format */
void add_transpose_blocked_tensor_to_tensor(const struct tensor_ *block_grid, tensor *gr)
{
    tensor tmp;
    initialize_tensor_3(&tmp, block_grid->blockDim[0], block_grid->blockDim[1], block_grid->blockDim[2]);
    int lower_corner[3], upper_corner[3];

    for (int y = 0; y < block_grid->size[1]; y++) {
        lower_corner[1] = y * block_grid->blockDim[1];
        upper_corner[1] = lower_corner[1] + min(gr->size[1] - lower_corner[1], block_grid->blockDim[1]);
        for (int x = 0; x < block_grid->size[2]; x++) {
            lower_corner[2] = x * block_grid->blockDim[2];
            upper_corner[2] = lower_corner[2] + min(gr->size[2] - lower_corner[2], block_grid->blockDim[2]);
            for (int z = 0; z < block_grid->size[0]; z++) {
                lower_corner[0] = z * block_grid->blockDim[0];
                upper_corner[0] = lower_corner[0] + min(gr->size[0] - lower_corner[0], block_grid->blockDim[0]);
                tmp.data = &idx4(block_grid[0], y, x, z, 0);

                const int sizex = upper_corner[2] - lower_corner[2];
                const int sizey = upper_corner[1] - lower_corner[1];
                const int sizez = upper_corner[0] - lower_corner[0];

                for (int z1 = 0; z1 < sizez; z1++) {
                    for (int y1 = 0; y1 < sizey; y1++) {
                        double *__restrict__ dst = &idx3(gr[0], lower_corner[0] + z1, lower_corner[1] + y1, lower_corner[2]);
                        double *__restrict__ src = &idx3(tmp, z1, y1, 0);
                        for (int x1 = 0; x1 < sizex; x1++) {
                            dst[x1] += src[x1];
                        }
                    }
                }
            }
        }
    }
}


/* recompose the natural grid from the block decomposed grid and add the result
 * to the grid gr */
void compare_blocked_tensor_to_tensor(const struct tensor_ *block_grid, tensor *gr)
{
    tensor tmp;
    initialize_tensor_3(&tmp, block_grid->blockDim[0], block_grid->blockDim[1], block_grid->blockDim[2]);
    int lower_corner[3], upper_corner[3];

    for (int z = 0; z < block_grid->size[0]; z++) {
        lower_corner[0] = z * block_grid->blockDim[0];
        upper_corner[0] = lower_corner[0] + min(gr->size[0] - lower_corner[0], block_grid->blockDim[0]);
        for (int y = 0; y < block_grid->size[1]; y++) {
            lower_corner[1] = y * block_grid->blockDim[1];
            upper_corner[1] = lower_corner[1] + min(gr->size[1] - lower_corner[1], block_grid->blockDim[1]);
            for (int x = 0; x < block_grid->size[2]; x++) {
                tmp.data = &idx4(block_grid[0], z, y, x, 0);
                lower_corner[2] = x * block_grid->blockDim[2];
                upper_corner[2] = lower_corner[2] + min(gr->size[2] - lower_corner[2], block_grid->blockDim[2]);

                const int sizex = upper_corner[2] - lower_corner[2];
                const int sizey = upper_corner[1] - lower_corner[1];
                const int sizez = upper_corner[0] - lower_corner[0];

                for (int z1 = 0; z1 < sizez; z1++) {
                    for (int y1 = 0; y1 < sizey; y1++) {
                        double *__restrict__ dst = &idx3(gr[0], lower_corner[0] + z1, lower_corner[1] + y1, lower_corner[2]);
                        double *__restrict__ src = &idx3(tmp, z1, y1, 0);
                        for (int x1 = 0; x1 < sizex; x1++) {
                            if (fabs(dst[x1] - src[x1]) > 1e-14) {
                                printf("%.15lf %.15lf\n", src[x1], dst[x1]);
                                abort();
                            }
                        }
                    }
                }
            }
        }
    }
}


/* initialize a tensor structure for a tensor of dimension dim <= 4 */

void initialize_tensor_blocked(struct tensor_ *a, const int dim, const int *const sizes, const int *const blockDim)
{
    assert(a != NULL);

    a->block =  (void *)malloc(sizeof(struct tensor_));
    memset(a->block, 0, sizeof(struct tensor_));

    switch(dim) {
    case 4:
        initialize_tensor_4((struct tensor_ *)a->block, blockDim[0], blockDim[1], blockDim[2], blockDim[3]);
        break;
    case 3:
        initialize_tensor_3((struct tensor_ *)a->block, blockDim[0], blockDim[1], blockDim[2]);
        break;
    case 2:
        initialize_tensor_2((struct tensor_ *)a->block, blockDim[0], blockDim[1]);
        break;
    default:
        printf("We should not be here");
        assert(0);
        break;
    }


    assert(a->block->alloc_size_ != 0);
    a->dim_ = dim + 1;

    for (int d = 0; d < dim; d++)
        a->blockDim[d] = blockDim[d];

    for (int d = 0; d < a->dim_ - 1; d++) {
        a->size[d] = sizes[d] / a->blockDim[d] + (sizes[d] % a->blockDim[d] != 0);
        a->unblocked_size[d] = sizes[d];
    }

    a->size[dim] = a->block->alloc_size_;
    // we need proper alignment here. But can be done later
    /* a->ld_ = (sizes[a->dim_ - 1] / 32 + 1) * 32; */
    a->ld_ = a->block->alloc_size_;
    switch(a->dim_) {
    case 5: {
        a->offsets[0] = a->ld_ * a->size[1] * a->size[2] * a->size[3];
        a->offsets[1] = a->ld_ * a->size[1] * a->size[2];
        a->offsets[2] = a->ld_ * a->size[2];
        a->offsets[3] = a->ld_;
        break;
    }
    case 4: {
        a->offsets[0] = a->ld_ * a->size[1] * a->size[2];
        a->offsets[1] = a->ld_ * a->size[2];
        a->offsets[2] = a->ld_;
        break;
    }
    case 3: {
        a->offsets[0] = a->ld_ * a->size[1];
        a->offsets[1] = a->ld_;
    }
        break;
    case 2: { // matrix case
        a->offsets[0] = a->ld_;
    }
        break;
    case 1:
        break;
    }

    a->alloc_size_ = a->offsets[0] * a->size[0];
    a->blocked_decomposition = true;
    assert(a->alloc_size_ != 0);
    return;
}

size_t realloc_tensor(tensor *t)
{
    if (t == NULL) {
        abort();
    }

    if (t->alloc_size_ == 0) {
        /* there is a mistake somewhere. We can not have t->old_alloc_size_ != 0 and no allocation */
        abort();
    }

    if ((t->old_alloc_size_ >= t->alloc_size_) && (t->data != NULL))
        return t->alloc_size_;

    if ((t->old_alloc_size_ < t->alloc_size_) && (t->data != NULL)) {
        free(t->data);
    }

    t->data = NULL;

    if (t->data == NULL) {
        if (posix_memalign((void **)&t->data, 32, sizeof(double) * t->alloc_size_) != 0)
            abort();
        t->old_alloc_size_ = t->alloc_size_;
    }

    return t->alloc_size_;
}

void dgemm_simplified(dgemm_params *const m, const bool use_libxsmm)
{
    if (m == NULL)
        abort();

#if defined(__LIBXSMM)
    if (use_libxsmm) {
        /* we are in row major but xsmm is in column major */
        m->prefetch = LIBXSMM_PREFETCH_AUTO;
        if ((m->op1 == 'N') && (m->op2 == 'N')) {
            m->flags =  LIBXSMM_GEMM_FLAG_NONE;
        }

        if ((m->op1 == 'T') && (m->op2 == 'N')) {
            m->flags =  LIBXSMM_GEMM_FLAG_TRANS_B;
        }

        if ((m->op1 == 'N') && (m->op2 == 'T')) {
            m->flags =  LIBXSMM_GEMM_FLAG_TRANS_A;
        }

        if ((m->op1 == 'T') && (m->op2 == 'T')) {
            m->flags =  LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B;
        }


        if (m->kernel == NULL) {
            m->kernel = libxsmm_dmmdispatch(m->n,
                                            m->m,
                                            m->k,
                                            &m->ldb,
                                            &m->lda,
                                            &m->ldc,
                                            &m->alpha,
                                            &m->beta,
                                            &m->flags,
                                            &m->prefetch);
        }

        if (m->kernel) {
            m->kernel(m->b, m->a, m->c, m->b, m->a, m->c);
            return;
        }
    }
#endif

#if defined(__MKL)
    // fall back to mkl
    if ((m->op1 == 'N') && (m->op2 == 'N'))
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m->m, m->n, m->k, m->alpha,
                    m->a, m->lda, m->b, m->ldb,
                    m->beta, m->c, m->ldc);

    if ((m->op1 == 'T') && (m->op2 == 'N'))
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    m->m, m->n, m->k, m->alpha,
                    m->a, m->lda, m->b, m->ldb,
                    m->beta, m->c, m->ldc);

    if ((m->op1 == 'N') && (m->op2 == 'T'))
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    m->m, m->n, m->k, m->alpha,
                    m->a, m->lda, m->b, m->ldb,
                    m->beta, m->c, m->ldc);

    if ((m->op1 == 'T') && (m->op2 == 'T'))
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
                    m->m, m->n, m->k, m->alpha,
                    m->a, m->lda, m->b, m->ldb,
                    m->beta, m->c, m->ldc);

#else

    if ((m->op1 == 'N') && (m->op2 == 'N'))
        dgemm_("N", "N", &m->n, &m->m, &m->k, &m->alpha, m->b, &m->ldb, m->a, &m->lda, &m->beta, m->c, &m->ldc);

    if ((m->op1 == 'T') && (m->op2 == 'N'))
        dgemm_("N", "T", &m->n, &m->m, &m->k, &m->alpha, m->b, &m->ldb, m->a, &m->lda, &m->beta, m->c, &m->ldc);

    if ((m->op1 == 'T') && (m->op2 == 'T'))
        dgemm_("T", "T", &m->n, &m->m, &m->k, &m->alpha, m->b, &m->ldb, m->a, &m->lda, &m->beta, m->c, &m->ldc);

    if ((m->op1 == 'N') && (m->op2 == 'T'))
        dgemm_("T", "N", &m->n, &m->m, &m->k, &m->alpha, m->b, &m->ldb, m->a, &m->lda, &m->beta, m->c, &m->ldc);

#endif
}

void batched_dgemm_simplified(dgemm_params *const m, const int batch_size, const bool use_libxsmm)
{
    assert(m != NULL);
    assert(batch_size > 0);

#if defined(__LIBXSMM)

    if (use_libxsmm) {
        libxsmm_dmmfunction kernel;

/* we are in row major but xsmm is in column major */
        m->prefetch = LIBXSMM_PREFETCH_AUTO;
        if ((m->op1 == 'N') && (m->op2 == 'N')) {
            m->flags =  LIBXSMM_GEMM_FLAG_NONE;
        }

        if ((m->op1 == 'T') && (m->op2 == 'N')) {
            m->flags =  LIBXSMM_GEMM_FLAG_TRANS_B;
        }

        if ((m->op1 == 'N') && (m->op2 == 'T')) {
            m->flags =  LIBXSMM_GEMM_FLAG_TRANS_A;
        }

        if ((m->op1 == 'T') && (m->op2 == 'T')) {
            m->flags =  LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B;
        }

        if (m->kernel == NULL) {
            m->kernel = libxsmm_dmmdispatch(m->n,
                                            m->m,
                                            m->k,
                                            &m->ldb,
                                            &m->lda,
                                            &m->ldc,
                                            &m->alpha,
                                            &m->beta,
                                            &m->flags,
                                            &m->prefetch);
        }

        kernel = m->kernel;

        if (kernel) {
            for (int s = 0; s < batch_size - 1; s++) {
                kernel(m[s].b, m[s].a, m[s].c ,
                       m[s + 1].b, m[s + 1].a, m[s + 1].c);
            }
            kernel(m[batch_size - 1].b, m[batch_size - 1].a, m[batch_size - 1].c,
                   m[batch_size - 1].b, m[batch_size - 1].a, m[batch_size - 1].c);
            return;
        }
    }
#endif

#if defined(__MKL)
    // fall back to mkl
    for (int s = 0; s < batch_size; s++) {
        if ((m[s].op1 == 'N') && (m[s].op2 == 'N'))
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m[s].m, m[s].n, m[s].k, m[s].alpha,
                        m[s].a, m[s].lda, m[s].b, m[s].ldb,
                        m[s].beta, m[s].c, m[s].ldc);

        if ((m[s].op1 == 'T') && (m[s].op2 == 'N'))
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    m[s].m, m[s].n, m[s].k, m[s].alpha,
                    m[s].a, m[s].lda, m[s].b, m[s].ldb,
                    m[s].beta, m[s].c, m[s].ldc);

        if ((m[s].op1 == 'N') && (m[s].op2 == 'T'))
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        m[s].m, m[s].n, m[s].k, m[s].alpha,
                        m[s].a, m[s].lda, m[s].b, m[s].ldb,
                        m[s].beta, m[s].c, m[s].ldc);

        if ((m[s].op1 == 'T') && (m[s].op2 == 'T'))
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
                        m[s].m, m[s].n, m[s].k, m[s].alpha,
                        m[s].a, m[s].lda, m[s].b, m[s].ldb,
                        m[s].beta, m[s].c, m[s].ldc);
    }
#else
    for (int s = 0; s < batch_size; s++) {
        if ((m[s].op1 == 'N') && (m[s].op2 == 'N'))
            dgemm_("N", "N", &m[s].n, &m[s].m, &m[s].k, &m[s].alpha, m[s].b, &m[s].ldb, m[s].a, &m[s].lda, &m[s].beta, m[s].c, &m[s].ldc);

        if ((m[s].op1 == 'T') && (m[s].op2 == 'N'))
            dgemm_("N", "T", &m[s].n, &m[s].m, &m[s].k, &m[s].alpha, m[s].b, &m[s].ldb, m[s].a, &m[s].lda, &m[s].beta, m[s].c, &m[s].ldc);

        if ((m[s].op1 == 'T') && (m[s].op2 == 'T'))
            dgemm_("T", "T", &m[s].n, &m[s].m, &m[s].k, &m[s].alpha, m[s].b, &m[s].ldb, m[s].a, &m[s].lda, &m[s].beta, m[s].c, &m[s].ldc);

        if ((m[s].op1 == 'N') && (m[s].op2 == 'T'))
            dgemm_("T", "N", &m[s].n, &m[s].m, &m[s].k, &m[s].alpha, m[s].b, &m[s].ldb, m[s].a, &m[s].lda, &m[s].beta, m[s].c, &m[s].ldc);
    }
#endif
}

void extract_sub_grid(const int *lower_corner,
                      const int *upper_corner,
                      const int *position,
                      const tensor *const grid,
                      tensor *const subgrid)
{
    int position1[3] = {0, 0, 0};

    if (position) {
        position1[0] = position[0];
        position1[1] = position[1];
        position1[2] = position[2];
    }

    const int sizex = upper_corner[2] - lower_corner[2];
    const int sizey = upper_corner[1] - lower_corner[1];
    const int sizez = upper_corner[0] - lower_corner[0];

    /* libxsmm_mcopy_descriptor_init(libxsmm_descriptor_blob* blob, */
    /*                               unsigned int typesize, unsigned int m, unsigned int n, unsigned int ldo, */
    /*                               unsigned int ldi, int flags, int prefetch, const int* unroll); */

    for (int z = 0; z < sizez; z++) {
/* #if defined(__LIBXSMM) */
/*         libxsmm_matcopy(&idx3(subgrid[0], position1[0] + z, position1[1], position1[2]), */
/*                         &idx3(grid[0], lower_corner[0] + z, lower_corner[1], lower_corner[2]), */
/*                         sizeof(double), */
/*                         sizex, */
/*                         sizey, */
/*                         grid[0].ld_, */
/*                         subgrid[0].ld_); */
/* #else */
        for (int y = 0; y < sizey; y++) {
            double *__restrict__ src = &idx3(grid[0], lower_corner[0] + z, lower_corner[1] + y, lower_corner[2]);
            double *__restrict__ dst = &idx3(subgrid[0], position1[0] + z, position1[1] + y, position1[2]);
            LIBXSMM_PRAGMA_SIMD
            for (int x = 0; x < sizex; x++) {
                dst[x] = src[x];
            }
        }
/* #endif */
    }

    return;
}

void add_sub_grid(const int *lower_corner,
                  const int *upper_corner,
                  const int *position,
                  const tensor *subgrid,
                  tensor *grid)
{

    int position1[3] = {0, 0, 0};

    if (position) {
        position1[0] = position[0];
        position1[1] = position[1];
        position1[2] = position[2];
    }
    for (int d = 0; d < 3; d++) {
        if ((lower_corner[d] >= grid->size[d]) ||
            (lower_corner[d] < 0) ||
            (lower_corner[d] >= upper_corner[d]) ||
            (upper_corner[d] > grid->size[d]) ||
            (upper_corner[d] <= 0) ||
            (upper_corner[d] - lower_corner[d] > subgrid->size[d]) ||
            (grid == NULL) ||
            (subgrid == NULL)) {

            printf("Error : invalid parameters. Values of the given parameters along the first wrong dimension\n");
            printf("      : lorner corner  [%d] = %d\n", d, lower_corner[d]);
            printf("      : upper  corner  [%d] = %d\n", d, upper_corner[d]);
            printf("      : diff           [%d] = %d\n", d, upper_corner[d] - lower_corner[d]);
            printf("      : src grid size  [%d] = %d\n", d, subgrid->size[d]);
            printf("      : dst grid size  [%d] = %d\n", d, grid->size[d]);
            abort();
        }
    }

    const int sizex = upper_corner[2] - lower_corner[2];
    const int sizey = upper_corner[1] - lower_corner[1];
    const int sizez = upper_corner[0] - lower_corner[0];


    for (int z = 0; z < sizez; z++) {
        double *__restrict__ dst = &idx3(grid[0], lower_corner[0] + z, lower_corner[1], lower_corner[2]);
        double *__restrict__ src = &idx3(subgrid[0], position1[0] + z, position1[1], position1[2]);
        for (int y = 0; y < sizey; y++) {
            LIBXSMM_PRAGMA_SIMD
            for (int x = 0; x < sizex; x++) {
                dst[x] += src[x];
            }
            dst += grid->ld_;
            src += subgrid->ld_;
        }
    }
    return;
}

/* add a 3D block given by subgrid and apply pcb along the fastest dimension */

void add_sub_grid_with_pcb(const int *period,
                           const int *position_inside_subgrid,
                           const int *lower_corner,
                           const int *upper_corner,
                           const tensor *subgrid,
                           tensor *grid)
{

    int position1[3] = {0, 0, 0};

    if (position_inside_subgrid) {
        position1[0] = position_inside_subgrid[0];
        position1[1] = position_inside_subgrid[1];
        position1[2] = position_inside_subgrid[2];
    }
    for (int d = 0; d < 2; d++) {
        if ((lower_corner[d] >= grid->size[d]) ||
            (lower_corner[d] < 0) ||
            (lower_corner[d] >= upper_corner[d]) ||
            (upper_corner[d] > grid->size[d]) ||
            (upper_corner[d] <= 0) ||
            (upper_corner[d] - lower_corner[d] > subgrid->size[d]) ||
            (grid == NULL) ||
            (subgrid == NULL)) {

            printf("Error : invalid parameters. Values of the given parameters along the first wrong dimension\n");
            printf("      : lorner corner  [%d] = %d\n", d, lower_corner[d]);
            printf("      : upper  corner  [%d] = %d\n", d, upper_corner[d]);
            printf("      : diff           [%d] = %d\n", d, upper_corner[d] - lower_corner[d]);
            printf("      : src grid size  [%d] = %d\n", d, subgrid->size[d]);
            printf("      : dst grid size  [%d] = %d\n", d, grid->size[d]);
            abort();
        }
    }

    const int sizey = upper_corner[1] - lower_corner[1];
    const int sizez = upper_corner[0] - lower_corner[0];

    const int offset = min(grid->size[2] - lower_corner[2], subgrid->size[2]);
    const int loop_number = (subgrid->size[2] - offset) / period[2];
    const int reminder = min(grid->size[2], subgrid->size[2] - offset - loop_number * period[2]);

    for (int z = 0; z < sizez; z++) {
        double *__restrict__ dst = &idx3(grid[0], lower_corner[0] + z, lower_corner[1], 0);
        double *__restrict__ src = &idx3(subgrid[0], position1[0] + z, position1[1], 0);
        for (int y = 0; y < sizey; y++) {
            //#pragma omp simd
            /* for (int x = 0; x < grid->size[2]; x++) */
#pragma GCC ivdep
            for (int x = 0; x < offset; x++) {
                dst[x + offset] += src[x];
            }

            int shift = offset;
            for (int l = 0; l < loop_number; l++) {
#pragma GCC ivdep
                for (int x = 0; x < grid->size[2]; x++) {
                    dst[x] += src[shift + x];
                }
                shift = offset + (l + 1) * period[2];
            }

#pragma GCC ivdep
            for (int x = 0; x < reminder; x++)
                dst[x] += src[shift + x];

            dst += grid->ld_;
            src += subgrid->ld_;
        }
    }

    return;
}


inline int compute_cube_properties(const bool ortho,
                                   const double radius,
                                   const double dh[3][3],
                                   const double dh_inv[3][3],
                                   const double *rp,
                                   double *disr_radius,
                                   double *roffset,
                                   int *cubecenter,
                                   int *lb_cube,
                                   int *ub_cube,
                                   int *cube_size)
{
    int cmax = 0;

    /* center of the gaussian in the lattice coordinates */
    double rp1[3];

    /* it is in the lattice vector frame */
    for (int i=0; i<3; i++) {
        double dh_inv_rp = 0.0;
        for (int j=0; j<3; j++) {
            dh_inv_rp += dh_inv[j][i] * rp[j];
        }
        rp1[2 - i] = dh_inv_rp;
        cubecenter[2 - i] = floor(dh_inv_rp);
    }

    if (ortho) {
        /* seting up the cube parameters */
        const double dx[3] = {dh[2][2], dh[1][1], dh[0][0]};
        const double dx_inv[3] = {dh_inv[2][2], dh_inv[1][1], dh_inv[0][0]};
        /* cube center */

        /* lower and upper bounds */

        // Historically, the radius gets discretized.
        const double drmin = min(dh[0][0], min(dh[1][1], dh[2][2]));
        *disr_radius = drmin * max(1.0, ceil(radius/drmin));

        for (int i=0; i<3; i++) {
            roffset[i] = rp[2 - i] - ((double) cubecenter[i]) * dx[i];
        }

        for (int i = 0; i < 3; i++) {
            lb_cube[i] = ceil(-1e-8 - *disr_radius * dx_inv[i]);
        }
    } else {
        for (int idir=0; idir<3; idir++) {
            lb_cube[idir] = INT_MAX;
            ub_cube[idir] = INT_MIN;
        }
        for (int i=-1; i<=1; i++) {
            for (int j=-1; j<=1; j++) {
                for (int k=-1; k<=1; k++) {
                    const double x = /* rp[0] + */ ((double)i) * radius;
                    const double y = /* rp[1] + */ ((double)j) * radius;
                    const double z = /* rp[2] + */ ((double)k) * radius;
                    for (int idir=0; idir<3; idir++) {
                        const double resc = dh_inv[0][idir] * x + dh_inv[1][idir] * y + dh_inv[2][idir] * z;
                        lb_cube[idir] = min(lb_cube[idir], lrint(resc));
                        ub_cube[idir] = max(ub_cube[idir], lrint(resc));
                    }
                }
            }
        }

        for (int i = 0; i < 3; i++) {
            lb_cube[i] = -(ub_cube[i] - lb_cube[i]) / 2;
        }

        /* compute the offset in lattice coordinates */

        for (int i=0; i<3; i++) {
            roffset[i] = rp1[i] - cubecenter[i];
        }
    }


    // Symetric interval
    for (int i = 0; i < 3; i++) {
        ub_cube[i] = - lb_cube[i];
    }

/* compute the cube size ignoring periodicity */
    cube_size[0] = ub_cube[0] - lb_cube[0] + 1;
    cube_size[1] = ub_cube[1] - lb_cube[1] + 1;
    cube_size[2] = ub_cube[2] - lb_cube[2] + 1;


    for (int i = 0; i < 3; i++) {
        cmax = max(cmax, cube_size[i]);
    }

    return cmax + 1;
}

void  return_cube_position(const int *grid_size,
                           const int *lb_grid,
                           const int *cube_center,
                           const int *lower_boundaries_cube,
                           const int *period,
                           int *const position)
{
    position[0] = (lb_grid[0] + cube_center[0] + lower_boundaries_cube[0] + 32 * period[0]) % period[0];
    position[1] = (lb_grid[1] + cube_center[1] + lower_boundaries_cube[1] + 32 * period[1]) % period[1];
    position[2] = (lb_grid[2] + cube_center[2] + lower_boundaries_cube[2] + 32 * period[2]) % period[2];

    if ((position[0] >= grid_size[0]) || (position[1] >= grid_size[1]) || (position[2] >= grid_size[2])) {
        printf("the lower corner of the cube is outside the grid\n");
        abort();
    }
}

void compute_block_boundaries(const int *blockDim,
                              const int *lb_grid,
                              const int *grid_size,
                              const int *blocked_grid_size,
                              const int *period,
                              const int *cube_center,
                              const int *cube_size,
                              const int *lower_boundaries_cube,
                              int *lower_block_corner,
                              int *upper_block_corner,
                              int *pol_offsets)
{
    int position[3];
    return_cube_position(grid_size,
                         lb_grid,
                         cube_center,
                         lower_boundaries_cube,
                         period,
                         position);
    pol_offsets[0] = 0;
    pol_offsets[1] = 0;
    pol_offsets[2] = 0;

    for (int axis = 0; axis < 3; axis++) {
        int tmp = position[axis];
        int blockidx = tmp / blockDim[axis];
        lower_block_corner[axis] = blockidx;
        pol_offsets[axis] = tmp - blockidx * blockDim[axis];
        tmp = position[axis] + cube_size[axis];
        if ((grid_size[axis] != period[axis]) && (tmp > grid_size[axis])) {
            upper_block_corner[axis] = blocked_grid_size[axis];
        } else {
            upper_block_corner[axis] = tmp / blockDim[axis] + ((tmp - blockidx * blockDim[axis]) != 0);
        }
    }

    return;
}

void verify_orthogonality(const double dh[3][3], bool orthogonal[3])
{
    double norm1, norm2, norm3;

    norm1 = dh[0][0] * dh[0][0] + dh[0][1] * dh[0][1] + dh[0][2] * dh[0][2];
    norm2 = dh[1][0] * dh[1][0] + dh[1][1] * dh[1][1] + dh[1][2] * dh[1][2];
    norm3 = dh[2][0] * dh[2][0] + dh[2][1] * dh[2][1] + dh[2][2] * dh[2][2];

    norm1 = 1.0 / sqrt(norm1);
    norm2 = 1.0 / sqrt(norm2);
    norm3 = 1.0 / sqrt(norm3);

    /* x z */
    orthogonal[0] = ((fabs(dh[0][0] * dh[2][0] + dh[0][1] * dh[2][1] + dh[0][2] * dh[2][2]) * norm1 * norm3) < 1e-12);
    /* y z */
    orthogonal[1] = ((fabs(dh[1][0] * dh[2][0] + dh[1][1] * dh[2][1] + dh[1][2] * dh[2][2]) * norm2 * norm3) < 1e-12);
    /* x y */
    orthogonal[2] = ((fabs(dh[0][0] * dh[1][0] + dh[0][1] * dh[1][1] + dh[0][2] * dh[1][2]) * norm1 * norm2) < 1e-12);
}

int return_exponents(const int index) {

    // the possible indices are encoded as z * 2 ^ 16 + y * 2 ^ 8 + x.
    static const int exponents[] = {
        0,      1,      256,    65536,  2,      257,    512,    65537,  65792,
        131072, 3,      258,    513,    768,    65538,  65793,  66048,  131073,
        131328, 196608, 4,      259,    514,    769,    1024,   65539,  65794,
        66049,  66304,  131074, 131329, 131584, 196609, 196864, 262144, 5,
        260,    515,    770,    1025,   1280,   65540,  65795,  66050,  66305,
        66560,  131075, 131330, 131585, 131840, 196610, 196865, 197120, 262145,
        262400, 327680, 6,      261,    516,    771,    1026,   1281,   1536,
        65541,  65796,  66051,  66306,  66561,  66816,  131076, 131331, 131586,
        131841, 132096, 196611, 196866, 197121, 197376, 262146, 262401, 262656,
        327681, 327936, 393216, 7,      262,    517,    772,    1027,   1282,
        1537,   1792,   65542,  65797,  66052,  66307,  66562,  66817,  67072,
        131077, 131332, 131587, 131842, 132097, 132352, 196612, 196867, 197122,
        197377, 197632, 262147, 262402, 262657, 262912, 327682, 327937, 328192,
        393217, 393472, 458752, 8,      263,    518,    773,    1028,   1283,
        1538,   1793,   2048,   65543,  65798,  66053,  66308,  66563,  66818,
        67073,  67328,  131078, 131333, 131588, 131843, 132098, 132353, 132608,
        196613, 196868, 197123, 197378, 197633, 197888, 262148, 262403, 262658,
        262913, 263168, 327683, 327938, 328193, 328448, 393218, 393473, 393728,
        458753, 459008, 524288, 9,      264,    519,    774,    1029,   1284,
        1539,   1794,   2049,   2304,   65544,  65799,  66054,  66309,  66564,
        66819,  67074,  67329,  67584,  131079, 131334, 131589, 131844, 132099,
        132354, 132609, 132864, 196614, 196869, 197124, 197379, 197634, 197889,
        198144, 262149, 262404, 262659, 262914, 263169, 263424, 327684, 327939,
        328194, 328449, 328704, 393219, 393474, 393729, 393984, 458754, 459009,
        459264, 524289, 524544, 589824};
    return exponents[index];
}
