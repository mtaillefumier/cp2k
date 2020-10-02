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

void convert_to_lattice_coordinates(const double dh_inv_[3][3], const double *__restrict__ const rp, double *__restrict__ rp_c)
{
    rp_c[0] = dh_inv_[0][0] * rp[0] + dh_inv_[1][0] * rp[1] + dh_inv_[0][0] * rp[2];
    rp_c[1] = dh_inv_[0][1] * rp[0] + dh_inv_[1][1] * rp[1] + dh_inv_[1][1] * rp[2];
    rp_c[2] = dh_inv_[0][2] * rp[0] + dh_inv_[1][2] * rp[1] + dh_inv_[2][2] * rp[2];
}

void
dgemm_simplified(dgemm_params* const m, const bool use_libxsmm)
{
    if (m == NULL)
        abort();

#if defined(__LIBXSMM)
    if (use_libxsmm) {
        /* we are in row major but xsmm is in column major */
        m->prefetch = LIBXSMM_PREFETCH_AUTO;
        if ((m->op1 == 'N') && (m->op2 == 'N')) {
            m->flags = LIBXSMM_GEMM_FLAG_NONE;
        }

        if ((m->op1 == 'T') && (m->op2 == 'N')) {
            m->flags = LIBXSMM_GEMM_FLAG_TRANS_B;
        }

        if ((m->op1 == 'N') && (m->op2 == 'T')) {
            m->flags = LIBXSMM_GEMM_FLAG_TRANS_A;
        }

        if ((m->op1 == 'T') && (m->op2 == 'T')) {
            m->flags = LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B;
        }

        if (m->kernel == NULL) {
            m->kernel = libxsmm_dmmdispatch(m->n, m->m, m->k, &m->ldb, &m->lda, &m->ldc, &m->alpha, &m->beta, &m->flags,
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
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m->m, m->n, m->k, m->alpha, m->a, m->lda, m->b, m->ldb,
                    m->beta, m->c, m->ldc);

    if ((m->op1 == 'T') && (m->op2 == 'N'))
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m->m, m->n, m->k, m->alpha, m->a, m->lda, m->b, m->ldb,
                    m->beta, m->c, m->ldc);

    if ((m->op1 == 'N') && (m->op2 == 'T'))
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m->m, m->n, m->k, m->alpha, m->a, m->lda, m->b, m->ldb,
                    m->beta, m->c, m->ldc);

    if ((m->op1 == 'T') && (m->op2 == 'T'))
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, m->m, m->n, m->k, m->alpha, m->a, m->lda, m->b, m->ldb,
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

void
batched_dgemm_simplified(dgemm_params* const m, const int batch_size, const bool use_libxsmm)
{
    assert(m != NULL);
    assert(batch_size > 0);

#if defined(__LIBXSMM)

    if (use_libxsmm) {
        libxsmm_dmmfunction kernel;

        /* we are in row major but xsmm is in column major */
        m->prefetch = LIBXSMM_PREFETCH_AUTO;
        if ((m->op1 == 'N') && (m->op2 == 'N')) {
            m->flags = LIBXSMM_GEMM_FLAG_NONE;
        }

        if ((m->op1 == 'T') && (m->op2 == 'N')) {
            m->flags = LIBXSMM_GEMM_FLAG_TRANS_B;
        }

        if ((m->op1 == 'N') && (m->op2 == 'T')) {
            m->flags = LIBXSMM_GEMM_FLAG_TRANS_A;
        }

        if ((m->op1 == 'T') && (m->op2 == 'T')) {
            m->flags = LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B;
        }

        if (m->kernel == NULL) {
            m->kernel = libxsmm_dmmdispatch(m->n, m->m, m->k, &m->ldb, &m->lda, &m->ldc, &m->alpha, &m->beta, &m->flags,
                                            &m->prefetch);
        }

        kernel = m->kernel;

        if (kernel) {
            for (int s = 0; s < batch_size - 1; s++) {
                kernel(m[s].b, m[s].a, m[s].c, m[s + 1].b, m[s + 1].a, m[s + 1].c);
            }
            kernel(m[batch_size - 1].b, m[batch_size - 1].a, m[batch_size - 1].c, m[batch_size - 1].b,
                   m[batch_size - 1].a, m[batch_size - 1].c);
            return;
        }
    }
#endif

#if defined(__MKL)
    // fall back to mkl
    for (int s = 0; s < batch_size; s++) {
        if ((m[s].op1 == 'N') && (m[s].op2 == 'N'))
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m[s].m, m[s].n, m[s].k, m[s].alpha, m[s].a, m[s].lda,
                        m[s].b, m[s].ldb, m[s].beta, m[s].c, m[s].ldc);

        if ((m[s].op1 == 'T') && (m[s].op2 == 'N'))
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m[s].m, m[s].n, m[s].k, m[s].alpha, m[s].a, m[s].lda,
                        m[s].b, m[s].ldb, m[s].beta, m[s].c, m[s].ldc);

        if ((m[s].op1 == 'N') && (m[s].op2 == 'T'))
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m[s].m, m[s].n, m[s].k, m[s].alpha, m[s].a, m[s].lda,
                        m[s].b, m[s].ldb, m[s].beta, m[s].c, m[s].ldc);

        if ((m[s].op1 == 'T') && (m[s].op2 == 'T'))
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, m[s].m, m[s].n, m[s].k, m[s].alpha, m[s].a, m[s].lda,
                        m[s].b, m[s].ldb, m[s].beta, m[s].c, m[s].ldc);
    }
#else
    for (int s = 0; s < batch_size; s++) {
        if ((m[s].op1 == 'N') && (m[s].op2 == 'N'))
            dgemm_("N", "N", &m[s].n, &m[s].m, &m[s].k, &m[s].alpha, m[s].b, &m[s].ldb, m[s].a, &m[s].lda, &m[s].beta,
                   m[s].c, &m[s].ldc);

        if ((m[s].op1 == 'T') && (m[s].op2 == 'N'))
            dgemm_("N", "T", &m[s].n, &m[s].m, &m[s].k, &m[s].alpha, m[s].b, &m[s].ldb, m[s].a, &m[s].lda, &m[s].beta,
                   m[s].c, &m[s].ldc);

        if ((m[s].op1 == 'T') && (m[s].op2 == 'T'))
            dgemm_("T", "T", &m[s].n, &m[s].m, &m[s].k, &m[s].alpha, m[s].b, &m[s].ldb, m[s].a, &m[s].lda, &m[s].beta,
                   m[s].c, &m[s].ldc);

        if ((m[s].op1 == 'N') && (m[s].op2 == 'T'))
            dgemm_("T", "N", &m[s].n, &m[s].m, &m[s].k, &m[s].alpha, m[s].b, &m[s].ldb, m[s].a, &m[s].lda, &m[s].beta,
                   m[s].c, &m[s].ldc);
    }
#endif
}

void
extract_sub_grid(const int* lower_corner, const int* upper_corner, const int* position, const tensor* const grid,
                 tensor* const subgrid)
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
            double* __restrict__ src =
                &idx3(grid[0], lower_corner[0] + z - grid->window_shift[0], lower_corner[1] + y - grid->window_shift[1],
                      lower_corner[2] - grid->window_shift[2]);
            double* __restrict__ dst = &idx3(subgrid[0], position1[0] + z, position1[1] + y, position1[2]);
            LIBXSMM_PRAGMA_SIMD
            for (int x = 0; x < sizex; x++) {
                dst[x] = src[x];
            }
        }
        /* #endif */
    }

    return;
}

void
add_sub_grid(const int* lower_corner, const int* upper_corner, const int* position, const tensor* subgrid, tensor* grid)
{

    int position1[3] = {0, 0, 0};

    if (position) {
        position1[0] = position[0];
        position1[1] = position[1];
        position1[2] = position[2];
    }
    for (int d = 0; d < 3; d++) {
        if ((lower_corner[d] < grid->window_shift[d]) || (lower_corner[d] < 0) ||
            (lower_corner[d] >= upper_corner[d]) ||
            (upper_corner[d] > (grid->window_shift[d] + grid->window_size[d])) || (upper_corner[d] <= 0) ||
            (upper_corner[d] - lower_corner[d] > subgrid->size[d]) || (grid == NULL) || (subgrid == NULL)) {

            printf("Error : invalid parameters. Values of the given parameters along the first wrong dimension\n");
            printf("      : lorner corner  [%d] = %d\n", d, lower_corner[d]);
            printf("      : upper  corner  [%d] = %d\n", d, upper_corner[d]);
            printf("      : diff           [%d] = %d\n", d, upper_corner[d] - lower_corner[d]);
            printf("      : src grid size  [%d] = %d\n", d, subgrid->size[d]);
            printf("      : dst grid size  [%d] = %d\n", d, grid->size[d]);
            printf("      : window dst grid size  [%d] = %d\n", d, grid->window_size[d]);
            printf("      : window dst shift  [%d] = %d\n", d, grid->window_shift[d]);
            abort();
        }
    }

    const int sizex = upper_corner[2] - lower_corner[2];
    const int sizey = upper_corner[1] - lower_corner[1];
    const int sizez = upper_corner[0] - lower_corner[0];

    for (int z = 0; z < sizez; z++) {
        double* __restrict__ dst =
            &idx3(grid[0], lower_corner[0] + z - grid->lower_corner[0], lower_corner[1] - grid->lower_corner[1],
                  lower_corner[2] - grid->lower_corner[2]);
        double* __restrict__ src = &idx3(subgrid[0], position1[0] + z, position1[1], position1[2]);
        for (int y = 0; y < sizey; y++) {
            /* memcpy(dst, src, sizeof(double) * sizex); */
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

/* compute the functions (x - x_i)^l exp (-eta (x - x_i)^2) for l = 0..lp using
 * a recursive relation to avoid computing the exponential on each grid point. I
 * think it is not really necessary anymore since it is *not* the dominating
 * contribution to computation of collocate and integrate */

void
grid_fill_pol_dgemm(const bool transpose, const double dr, const double roffset, const int pol_offset, const int xmin,
                    const int xmax, const int lp, const int cmax, const double zetp, double* pol_)
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
            double pg                            = t_exp_min_1;
            idx2(pol, 0, pol_offset + ig - xmin) = pg;
            if (lp > 0)
                idx2(pol, 1, pol_offset + ig - xmin) = rpg;
        }

        for (int ig = 1; ig <= xmax; ig++) {
            const double rpg = ig * dr - roffset;
            t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
            t_exp_plus_2 *= t_exp_2;
            double pg                            = t_exp_plus_1;
            idx2(pol, 0, pol_offset + ig - xmin) = pg;
            if (lp > 0)
                idx2(pol, 1, pol_offset + ig - xmin) = rpg;
        }

        /* compute the remaining powers using previously computed stuff */
        if (lp >= 2) {
            double* __restrict__ poly = &idx2(pol, 1, 0);
            double* __restrict__ src1 = &idx2(pol, 0, 0);
            double* __restrict__ dst  = &idx2(pol, 2, 0);
//#pragma omp simd
#pragma GCC ivdep
            for (int ig = 0; ig < (xmax - xmin + 1 + pol_offset); ig++)
                dst[ig] = src1[ig] * poly[ig] * poly[ig];
        }

        for (int icoef = 3; icoef <= lp; icoef++) {
            const double* __restrict__ poly = &idx2(pol, 1, 0);
            const double* __restrict__ src1 = &idx2(pol, icoef - 1, 0);
            double* __restrict__ dst        = &idx2(pol, icoef, 0);
//#pragma omp simd
#pragma GCC ivdep
            for (int ig = 0; ig < (xmax - xmin + 1 + pol_offset); ig++) {
                dst[ig] = src1[ig] * poly[ig];
            }
        }

        //
        if (lp > 0) {
            double* __restrict__ dst       = &idx2(pol, 1, 0);
            const double* __restrict__ src = &idx2(pol, 0, 0);
#pragma GCC ivdep
            for (int ig = 0; ig < (xmax - xmin + 1 + pol_offset); ig++) {
                dst[ig] *= src[ig];
            }
        }
    }
}

int
compute_cube_properties(const bool ortho, const double radius, const double dh[3][3], const double dh_inv[3][3],
                        const double* rp, double* disr_radius, double* roffset, int* cubecenter, int* lb_cube,
                        int* ub_cube, int* cube_size)
{
    int cmax = 0;

    /* center of the gaussian in the lattice coordinates */
    double rp1[3];


    /* it is in the lattice vector frame */
    for (int i = 0; i < 3; i++) {
        double dh_inv_rp = 0.0;
        for (int j = 0; j < 3; j++) {
            dh_inv_rp += dh_inv[j][i] * rp[j];
        }
        rp1[2 - i]        = dh_inv_rp;
        cubecenter[2 - i] = floor(dh_inv_rp);
    }

    if (ortho) {
        /* seting up the cube parameters */
        const double dx[3]     = {dh[2][2], dh[1][1], dh[0][0]};
        const double dx_inv[3] = {dh_inv[2][2], dh_inv[1][1], dh_inv[0][0]};
        /* cube center */

        /* lower and upper bounds */

        // Historically, the radius gets discretized.
        const double drmin = min(dh[0][0], min(dh[1][1], dh[2][2]));
        *disr_radius       = drmin * max(1.0, ceil(radius / drmin));

        for (int i = 0; i < 3; i++) {
            roffset[i] = rp[2 - i] - ((double)cubecenter[i]) * dx[i];
        }

        for (int i = 0; i < 3; i++) {
            lb_cube[i] = ceil(-1e-8 - *disr_radius * dx_inv[i]);
        }

        // Symmetric interval
        for (int i = 0; i < 3; i++) {
            ub_cube[i] = 1 - lb_cube[i];
        }

    } else {
        for (int idir = 0; idir < 3; idir++) {
            lb_cube[idir] = INT_MAX;
            ub_cube[idir] = INT_MIN;
        }

        /* compute the size of the box. It is a fairly trivial way to compute
         * the box and it may have far more point than needed */
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    double x[3] = {
                        /* rp[0] + */ ((double)i) * radius,
                        /* rp[1] + */ ((double)j) * radius,
                        /* rp[2] + */ ((double)k) * radius
                    };
                    /* convert_to_lattice_coordinates(dh_inv, x, y); */
                    for (int idir = 0; idir < 3; idir++) {
                        const double resc = dh_inv[0][idir] * x[0] + dh_inv[1][idir] * x[1] + dh_inv[2][idir] * x[2];
                        lb_cube[2 - idir]     = min(lb_cube[2 - idir], floor(resc));
                        ub_cube[2 - idir]     = max(ub_cube[2 - idir], ceil(resc));
                    }
                }
            }
        }


        /* for (int idir = 0; idir < 3; idir++) { */
        /*     lb_cube[idir] -= 1; */
        /*     ub_cube[idir] += 1; */
        /* } */

        /* compute the offset in lattice coordinates */

        for (int i = 0; i < 3; i++) {
            roffset[i] = rp1[i] - cubecenter[i];
        }
    }

    /* compute the cube size ignoring periodicity */
    cube_size[0] = ub_cube[0] - lb_cube[0] + 1;
    cube_size[1] = ub_cube[1] - lb_cube[1] + 1;
    cube_size[2] = ub_cube[2] - lb_cube[2] + 1;

    for (int i = 0; i < 3; i++) {
        cmax = max(cmax, cube_size[i]);
    }

    return cmax;
}

void
return_cube_position(const int* grid_size,
                     const int* lb_grid,
                     const int* cube_center,
                     const int* lower_boundaries_cube,
                     const int* period,
                     int* const position)
{
    position[0] = (lb_grid[0] + cube_center[0] + lower_boundaries_cube[0] + 1024 * period[0]) % period[0];
    position[1] = (lb_grid[1] + cube_center[1] + lower_boundaries_cube[1] + 1024 * period[1]) % period[1];
    position[2] = (lb_grid[2] + cube_center[2] + lower_boundaries_cube[2] + 1024 * period[2]) % period[2];

    if ((position[0] >= grid_size[0]) || (position[1] >= grid_size[1]) || (position[2] >= grid_size[2])) {
        printf("the lower corner of the cube is outside the grid\n");
        abort();
    }
}

void
verify_orthogonality(const double dh[3][3], bool orthogonal[3])
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
