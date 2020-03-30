#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#if defined(__MKL) || defined(HAVE_MKL)
#include <mkl.h>
#include <mkl_cblas.h>
#endif

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

#include "tensor_local.h"
// *****************************************************************************
#define min(x, y) ( ((x) < (y)) ? x : y )

// *****************************************************************************
#define max(x, y) ( ((x) > (y)) ? x : y )

// *****************************************************************************
#define mod(a, m)  ( ((a)%(m) + (m)) % (m) )



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
        /* for (int i=0; i <= 2*cmax; i++) */
        /*     map[i] = mod(cubecenter + i - cmax, npts) + 1; */
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

void collocate_core_rectangular_variant1(char *scratch,
                                         const int zmin,
                                         const int ymin,
                                         const int xmin,
                                         const int zmax,
                                         const int ymax,
                                         const int xmax,
                                         const int zoffset,
                                         const int yoffset,
                                         const int xoffset,
                                         const struct tensor_ *co,
                                         const struct tensor_ *p_alpha_beta_reduced_,
                                         struct tensor_ *grid)
{
    /* printf("cube size %d %d %d\n" , zmax - zmin, ymax - ymin, xmax - xmin); */
    /* printf("coord : %d %d %d\n" , zmin, ymin, xmin); */
    /* printf("grid size : %d %d %d\n" , grid->size[0], grid->size[1], grid->size[2]); */
    if (co->size[1] > 1) {
        // an helper structure for the dgemm parameters (*ROW MAJOR* format for
        // the matrices)
        struct {
            double alpha;
            double beta;
            double *a, *b, *c;
            int m, n, k, lda, ldb, ldc;
        } m1, m2, m3;

        tensor T;
        tensor W;

        initialize_tensor_3(&T, co->size[0] /* alpha */, co->size[1] /* gamma */, ymax - ymin /* j */);

        initialize_tensor_3(&W, co->size[1] /* alpha */ , zmax - zmin /* k */, ymax - ymin /* j */);

#if defined(__LIBXSMM)
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

        m1.alpha = 1.0;
        m1.beta = 0.0;
        m1.m = co->size[0] * co->size[1]; /* alpha gamma */
        m1.n = T.size[2]; /* j */
        m1.k = co->size[2]; /* beta */
        m1.a = co->data; // Coef_{alpha,gamma,beta} Coef_xzy
        m1.lda = co->ld_;
        m1.b = &idx3(p_alpha_beta_reduced_[0], 1, 0, yoffset); // Y_{beta, j} = p_alpha_beta_reduced(1, beta, j)
        m1.ldb = p_alpha_beta_reduced_->ld_;
        m1.c = T.data; // T_{\alpha, \gamma, j} = T(alpha, gamma, j)
        m1.ldc = T.ld_;

        if ((m1.m <= 0) || (m1.n <= 0) || (m1.k <= 0)) {
            printf("xmin %d\nxmax %d\nymin %d\nymax %d\nzmin %d\n zmax %d\n", xmin, xmax, ymin, ymax, zmin, zmax);
            printf("m1 (mnk) : %d %d %d\n", m1.m, m1.n, m1.k);
            abort();
        }

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

        m2.alpha = 1.0;
        m2.beta = 0.0;
        m2.m = zmax - zmin; // k direction
        m2.n = ymax - ymin; // j direction
        m2.k = co->size[2]; // gamma
        m2.a = &idx3(p_alpha_beta_reduced_[0], 2, 0, zoffset); // p_alpha_beta_reduced(0, gamma, j)
        m2.lda = p_alpha_beta_reduced_->ld_;
        m2.b = T.data; // T_{\alpha, \gamma, j}
        m2.ldb = T.ld_;
        m2.c = W.data; // W_{\alpha, k, j}
        m2.ldc = W.ld_;

        if ((m2.m <= 0) || (m2.n <= 0) || (m2.k <= 0)) {
            printf("xmin %d\nxmax %d\nymin %d\nymax %d\nzmin %d\n zmax %d\n", xmin, xmax, ymin, ymax, zmin, zmax);
            printf("m2 (mnk) : %d %d %d\n", m2.m, m2.n, m2.k);
            abort();
        }

/* the final step is again a reduction along the alpha indice. It can
 * again be done with one dgemm. The operation is simply
 *
 * Cube_{k, j, i} = \sum_{alpha} W_{k, j, alpha} X_{alpha, i}
 *
 * which means we need to permute W_{\alpha, k, j} in to W_{k, j,
 * \alpha} which can be done with one transposition if we consider (k,j)
 * as a composite index.
 */

        m3.alpha = 1.0;
        m3.beta = 0.0;
        m3.m = ymax - ymin; // (k, j)
        m3.n = xmax - xmin; // i direction
        m3.k = co->size[2]; // alpha
        m3.a = &idx3(W, 0, 0, 0); // W_{\alpha, k, j}
        m3.lda = W.size[1] * W.ld_;
        m3.b = &idx3(p_alpha_beta_reduced_[0], 0, 0, xoffset); // p_alpha_beta_reduced(2, alpha, i)
        m3.ldb = p_alpha_beta_reduced_->ld_;
        m3.c = &idx3(grid[0], 0, 0, 0); // cube_{kji}
        m3.ldc = grid->ld_;
        if ((m3.m <= 0) || (m3.n <= 0) || (m3.k <= 0)) {
            printf("xmin %d\nxmax %d\nymin %d\nymax %d\nzmin %d\n zmax %d\n", xmin, xmax, ymin, ymax, zmin, zmax);
            printf("m3 (mnk) : %d %d %d\n", m3.m, m3.n, m3.k);
            abort();
        }
// openblas and mkl have a C interface
#if defined(__LIBXSMM)
        libxsmm_dgemm("N",
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
#endif


/* #if defined(__LIBXSMM) */
/*         /\* batched interface *\/ */
/*         int prefetch = LIBXSMM_PREFETCH_AUTO; */
/*         int flags = LIBXSMM_GEMM_FLAG_TRANS_B; /\* LIBXSMM_FLAGS *\/; */
/*         libxsmm_dmmfunction xmm1 = NULL; */

/*         xmm1 = libxsmm_dmmdispatch(m2.n, m2.m, m2.k, */
/*                                   &m2.ldb, &m2.lda, &m2.ldc, */
/*                                   &m2.alpha, &m2.beta, */
/*                                   &flags, &prefetch); */

/*         for (int a1 = 0; a1 < co->size[0] - 1; a1++) { */
/*             xmm1(m2.b + a1 * T.offsets[0], m2.a, m2.c + a1 * W.offsets[0], */
/*                  m2.b + (a1 + 1) * T.offsets[0], m2.a, m2.c + (a1 + 1) * W.offsets[0]); */
/*         } */

/*         xmm1(m2.b + (co->size[0] - 1) * T.offsets[0], m2.a, m2.c + (co->size[0] - 1) * W.offsets[0], */
/*              m2.b + (co->size[0] - 1) * T.offsets[0], m2.a, m2.c + (co->size[0] - 1) * W.offsets[0]); */

/* #elif
 */
#if defined(__MKL) || defined(HAVE_MKL)
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
#else
        for (int a1 = 0; a1 < co->size[0]; a1++) {
            dgemm_("N",
                   "T",
                   &m2.n,
                   &m2.m,
                   &m2.k,
                   &m2.alpha,
                   m2.b,
                   &m2.ldb,
                   m2.a,
                   &m2.lda,
                   &m2.beta,
                   m2.c,
                   &m2.ldc);
            m2.b +=  T.offsets[0];
            m2.c +=  W.offsets[0];
        }
#endif

        /*
          In a normal collocate, I would not have the for loop, but here the
          the third dimension, needs a special care because I add the result
          directly to the grid without storing the result in a temporary grid.
        */

/* #if defined(__LIBXSMM) */
/*         libxsmm_dmmfunction xmm2 = NULL; */
/*         int prefetch = LIBXSMM_PREFETCH_NONE; */
/*         int flags = LIBXSMM_GEMM_FLAG_TRANS_B; /\* LIBXSMM_FLAGS *\/; */
/*         xmm2 = libxsmm_dmmdispatch(m3.n, */
/*                                    m3.m, */
/*                                    m3.k, */
/*                                    &m3.ldb, */
/*                                    &m3.lda, */
/*                                    &grid[0].ld_, */
/*                                    &m3.alpha, */
/*                                    &m3.beta, */
/*                                    &flags, */
/*                                    &prefetch); */

/*         for (int z = zmin; z < zmax; z++) { */
/*             xmm2(m3.b, */
/*                  &idx3(W, 0, z - zmin, 0), */
/*                  &idx3(grid[0], z, ymin, xmin)); */
/*         } */


        /* libxsmm_release_kernel(xmm1); */
        /* libxsmm_release_kernel(xmm2); */

#if defined(__MKL) || defined(HAVE_MKL)
        for (int z = zmin; z < zmax; z++) {
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m3.m, m3.n, m3.k,
                        1.0,
                        &idx3(W, 0, z - zmin, 0),
                        m3.lda,
                        m3.b,
                        m3.ldb,
                        1.0,
                        &idx3(grid[0],
                                 z,
                                 ymin,
                                 xmin),
                        grid[0].ld_);
        }
#else
        for (int z = zmin; z < zmax; z++) {
            dgemm_("N",
                   "T",
                   &m3.n,
                   &m3.m,
                   &m3.k,
                   &m3.alpha,
                   m3.b,
                   &m3.ldb,
                   &idx3(W, 0, z - zmin, 0),
                   &m3.lda,
                   &m3.beta,
                   &idx3(grid[0],
                         z,
                         ymin,
                         xmin),
                   &grid[0].ld_);
        }
#endif

#if defined(__LIBXSMM)
        libxsmm_free(W.data);
        libxsmm_free(T.data);
#endif

        return;
    }

    /* l = 0 case */
    const double *__restrict pz = &idx3(p_alpha_beta_reduced_[0], 2, 0, zoffset); /* k indice */
    const double *__restrict py = &idx3(p_alpha_beta_reduced_[0], 1, 0, yoffset); /* j indice */
    const double *__restrict px = &idx3(p_alpha_beta_reduced_[0], 0, 0, xoffset); /* i indice */
    const double coo = idx3 (co[0], 0, 0, 0);
    const double tz1 = pz[0];
    tensor tmp;

    initialize_tensor_2(&tmp, ymax - ymin, xmax - xmin);

#if defined(__LIBXSMM)
    tmp.data = libxsmm_aligned_scratch(sizeof(double) * tmp.alloc_size_, 0/*auto-alignment*/);
#else
#endif
    double *__restrict dst = &idx2(tmp, 0, 0);

    for (int y = 0; y < ymax - ymin; y++) {
        const double tp1 = coo * py[y];
#pragma GCC unroll 4
#pragma GCC ivdep
        for (int x1 = 0; x1 < xmax - xmin; x1++) {
            dst[x1] = tp1 * px[x1];
        }
        dst += tmp.ld_;
    }

    for (int z = 0; z < zmax - zmin; z++) {
        const double tz = pz[z];

        dst = &idx3(grid[0], zmin + z, ymin, xmin);
        const double *__restrict src1 = &idx2(tmp, 0, 0);


        for (int y = 0; y < tmp.size[0]; y++) {
#pragma GCC unroll 4
#pragma GCC ivdep
            for (int x = 0; x < tmp.size[1]; x++) {
                dst[x] += src1[x] * tz;
            }
            src1 += tmp.ld_;
            dst += grid->ld_;
        }
    }

#if defined(__LIBXSMM)
    libxsmm_free(tmp.data);
#endif
}


/* this function needs serious rethinking. Basically we apply a spherical mask
 * on a 3D grid and then add the result on a grid with PBC */

void apply_spherical_cutoff(const double disr_radius,
                            const double dh[3][3],
                            const double dh_inv[3][3],
                            const int lb_cube[3],
                            const int cmax,
                            tensor *cube)
{
    const int kgmin = (cube->size[0] - 1) / 2;
    const int jgmin = (cube->size[1] - 1) / 2;
    const int igmin = (cube->size[2] - 1) / 2;
    const double r2 = disr_radius * disr_radius;
    const double inv_dx = 1.0 / dh[0][0];
    const double rx = dh[0][0];
    const double ry = dh[1][1];
    const double rz = dh[2][2];

    for (int kg = -kgmin; kg <= kgmin + 1; kg++) {
        double kr = ((double)kg) * rz;   // distance from center in a.u.
        kr *= kr;
        for (int jg = -jgmin; jg <= jgmin + 1; jg++) {
            double jr = ((double)jg) * ry;   // distance from center in a.u.
            jr *= jr;
            const double rest = r2 - kr - jr;
            if (rest < 1e-8) {
                memset(&idx3(cube[0], kg + kgmin, jg + jgmin, 0), 0, cube->size[2] * sizeof(double));
                continue;
            }
            double *__restrict dst =  &idx3(cube[0], kg + kgmin, jg + jgmin, 0);
            int rix = ceil(-1e-8 - sqrt(max(0.0, rest)) * inv_dx);
            for (int ig = -igmin; ig <= rix; ig ++) {
                dst[ig + igmin] = 0.0;
            }
            for (int ig = -rix + 1; ig < cube->size[2]; ig ++) {
                dst[ig + igmin] = 0.0;
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
    const double dy = dh[1][1];
    const double inv_dy = dh_inv[1][1];
    const double inv_dx = dh_inv[0][0];
    const int *__restrict map_x = map[0];
    const int *__restrict map_y = map[1];
    const int *__restrict map_z = map[2];

    for (int kg = kgmin; kg <= 1 - kgmin; kg++) {
        const int k = map_z[kg + cmax];   // target location on the grid
        const int kd = (2 * kg - 1) / 2;     // distance from center in grid points
        const double kr = kd * dz;   // distance from center in a.u.
        const double kremain = disr_radius * disr_radius - kr * kr;
        if (kremain < 0.0)
            continue;
        /* const int jgmin = ceil(-1e-8 - sqrt(max(0.0, kremain)) * inv_dy); */
        for (int jg = -cmax; jg <= cmax; jg++) {
            const int j = map_y[jg + cmax];  // target location on the grid
            const int jd = (2 * jg - 1) / 2;    // distance from center in grid points
            const double jr = jd * dy;  // distance from center in a.u.
            const double jremain = kremain - jr * jr;
            if (jremain < 0.0)
                continue;
            const int igmin = ceil(-1e-8 - sqrt(max(0.0, jremain)) * inv_dx);
            double *__restrict dst = &idx3(grid[0], k - 1, j - 1, 0);
            const double *__restrict src = &idx3(cube[0], kg - lb_cube[2], jg - lb_cube[1], - lb_cube[0]);
            for (int ig = igmin; ig <= 1 - igmin; ig++) {
                const int i = map_x[ig + cmax];  // target location on the grid
                dst[i - 1] += src[ig];
            }
        }
    }
}
/*
 * collocate method where the pcb is applied on the polynomials already. For the
 * orthogonal case it might be advantageous to do it that way. The result is
 * directly sum up in the grid so that we do not need a temporary cube
 */
void colloc_ortho(const int *non_zero_elements[3],
                  const int *number_of_non_zero_elements,
                  const tensor *co,
                  const tensor *pol,
                  tensor *dst)
{
    const int *__restrict z_int = non_zero_elements[2];
    const int *__restrict y_int = non_zero_elements[1];
    const int *__restrict x_int = non_zero_elements[0];

    if ((non_zero_elements[2] == 0) || (non_zero_elements[1] == 0) || (non_zero_elements[0] == 0))
        return;

    int z_offset = 0;

    for (int z = 0; z < dst->size[0]; z++) {

        for (;(z_int[z] == 0) && (z < dst->size[0] - 1); z++);
        const int zmin = z;

        for (;(z_int[z] == 1) && (z < dst->size[0] - 1); z++);
        const int zmax = z + z_int[z];

        if (zmax - zmin) {
            int y_offset = 0;
            for (int y = 0; y < dst->size[0]; y++) {

                /* if (y_int[y] == 0) */
                /*     continue; */
                /* continue itterate */
                for (;!y_int[y] && (y < dst->size[1] - 1); y++);

                const int ymin = y;

                /* continue itterate */
                for (;y_int[y] && (y < dst->size[1] - 1); y++);

                const int ymax = y + y_int[y];

                y = ymax;
                if (ymax - ymin) {
                    int x_offset = 0;
                    for (int x = 0; x < dst->size[2]; x++) {

                        /* if (x_int[x] == 0) */
                        /*     continue; */
                        /* continue itterate */
                        for (;!x_int[x] && (x < dst->size[1] - 1); x++);


                        const int xmin = x;


                        /* continue itterate */
                        for (;x_int[x] && (x < dst->size[1] - 1); x++);

                        const int xmax = x + x_int[x];
                        const int y_bound = ymax - ymin;
                        const int x_bound = xmax - xmin;

                        x = xmax;
                        if (xmax - xmin) {
                            collocate_core_rectangular_variant1(NULL,
                                                                zmin,
                                                                ymin,
                                                                xmin,
                                                                zmax,
                                                                ymax,
                                                                xmax,
                                                                z_offset,
                                                                y_offset,
                                                                x_offset,
                                                                co,
                                                                pol,
                                                                dst);
                            x_offset += (xmax - xmin);
                        }
                    }
                    y_offset += (ymax - ymin);
                }
            }
            z_offset += (zmax - zmin);
        }
    }
}

/* just apply the mapping assuming the collocate is already stored in the cube grid. */

void apply_mapping_ortho(const int *non_zero_elements[3],
                         const int *number_of_non_zero_elements,
                         const tensor *src,
                         tensor *dst)
{
    const int *__restrict z_int = non_zero_elements[2];
    const int *__restrict y_int = non_zero_elements[1];
    const int *__restrict x_int = non_zero_elements[0];

    if ((non_zero_elements[2] == 0) || (non_zero_elements[1] == 0) || (non_zero_elements[0] == 0))
        return;

    if ((number_of_non_zero_elements[2] == dst->size[0]) || (number_of_non_zero_elements[1] == dst->size[1]) || (number_of_non_zero_elements[0] == dst->size[2])) {
#if defined(__MKL) || defined(HAVE_CBLAS)
        cblas_daxpy(dst->alloc_size_, 1.0, src->data, 1, dst->data, 1);
#else
        double ONE = 1.0;
        int ione = 1;
        daxpy_(dst->alloc_size_, &ONE, src->data, &ione, dst->data, &ione);
#endif
        return;
    }

    int z_offset = 0;

    for (int z = 0; z < dst->size[0]; z++) {

        for (;!z_int[z] && (z < dst->size[0] - 1); z++);

        const int zmin = z;

        for (;z_int[z] && (z < dst->size[0] - 1); z++);

        const int zmax = z + z_int[z];

        if (zmax - zmin) {
            int y_offset = 0;
            for (int y = 0; y < dst->size[0]; y++) {

                /* if (y_int[y] == 0) */
                /*     continue; */
                /* continue itterate */
                for (;!y_int[y] && (y < dst->size[1] - 1); y++);

                const int ymin = y;

                /* continue itterate */
                for (;y_int[y] && (y < dst->size[1] - 1); y++);

                const int ymax = y + y_int[y];

                y = ymax;
                if (ymax - ymin) {

                    int x_offset = 0;
                    for (int x = 0; x < dst->size[2]; x++) {

                        /* if (x_int[x] == 0) */
                        /*     continue; */
                        /* continue itterate */
                        for (;!x_int[x] && (x < dst->size[1] - 1); x++);


                        const int xmin = x;

                        /* continue itterate */
                        for (;x_int[x] && (x < dst->size[1] - 1); x++);

                        const int xmax = x + x_int[x];
                        const int y_bound = ymax - ymin;
                        const int x_bound = xmax - xmin;

                        if (xmax - xmin) {
                            x = xmax;
                            const int lower_corner[3] = {zmin, ymin, xmin};
                            const int upper_corner[3] = {zmax, ymax, xmax};
                            const int position[3] = {z_offset, y_offset, x_offset};

                            add_sub_grid(lower_corner, // lower corner position where the subgrid should placed
                                         upper_corner, // upper boundary
                                         position, // starting position of in the subgrid
                                         src, // subgrid
                                         dst);

                            x_offset += (xmax - xmin);
                        }
                    }
                    y_offset += (ymax - ymin);
                }
            }
            z_offset += (zmax - zmin);
        }
    }
}
