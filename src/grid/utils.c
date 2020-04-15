#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#if defined(__MKL) || defined(HAVE_MKL)
#include <mkl.h>
#include <mkl_cblas.h>
#endif

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "utils.h"
#include "tensor_local.h"
void extract_sub_grid(const int *lower_corner,
                      const int *upper_corner,
                      const int *position,
                      const tensor *grid,
                      const tensor *subgrid)
{
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
            printf("      : src grid size  [%d] = %d\n", d, grid->size[d]);
            printf("      : dst grid size  [%d] = %d\n", d, subgrid->size[d]);
            abort();
        }
    }

    int position1[3] = {0, 0, 0};

    if (position) {
        position1[0] = position[0];
        position1[1] = position[1];
        position1[2] = position[2];
    }

    const int sizex = upper_corner[2] - lower_corner[2];
    const int sizey = upper_corner[1] - lower_corner[1];
    const int sizez = upper_corner[0] - lower_corner[0];

#if defined(__LIBXSMM)
    const libxsmm_mcopy_descriptor* desc;
    libxsmm_xmcopyfunction kernel;
    libxsmm_descriptor_blob blob;
    int prefetch = 1;
    desc = libxsmm_mcopy_descriptor_init(&blob, sizeof(double),
                                         sizex,
                                         sizey,
                                         subgrid->ld_,
                                         grid->ld_,
                                         LIBXSMM_MATCOPY_FLAG_DEFAULT,
                                         LIBXSMM_PREFETCH_AUTO,
                                         NULL);

    kernel = libxsmm_dispatch_mcopy(desc);
    if (kernel == 0) {
        printf("JIT error -> exit!!!!\n");
        exit(EXIT_FAILURE);
    }
#endif
    for (int z = 0; z < sizez; z++) {
#if defined(__LIBXSMM)
        kernel(&idx3(grid[0], lower_corner[0] + z, lower_corner[1], lower_corner[2]),
               &grid->ld_,
               &idx3(subgrid[0], position1[0] + z, position1[1], position1[2]),
               &subgrid->ld_,
               &prefetch);
#elif defined(__MKL)
        mkl_domatcopy(CblasRowMajor,
                      CblasNoTrans,
                      sizey,
                      sizex,
                      1.0,
                      &idx3(grid[0], lower_corner[0] + z, lower_corner[1], lower_corner[2]),
                      grid->ld_,
                      &idx3(subgrid[0], position1[0] + z, position1[1], position1[2]),
                      subgrid->ld_);
#else
        for (int y = 0; y < sizey; y++) {
            memcpy(&idx3(subgrid[0], position1[0] + z, position1[1] + y, position1[2]),
                   &idx3(grid[0], lower_corner[0] + z, lower_corner[1] + y, lower_corner[2]),
                   sizeof(double) * sizex);
        }
#endif
    }

#if defined(__LIBXSMM)
    libxsmm_release_kernel(kernel);
#endif
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

/* #if defined (__MKL) */
/*     double *tmp = NULL; */
/*     posix_memalign(&tmp, 32, sizeof(double) * sizey * sizex); */
/* #endif */

/* #if defined (__MKL) */
/*     for (int z = 0; z < sizez; z++) { */
/*         double *__restrict__ dst = &idx3(grid[0], lower_corner[0] + z, lower_corner[1], lower_corner[2]); */
/*         double *__restrict__ src = &idx3(subgrid[0], position1[0] + z, position1[1], position1[2]); */

/*         /\* for (int y = 0; y < sizey; y++) *\/ */
/*         /\*     memcpy(tmp + y * sizex, *\/ */
/*         /\*            &idx3(grid[0], lower_corner[0] + z, lower_corner[1] + y, lower_corner[2]), *\/ */
/*         /\*            sizeof(double) * sizex); *\/ */

/*         mkl_domatadd ('r', */
/*                       'N', */
/*                       'N', */
/*                       sizey, */
/*                       sizex, */
/*                       1.0, */
/*                       src, */
/*                       subgrid->ld_, */
/*                       1.0, */
/*                       dst, */
/*                       grid->ld_, */
/*                       dst, */
/*                       grid->ld_); */
/*     } */
/* #else */

    for (int z = 0; z < sizez; z++) {
        double *__restrict__ dst = &idx3(grid[0], lower_corner[0] + z, lower_corner[1], lower_corner[2]);
        double *__restrict__ src = &idx3(subgrid[0], position1[0] + z, position1[1], position1[2]);
        for (int y = 0; y < sizey; y++) {
#pragma GCC unroll 4
#pragma GCC ivdep
            for (int x = 0; x < sizex; x++) {
                dst[x] += src[x];
            }
            dst += grid->ld_;
            src += subgrid->ld_;
        }
    }
/* #endif */

/* #if defined(__MKL) */
/*     free(tmp); */
/* #endif */
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

    /* it is in the lattice vector frame */
    for (int i=0; i<3; i++) {
        double dh_inv_rp = 0.0;
        for (int j=0; j<3; j++) {
            dh_inv_rp += dh_inv[j][i] * rp[j];
        }
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
        *disr_radius = drmin * max(1, ceil(radius/drmin));

        for (int i = 0; i < 3; i++) {
            lb_cube[i] = ceil(-1e-8 - *disr_radius * dx_inv[i]);
            ub_cube[i] = 1 - lb_cube[i];
        }

        /* compute the cube size ignoring periodicity */
        cube_size[0] = ub_cube[0] - lb_cube[0] + 1;
        cube_size[1] = ub_cube[1] - lb_cube[1] + 1;
        cube_size[2] = ub_cube[2] - lb_cube[2] + 1;

        for (int i=0; i<3; i++) {
            roffset[i] = rp[2 - i] - ((double) cubecenter[i]) * dx[i];
        }

        for (int i = 0; i < 3; i++) {
            cmax = max(cmax, ub_cube[i]);
        }

        return cmax;
    } else {
        /* A little bit of simple geometry. What it the smallest parallelepiped
           that contains the sphere of radius r centered around the origin */
        const double norm1 = sqrt(dh[0][0] * dh[0][0] + dh[0][1] * dh[0][1] + dh[0][2] * dh[0][2]);
        const double norm2 = sqrt(dh[1][0] * dh[1][0] + dh[1][1] * dh[1][1] + dh[1][2] * dh[1][2]);
        const double norm3 = sqrt(dh[2][0] * dh[2][0] + dh[2][1] * dh[2][1] + dh[2][2] * dh[2][2]);
        const double theta = acos((dh[0][0] * dh[1][0] + dh[0][1] * dh[1][1] + dh[0][2] * dh[1][2]) / (norm1 * norm2));
        const double phi = acos((dh[0][0] * dh[2][0] + dh[0][1] * dh[2][1] + dh[0][2] * dh[2][2]) / (norm1 * norm3));
        lb_cube[1] = ceil(-radius / (norm2 * sin(theta)) - 1e-8) - 1;
        lb_cube[2] = ceil(-radius / (norm1 * cos(M_PI * 0.5 - theta)) - 1e-8) - 1;
        lb_cube[0] = ceil(-radius / (norm3 * sin(phi)) - 1e-8) - 1;
        /* compute the offset now in the lattice basis */
        double rp1[3];

        /* compute the cube center */
        for (int i=0; i<3; i++) {
            double dh_inv_rp = 0.0;
            for (int j=0; j<3; j++) {
                dh_inv_rp += dh_inv[j][i] * rp[j];
            }
            rp1[2 - i] = dh_inv_rp;
        }

        double dx[3] = {norm3, norm2, norm1};
        for (int i=0; i<3; i++) {
            roffset[i] = rp1[i] - ((double) cubecenter[i]);
        }
    }

    for (int i = 0; i < 3; i++) {
        ub_cube[i] = 1 - lb_cube[i];
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

void  return_cube_position(const int *grid_size,
                           const int *lb_grid,
                           const int *cube_center,
                           const int *lower_boundaries_cube,
                           const int *period, int *const position)
{
    position[0] = (lb_grid[0] + cube_center[0] + lower_boundaries_cube[0] + 32 * period[0]) % period[0];
    position[1] = (lb_grid[1] + cube_center[1] + lower_boundaries_cube[1] + 32 * period[1]) % period[1];
    position[2] = (lb_grid[2] + cube_center[2] + lower_boundaries_cube[2] + 32 * period[2]) % period[2];

    if ((position[0] >= grid_size[0]) || (position[1] >= grid_size[1]) || (position[2] >= grid_size[2])) {
        printf("the lower corner of the cube is outside the grid\n");
        abort();
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
    const int *__restrict map_x = map[2];
    const int *__restrict map_y = map[1];
    const int *__restrict map_z = map[0];

    for (int kg = kgmin; kg <= 1 - kgmin; kg++) {
        const int k = map_z[kg + cmax];   // target location on the grid
        const int kd = (2 * kg - 1) / 2;     // distance from center in grid points
        const double kr = kd * dz;   // distance from center in a.u.
        const double kremain = disr_radius * disr_radius - kr * kr;
        const int jgmin = ceil(-1e-8 - sqrt(max(0.0, kremain)) * inv_dy);
        for (int jg = jgmin; jg <= 1 - jgmin; jg++) {
            const int j = map_y[jg + cmax];  // target location on the grid
            const int jd = (2 * jg - 1) / 2;    // distance from center in grid points
            const double jr = jd * dy;  // distance from center in a.u.
            const double jremain = kremain - jr * jr;
            const int igmin = ceil(-1e-8 - sqrt(max(0.0, jremain)) * inv_dx);
            double *__restrict dst = &idx3(grid[0], k - 1, j - 1, 0);
            const double *__restrict src = &idx3(cube[0], kg - lb_cube[0], jg - lb_cube[1], - lb_cube[2]);
            for (int ig = igmin; ig <= 1 - igmin; ig++) {
                const int i = map_x[ig + cmax];  // target location on the grid
                dst[i - 1] += src[ig];
            }
        }
    }
}


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


void compute_folded_polynomial(const int cube_center,
                               const int pol_length,
                               const int period,
                               const int grid_size,
                               const int grid_lower_boundaries,
                               const int *__restrict non_zero_elements,
                               const double *__restrict pol,
                               double *__restrict res,
                               double *__restrict tmp)
{

    const int start = (cube_center - grid_lower_boundaries - (pol_length - 1) / 2 + 32 * period) % period;

#pragma GCC unroll 4
#pragma GCC ivdep
    for (int s = start; s < min(grid_size, pol_length + start); s++) {
        tmp[s] = pol[s - start];
    }

    for (int s = min(grid_size - start, pol_length); s < pol_length; s++) {
#pragma GCC unroll 4
#pragma GCC ivdep
        for (int s1 = 0; s1 < min(grid_size, pol_length - s); s1++)
            tmp[s1] += pol[s + s1];

        s += period;
    }

    // now compress the all thing. I do not care about zeros.
    int offset = 0;

    for (int s = 0; s < grid_size; s++) {
        /* if (non_zero_elements[s] == 0) */
        /*     continue; */
        for (; (s < grid_size - 1) && (non_zero_elements[s] == 0); s++);
        int smin = s;

        for (; (s < grid_size - 1) && (non_zero_elements[s] == 1); s++);

        int smax = s + non_zero_elements[s];

        for (int si = 0; si < (smax - smin); si++)
            res[offset + si] = tmp[smin + si];
        offset += smax - smin;
    }
}

double exp_recursive(const double c_exp, const double c_exp_minus_1, const int index)
{
    if (index == 0)
        return 1.0;

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
}
void exp_i(const double alpha, const int imin, const int imax, double *__restrict__ const res)
{
    const double c_exp = exp(alpha);
    const double c_exp_minus_1 = 1/ c_exp;
    for (int i = 0; i < (imax - imin); i++) {
        res[i] = exp_recursive(c_exp, c_exp_minus_1, (i + imin));
    }
}


void exp_ij(const double alpha, const int imin, const int imax, const int jmin, const int jmax, tensor *exp_ij_)
{
    double c_exp = exp(alpha * imin);
    const double c_exp_co = exp(alpha);

    for (int i = 0; i < (imax - imin); i++) {
        double *__restrict dst = &idx2(exp_ij_[0], i, 0);
#pragma GCC ivdep
        for (int j = 0; j < (jmax - jmin); j++) {
            dst[j] *= exp_recursive(c_exp, 1.0 / c_exp, j + jmin);
        }
        c_exp *= c_exp_co;
    }
}


void calculate_non_orthorombic_corrections_tensor(const double mu_mean,
                                                  const double *r_ab,
                                                  const double basis[3][3],
                                                  const int *const xmin,
                                                  const int *const xmax,
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
    tensor exp_tmp;
    double *x1, *x2;

    initialize_tensor_2(&exp_tmp, Exp->size[1], Exp->size[2]);
    const int max_elem = max(max(xmax[0] - xmin[0], xmax[1] - xmin[1]), xmax[2] - xmin[2]) + 1;
    posix_memalign((void **)&x1, 64, sizeof(double) * max_elem);
    posix_memalign((void **)&x2, 64, sizeof(double) * max_elem);

    memset(Exp->data, 0, sizeof(double) * Exp->alloc_size_);
    for (int dir = 0; dir < 3; dir++) {
        int d1 = n[dir][0];
        int d2 = n[dir][1];
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
    }

    free(x1);
    free(x2);
}

void apply_non_orthorombic_corrections(const tensor *const Exp,
                                       tensor *const cube)
{
    for (int z = 0; z < cube->size[0]; z++) {
        double *__restrict__ zx = &idx3(Exp[0], 1, z, 0);
        for (int y = 0; y < cube->size[1]; y++) {
            const double zy = idx3(Exp[0], 0, z, y);
            const double *__restrict__ yx = &idx3(Exp[0], 2, y, 0);
            LIBXSMM_PRAGMA_SIMD
            for (int x = 0; x < cube->size[2]; x++) {
                idx3(cube[0], z, y, x) *= zx[x] * zy * yx[x];
                /* printf("Exp(%d, %d) = %.15lf\n", y, x, idx3(cube[0], z, y, x)); */
            }
            zx += Exp->ld_;
            yx += Exp->ld_;
        }
        /* printf("\n"); */
    }
    /* abort(); */
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


int multinomial3(const int a, const int b, const int c)
{
    /* multinomial (a,b,c) up to a+b+c = 10. Mathematica code to generate this is
     *
     Flatten[Table[
     Reverse[Flatten[
     Table[{ll[[l + 1]] + (l - a) (l - a + 1)/2 + l - a - b,
     Multinomial[a, b, l - a - b]}, {a, 0, l}, {b, 0, l - a}],
     1]], {l, 0, 20}], 1][[All, 2]]
     *
     */


    int multinomial[1771] = {1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 3, 3, 3, 6, 3, 1, 3, 3, 1, 1, 4, 4, \
                             6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1, 1, 5, 5, 10, 20, 10, 10, 30, \
                             30, 10, 5, 20, 30, 20, 5, 1, 5, 10, 10, 5, 1, 1, 6, 6, 15, 30, 15, \
                             20, 60, 60, 20, 15, 60, 90, 60, 15, 6, 30, 60, 60, 30, 6, 1, 6, 15, \
                             20, 15, 6, 1, 1, 7, 7, 21, 42, 21, 35, 105, 105, 35, 35, 140, 210, \
                             140, 35, 21, 105, 210, 210, 105, 21, 7, 42, 105, 140, 105, 42, 7, 1, \
                             7, 21, 35, 35, 21, 7, 1, 1, 8, 8, 28, 56, 28, 56, 168, 168, 56, 70, \
                             280, 420, 280, 70, 56, 280, 560, 560, 280, 56, 28, 168, 420, 560, \
                             420, 168, 28, 8, 56, 168, 280, 280, 168, 56, 8, 1, 8, 28, 56, 70, 56, \
                             28, 8, 1, 1, 9, 9, 36, 72, 36, 84, 252, 252, 84, 126, 504, 756, 504, \
                             126, 126, 630, 1260, 1260, 630, 126, 84, 504, 1260, 1680, 1260, 504, \
                             84, 36, 252, 756, 1260, 1260, 756, 252, 36, 9, 72, 252, 504, 630, \
                             504, 252, 72, 9, 1, 9, 36, 84, 126, 126, 84, 36, 9, 1, 1, 10, 10, 45, \
                             90, 45, 120, 360, 360, 120, 210, 840, 1260, 840, 210, 252, 1260, \
                             2520, 2520, 1260, 252, 210, 1260, 3150, 4200, 3150, 1260, 210, 120, \
                             840, 2520, 4200, 4200, 2520, 840, 120, 45, 360, 1260, 2520, 3150, \
                             2520, 1260, 360, 45, 10, 90, 360, 840, 1260, 1260, 840, 360, 90, 10, \
                             1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1, 1, 11, 11, 55, 110, \
                             55, 165, 495, 495, 165, 330, 1320, 1980, 1320, 330, 462, 2310, 4620, \
                             4620, 2310, 462, 462, 2772, 6930, 9240, 6930, 2772, 462, 330, 2310, \
                             6930, 11550, 11550, 6930, 2310, 330, 165, 1320, 4620, 9240, 11550, \
                             9240, 4620, 1320, 165, 55, 495, 1980, 4620, 6930, 6930, 4620, 1980, \
                             495, 55, 11, 110, 495, 1320, 2310, 2772, 2310, 1320, 495, 110, 11, 1, \
                             11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1, 1, 12, 12, 66, 132, \
                             66, 220, 660, 660, 220, 495, 1980, 2970, 1980, 495, 792, 3960, 7920, \
                             7920, 3960, 792, 924, 5544, 13860, 18480, 13860, 5544, 924, 792, \
                             5544, 16632, 27720, 27720, 16632, 5544, 792, 495, 3960, 13860, 27720, \
                             34650, 27720, 13860, 3960, 495, 220, 1980, 7920, 18480, 27720, 27720, \
                             18480, 7920, 1980, 220, 66, 660, 2970, 7920, 13860, 16632, 13860, \
                             7920, 2970, 660, 66, 12, 132, 660, 1980, 3960, 5544, 5544, 3960, \
                             1980, 660, 132, 12, 1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, \
                             12, 1, 1, 13, 13, 78, 156, 78, 286, 858, 858, 286, 715, 2860, 4290, \
                             2860, 715, 1287, 6435, 12870, 12870, 6435, 1287, 1716, 10296, 25740, \
                             34320, 25740, 10296, 1716, 1716, 12012, 36036, 60060, 60060, 36036, \
                             12012, 1716, 1287, 10296, 36036, 72072, 90090, 72072, 36036, 10296, \
                             1287, 715, 6435, 25740, 60060, 90090, 90090, 60060, 25740, 6435, 715, \
                             286, 2860, 12870, 34320, 60060, 72072, 60060, 34320, 12870, 2860, \
                             286, 78, 858, 4290, 12870, 25740, 36036, 36036, 25740, 12870, 4290, \
                             858, 78, 13, 156, 858, 2860, 6435, 10296, 12012, 10296, 6435, 2860, \
                             858, 156, 13, 1, 13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, \
                             78, 13, 1, 1, 14, 14, 91, 182, 91, 364, 1092, 1092, 364, 1001, 4004, \
                             6006, 4004, 1001, 2002, 10010, 20020, 20020, 10010, 2002, 3003, \
                             18018, 45045, 60060, 45045, 18018, 3003, 3432, 24024, 72072, 120120, \
                             120120, 72072, 24024, 3432, 3003, 24024, 84084, 168168, 210210, \
                             168168, 84084, 24024, 3003, 2002, 18018, 72072, 168168, 252252, \
                             252252, 168168, 72072, 18018, 2002, 1001, 10010, 45045, 120120, \
                             210210, 252252, 210210, 120120, 45045, 10010, 1001, 364, 4004, 20020, \
                             60060, 120120, 168168, 168168, 120120, 60060, 20020, 4004, 364, 91, \
                             1092, 6006, 20020, 45045, 72072, 84084, 72072, 45045, 20020, 6006, \
                             1092, 91, 14, 182, 1092, 4004, 10010, 18018, 24024, 24024, 18018, \
                             10010, 4004, 1092, 182, 14, 1, 14, 91, 364, 1001, 2002, 3003, 3432, \
                             3003, 2002, 1001, 364, 91, 14, 1, 1, 15, 15, 105, 210, 105, 455, \
                             1365, 1365, 455, 1365, 5460, 8190, 5460, 1365, 3003, 15015, 30030, \
                             30030, 15015, 3003, 5005, 30030, 75075, 100100, 75075, 30030, 5005, \
                             6435, 45045, 135135, 225225, 225225, 135135, 45045, 6435, 6435, \
                             51480, 180180, 360360, 450450, 360360, 180180, 51480, 6435, 5005, \
                             45045, 180180, 420420, 630630, 630630, 420420, 180180, 45045, 5005, \
                             3003, 30030, 135135, 360360, 630630, 756756, 630630, 360360, 135135, \
                             30030, 3003, 1365, 15015, 75075, 225225, 450450, 630630, 630630, \
                             450450, 225225, 75075, 15015, 1365, 455, 5460, 30030, 100100, 225225, \
                             360360, 420420, 360360, 225225, 100100, 30030, 5460, 455, 105, 1365, \
                             8190, 30030, 75075, 135135, 180180, 180180, 135135, 75075, 30030, \
                             8190, 1365, 105, 15, 210, 1365, 5460, 15015, 30030, 45045, 51480, \
                             45045, 30030, 15015, 5460, 1365, 210, 15, 1, 15, 105, 455, 1365, \
                             3003, 5005, 6435, 6435, 5005, 3003, 1365, 455, 105, 15, 1, 1, 16, 16, \
                             120, 240, 120, 560, 1680, 1680, 560, 1820, 7280, 10920, 7280, 1820, \
                             4368, 21840, 43680, 43680, 21840, 4368, 8008, 48048, 120120, 160160, \
                             120120, 48048, 8008, 11440, 80080, 240240, 400400, 400400, 240240, \
                             80080, 11440, 12870, 102960, 360360, 720720, 900900, 720720, 360360, \
                             102960, 12870, 11440, 102960, 411840, 960960, 1441440, 1441440, \
                             960960, 411840, 102960, 11440, 8008, 80080, 360360, 960960, 1681680, \
                             2018016, 1681680, 960960, 360360, 80080, 8008, 4368, 48048, 240240, \
                             720720, 1441440, 2018016, 2018016, 1441440, 720720, 240240, 48048, \
                             4368, 1820, 21840, 120120, 400400, 900900, 1441440, 1681680, 1441440, \
                             900900, 400400, 120120, 21840, 1820, 560, 7280, 43680, 160160, \
                             400400, 720720, 960960, 960960, 720720, 400400, 160160, 43680, 7280, \
                             560, 120, 1680, 10920, 43680, 120120, 240240, 360360, 411840, 360360, \
                             240240, 120120, 43680, 10920, 1680, 120, 16, 240, 1680, 7280, 21840, \
                             48048, 80080, 102960, 102960, 80080, 48048, 21840, 7280, 1680, 240, \
                             16, 1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, \
                             4368, 1820, 560, 120, 16, 1, 1, 17, 17, 136, 272, 136, 680, 2040, \
                             2040, 680, 2380, 9520, 14280, 9520, 2380, 6188, 30940, 61880, 61880, \
                             30940, 6188, 12376, 74256, 185640, 247520, 185640, 74256, 12376, \
                             19448, 136136, 408408, 680680, 680680, 408408, 136136, 19448, 24310, \
                             194480, 680680, 1361360, 1701700, 1361360, 680680, 194480, 24310, \
                             24310, 218790, 875160, 2042040, 3063060, 3063060, 2042040, 875160, \
                             218790, 24310, 19448, 194480, 875160, 2333760, 4084080, 4900896, \
                             4084080, 2333760, 875160, 194480, 19448, 12376, 136136, 680680, \
                             2042040, 4084080, 5717712, 5717712, 4084080, 2042040, 680680, 136136, \
                             12376, 6188, 74256, 408408, 1361360, 3063060, 4900896, 5717712, \
                             4900896, 3063060, 1361360, 408408, 74256, 6188, 2380, 30940, 185640, \
                             680680, 1701700, 3063060, 4084080, 4084080, 3063060, 1701700, 680680, \
                             185640, 30940, 2380, 680, 9520, 61880, 247520, 680680, 1361360, \
                             2042040, 2333760, 2042040, 1361360, 680680, 247520, 61880, 9520, 680, \
                             136, 2040, 14280, 61880, 185640, 408408, 680680, 875160, 875160, \
                             680680, 408408, 185640, 61880, 14280, 2040, 136, 17, 272, 2040, 9520, \
                             30940, 74256, 136136, 194480, 218790, 194480, 136136, 74256, 30940, \
                             9520, 2040, 272, 17, 1, 17, 136, 680, 2380, 6188, 12376, 19448, \
                             24310, 24310, 19448, 12376, 6188, 2380, 680, 136, 17, 1, 1, 18, 18, \
                             153, 306, 153, 816, 2448, 2448, 816, 3060, 12240, 18360, 12240, 3060, \
                             8568, 42840, 85680, 85680, 42840, 8568, 18564, 111384, 278460, \
                             371280, 278460, 111384, 18564, 31824, 222768, 668304, 1113840, \
                             1113840, 668304, 222768, 31824, 43758, 350064, 1225224, 2450448, \
                             3063060, 2450448, 1225224, 350064, 43758, 48620, 437580, 1750320, \
                             4084080, 6126120, 6126120, 4084080, 1750320, 437580, 48620, 43758, \
                             437580, 1969110, 5250960, 9189180, 11027016, 9189180, 5250960, \
                             1969110, 437580, 43758, 31824, 350064, 1750320, 5250960, 10501920, \
                             14702688, 14702688, 10501920, 5250960, 1750320, 350064, 31824, 18564, \
                             222768, 1225224, 4084080, 9189180, 14702688, 17153136, 14702688, \
                             9189180, 4084080, 1225224, 222768, 18564, 8568, 111384, 668304, \
                             2450448, 6126120, 11027016, 14702688, 14702688, 11027016, 6126120, \
                             2450448, 668304, 111384, 8568, 3060, 42840, 278460, 1113840, 3063060, \
                             6126120, 9189180, 10501920, 9189180, 6126120, 3063060, 1113840, \
                             278460, 42840, 3060, 816, 12240, 85680, 371280, 1113840, 2450448, \
                             4084080, 5250960, 5250960, 4084080, 2450448, 1113840, 371280, 85680, \
                             12240, 816, 153, 2448, 18360, 85680, 278460, 668304, 1225224, \
                             1750320, 1969110, 1750320, 1225224, 668304, 278460, 85680, 18360, \
                             2448, 153, 18, 306, 2448, 12240, 42840, 111384, 222768, 350064, \
                             437580, 437580, 350064, 222768, 111384, 42840, 12240, 2448, 306, 18, \
                             1, 18, 153, 816, 3060, 8568, 18564, 31824, 43758, 48620, 43758, \
                             31824, 18564, 8568, 3060, 816, 153, 18, 1, 1, 19, 19, 171, 342, 171, \
                             969, 2907, 2907, 969, 3876, 15504, 23256, 15504, 3876, 11628, 58140, \
                             116280, 116280, 58140, 11628, 27132, 162792, 406980, 542640, 406980, \
                             162792, 27132, 50388, 352716, 1058148, 1763580, 1763580, 1058148, \
                             352716, 50388, 75582, 604656, 2116296, 4232592, 5290740, 4232592, \
                             2116296, 604656, 75582, 92378, 831402, 3325608, 7759752, 11639628, \
                             11639628, 7759752, 3325608, 831402, 92378, 92378, 923780, 4157010, \
                             11085360, 19399380, 23279256, 19399380, 11085360, 4157010, 923780, \
                             92378, 75582, 831402, 4157010, 12471030, 24942060, 34918884, \
                             34918884, 24942060, 12471030, 4157010, 831402, 75582, 50388, 604656, \
                             3325608, 11085360, 24942060, 39907296, 46558512, 39907296, 24942060, \
                             11085360, 3325608, 604656, 50388, 27132, 352716, 2116296, 7759752, \
                             19399380, 34918884, 46558512, 46558512, 34918884, 19399380, 7759752, \
                             2116296, 352716, 27132, 11628, 162792, 1058148, 4232592, 11639628, \
                             23279256, 34918884, 39907296, 34918884, 23279256, 11639628, 4232592, \
                             1058148, 162792, 11628, 3876, 58140, 406980, 1763580, 5290740, \
                             11639628, 19399380, 24942060, 24942060, 19399380, 11639628, 5290740, \
                             1763580, 406980, 58140, 3876, 969, 15504, 116280, 542640, 1763580, \
                             4232592, 7759752, 11085360, 12471030, 11085360, 7759752, 4232592, \
                             1763580, 542640, 116280, 15504, 969, 171, 2907, 23256, 116280, \
                             406980, 1058148, 2116296, 3325608, 4157010, 4157010, 3325608, \
                             2116296, 1058148, 406980, 116280, 23256, 2907, 171, 19, 342, 2907, \
                             15504, 58140, 162792, 352716, 604656, 831402, 923780, 831402, 604656, \
                             352716, 162792, 58140, 15504, 2907, 342, 19, 1, 19, 171, 969, 3876, \
                             11628, 27132, 50388, 75582, 92378, 92378, 75582, 50388, 27132, 11628, \
                             3876, 969, 171, 19, 1, 1, 20, 20, 190, 380, 190, 1140, 3420, 3420, \
                             1140, 4845, 19380, 29070, 19380, 4845, 15504, 77520, 155040, 155040, \
                             77520, 15504, 38760, 232560, 581400, 775200, 581400, 232560, 38760, \
                             77520, 542640, 1627920, 2713200, 2713200, 1627920, 542640, 77520, \
                             125970, 1007760, 3527160, 7054320, 8817900, 7054320, 3527160, \
                             1007760, 125970, 167960, 1511640, 6046560, 14108640, 21162960, \
                             21162960, 14108640, 6046560, 1511640, 167960, 184756, 1847560, \
                             8314020, 22170720, 38798760, 46558512, 38798760, 22170720, 8314020, \
                             1847560, 184756, 167960, 1847560, 9237800, 27713400, 55426800, \
                             77597520, 77597520, 55426800, 27713400, 9237800, 1847560, 167960, \
                             125970, 1511640, 8314020, 27713400, 62355150, 99768240, 116396280, \
                             99768240, 62355150, 27713400, 8314020, 1511640, 125970, 77520, \
                             1007760, 6046560, 22170720, 55426800, 99768240, 133024320, 133024320, \
                             99768240, 55426800, 22170720, 6046560, 1007760, 77520, 38760, 542640, \
                             3527160, 14108640, 38798760, 77597520, 116396280, 133024320, \
                             116396280, 77597520, 38798760, 14108640, 3527160, 542640, 38760, \
                             15504, 232560, 1627920, 7054320, 21162960, 46558512, 77597520, \
                             99768240, 99768240, 77597520, 46558512, 21162960, 7054320, 1627920, \
                             232560, 15504, 4845, 77520, 581400, 2713200, 8817900, 21162960, \
                             38798760, 55426800, 62355150, 55426800, 38798760, 21162960, 8817900, \
                             2713200, 581400, 77520, 4845, 1140, 19380, 155040, 775200, 2713200, \
                             7054320, 14108640, 22170720, 27713400, 27713400, 22170720, 14108640, \
                             7054320, 2713200, 775200, 155040, 19380, 1140, 190, 3420, 29070, \
                             155040, 581400, 1627920, 3527160, 6046560, 8314020, 9237800, 8314020, \
                             6046560, 3527160, 1627920, 581400, 155040, 29070, 3420, 190, 20, 380, \
                             3420, 19380, 77520, 232560, 542640, 1007760, 1511640, 1847560, \
                             1847560, 1511640, 1007760, 542640, 232560, 77520, 19380, 3420, 380, \
                             20, 1, 20, 190, 1140, 4845, 15504, 38760, 77520, 125970, 167960, \
                             184756, 167960, 125970, 77520, 38760, 15504, 4845, 1140, 190, 20, 1};

    return multinomial[return_linear_index_from_exponents(a, b, c)];
}
