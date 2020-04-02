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

#include "utils.h"

inline void find_interval(const int start, const int end, const int *non_zero_elements_, int *zmin, int *zmax)
{
    int si;

    // loop over the table until we reach a 1
    for (si = start;(si < end - 1) && (non_zero_elements_[si] == 0); si++);

    // interval starts here
    *zmin = si;

    // now search where it ends;

    // loop over the table until we reach a 1
    for (;(si < (end - 1)) && (non_zero_elements_[si] == 1); si++);

    *zmax = si + non_zero_elements_[si];
}

inline int return_length_l(const int l) {
    static const int length_[] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55};
    return length_[l];
}

inline int return_offset_l(const int l) {
    static const int offset_[10] = {0, 1, 4, 10, 20, 35, 56, 84, 120, 165};
    return offset_[l];
}

inline int return_linear_index_from_exponents(const int alpha, const int beta,
                                              const int gamma) {
    const int l = alpha + beta + gamma;
    return return_offset_l(l) + (l - alpha) * (l - alpha + 1) / 2 + gamma;
}

inline void extract_sub_grid(const int *lower_corner,
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

    for (int z = 0; z < sizez; z++) {
        for (int y = 0; y < sizey; y++) {
            memcpy(&idx3(subgrid[0], position1[0] + z, position1[1] + y, position1[2]),
                   &idx3(grid[0], lower_corner[0] + z, lower_corner[1] + y, lower_corner[2]),
                   sizeof(double) * sizex);
        }
    }

    return;
}

inline void add_sub_grid(const int *lower_corner,
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
#pragma GCC unroll 4
#pragma GCC ivdep
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
    const int remainder = min(grid->size[2], subgrid->size[2] - offset - loop_number * period[2]);

    for (int z = 0; z < sizez; z++) {
        double *__restrict__ dst = &idx3(grid[0], lower_corner[0] + z, lower_corner[1], 0);
        double *__restrict__ src = &idx3(subgrid[0], position1[0] + z, position1[1], 0);

        for (int y = 0; y < sizey; y++) {
            //#pragma omp simd
#pragma GCC ivdep
            for (int x = 0; x < offset; x++)
                dst[x + lower_corner[2]] += src[x];

            int shift = offset;
            for (int l = 0; l < loop_number; l++) {
//#pragma omp simd
#pragma GCC ivdep
                for (int x = 0; x < grid->size[2]; x++)
                    dst[x] += src[shift + x];

                shift = offset + (l + 1)  * period[2];
            }
//#pragma omp simd
#pragma GCC ivdep
            for (int x = 0; x < remainder; x++)
                dst[x] += src[shift + x];

            dst += grid->ld_;
            src += subgrid->ld_;
        }
    }

    return;
}


inline int compute_cube_properties(const double radius,
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

    /* seting up the cube parameters */
    const double dx[3] = {dh[2][2], dh[1][1], dh[0][0]};
    const double dx_inv[3] = {dh_inv[2][2], dh_inv[1][1], dh_inv[0][0]};
    /* cube center */
    for (int i=0; i<3; i++) {
        double dh_inv_rp = 0.0;
        for (int j=0; j<3; j++) {
            dh_inv_rp += dh_inv[j][i] * rp[j];
        }
        cubecenter[2 - i] = floor(dh_inv_rp);
    }

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
}

void  return_cube_position(const int *lb_grid, const int *cube_center, const int *lower_boundaries_cube, const int *period, int *const position)
{
    position[0] = (lb_grid[0] + cube_center[0] + lower_boundaries_cube[0] + 32 * period[0]) % period[0];
    position[1] = (lb_grid[1] + cube_center[1] + lower_boundaries_cube[1] + 32 * period[1]) % period[1];
    position[2] = (lb_grid[2] + cube_center[2] + lower_boundaries_cube[2] + 32 * period[2]) % period[2];
}
