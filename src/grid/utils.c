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

    /* cube center */
    for (int i=0; i<3; i++) {
        double dh_inv_rp = 0.0;
        for (int j=0; j<3; j++) {
            dh_inv_rp += dh_inv[j][i] * rp[j];
        }
        cubecenter[i] = floor(dh_inv_rp);
    }

    /* lower and upper bounds */

    // Historically, the radius gets discretized.
    const double drmin = min(dh[0][0], min(dh[1][1], dh[2][2]));
    *disr_radius = drmin * max(1, ceil(radius/drmin));

    for (int i = 0; i < 3; i++) {
        lb_cube[i] = ceil(-1e-8 - *disr_radius * dh_inv[i][i]);
        ub_cube[i] = 1 - lb_cube[i];
    }

    /* compute the cube size ignoring periodicity */
    cube_size[0] = ub_cube[2] - lb_cube[2] + 1;
    cube_size[1] = ub_cube[1] - lb_cube[1] + 1;
    cube_size[2] = ub_cube[0] - lb_cube[0] + 1;

    for (int i=0; i<3; i++) {
        roffset[i] = rp[i] - ((double) cubecenter[i]) * dh[i][i];
    }

    for (int i = 0; i < 3; i++) {
        cmax = max(cmax, ub_cube[i]);
    }

    return cmax;
}
