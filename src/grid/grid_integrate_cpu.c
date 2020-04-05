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
#include "tensor_local.h"
#include "utils.h"
#include "coefficients.h"
#include "thpool.h"

extern void collocate_core_rectangular_variant2(double *scratch,
                                                const struct tensor_ *co,
                                                const struct tensor_ *p_alpha_beta_reduced_,
                                                struct tensor_ *cube);

void integrate_cubic(const double radius,
                     const double dh[3][3],
                     const int *lower_boundaries_cube,
                     const int *cube_center,
                     const int *cube_size,
                     const int *period,
                     const tensor *pol_,
                     const int *lb_grid,
                     const tensor *grid,
                     tensor *coef)
{
    const int startx = (lb_grid[0] + cube_center[2] + lower_boundaries_cube[2] + 32 * period[0]) % period[0];
    const int starty = (lb_grid[1] + cube_center[1] + lower_boundaries_cube[1] + 32 * period[1]) % period[1];
    const int startz = (lb_grid[2] + cube_center[0] + lower_boundaries_cube[0] + 32 * period[2]) % period[2];

    tensor cube;
    initialize_tensor_3(&cube, cube_size[0], cube_size[1], cube_size[2]);
#if defined(__LIBXSMM)
    cube.data = libxsmm_aligned_scratch(sizeof(double) * cube.alloc_size_, 0/*auto-alignment*/);
#else
#error need implementation
#endif

    if ((starty + cube_size[1] <= grid->size[1]) &&
        (startx + cube_size[2] <= grid->size[2]) &&
        (startz + cube_size[0] <= grid->size[0])) {


        // it means that the cube is completely inside the grid without touching
        // the grid borders. periodic boundaries conditions are pointless here.
        // we can simply loop over all three dimensions.
        const int lower_corner[3] = {startz, starty, startx};
        const int upper_corner[3] = {startz + cube_size[0], starty + cube_size[1], startx + cube_size[2]};
        const int position[3] = {0, 0, 0};
        extract_sub_grid(lower_corner, upper_corner, position, grid, &cube);

        return;
    }

    int z1 = startz;
    int zoffset = 0;
    for (int z = 0; (z < cube_size[0]); z++, z1++) {
        const int zmin = z1;
        for (;((z1 < grid->size[0]) || (z1 < period[2])) && (z < cube_size[0]); z1++, z++);
        const int zmax = z1;

        if (zmax - zmin) {
            int y1 = starty;
            int yoffset = 0;
            for (int y = 0; (y < cube_size[1]); y++, y1++) {
                const int ymin = y1;
                for (;((y1 < grid->size[1]) || (y1 < period[1])) && (y < cube_size[1]); y1++, y++);
                const int ymax = y1;

                if (ymax - ymin) {
                    int x1 = startx;
                    int xoffset = 0;
                    for (int x = 0; (x < cube_size[2]); x++, x1++) {
                        const int xmin = x1;
                        for (;((x1 < grid->size[2]) || (x1 < period[0])) && (x < cube_size[2]); x1++, x++);
                        const int xmax = x1;

                        if (xmax - xmin) {
                            const int lower_corner[3] = {zmin, ymin, xmin};
                            const int upper_corner[3] = {zmax, ymax, xmax};
                            const int position[3] = {zoffset, yoffset, xoffset};
                            extract_sub_grid(lower_corner,
                                             upper_corner,
                                             position,
                                             grid,
                                             &cube);
                        }
                        xoffset += xmax - xmin;
                        if (x1 == period[0])
                            x1 = -1;
                    }
                }
                if (y1 == period[1])
                    y1 = -1;
                yoffset += ymax - ymin;
            }
        }

        if(z1 == period[2])
            z1 = -1;
        zoffset += zmax - zmin;
    }

    // then i have to shuffle thing around a bit to use the collocate routine I
    // have the cube in the format cube_kji, but in collocate the coefficients
    // should be coef_{\gamma\alpha\beta} meaning that the fastest indice of
    // cube_{kji} will become the middle indice in the end result.

    // it also mean that I have to permute the polynomials. i becomes y, k becomes z, and j - becomes x

    collocate_core_rectangular_variant2(NULL, // will need to change that eventually.
                                        // pointer to scratch memory
                                        &cube,
                                        pol_,
                                        coef);

    // now I have the coefficients in the form coef_{\beta, \alpha, \gamma}

    // I need to transpose them
#if defined(__LIBXSMM)
    libxsmm_free(cube.data);
#endif
}
