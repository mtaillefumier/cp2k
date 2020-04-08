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
                          const int lb_cube,
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

extern void collocate_core_rectangular(double *scratch,
                                       double prefactor,
                                       const struct tensor_ *co,
                                       const struct tensor_ *p_alpha_beta_reduced_,
                                       struct tensor_ *cube);

void grid_integrate_ortho(collocation_integration *const handler,
                          const double zetp,
                          const double dh[3][3],
                          const double dh_inv[3][3],
                          const double rp[3],
                          const int npts[3],
                          const int lb_grid[3],
                          const bool periodic[3],
                          const double radius,
                          const tensor *const grid,
                          tensor *const coef)
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
    int cmax = compute_cube_properties(true,
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

    /* initialize the multidimensional array containing the polynomials */
    /* I need to have the polynomials transposed */

    initialize_tensor_3(&handler->pol, 3, 2 * cmax + 1, handler->coef.size[0]);
    handler->pol_alloc_size = realloc_tensor(handler->pol.data, handler->pol_alloc_size, handler->pol.alloc_size_,  (void **)&handler->pol.data);

    /* allocate memory for the polynomial and the cube */

    if (handler->sequential_mode) {
        initialize_tensor_3(&handler->cube,
                            cube_size[0],
                            cube_size[1],
                            cube_size[2]);

        handler->cube_alloc_size = realloc_tensor(handler->cube.data, handler->cube_alloc_size, handler->cube.alloc_size_, (void **)&handler->cube.data);

        size_t tmp1 = max(handler->T_alloc_size,
                                 compute_memory_space_tensor_3(handler->cube.size[2] /* alpha */,
                                                               handler->cube.size[0] /* gamma */,
                                                               handler->coef.size[1] /* j */));
        size_t tmp2 = max(handler->W_alloc_size,
                          compute_memory_space_tensor_3(handler->cube.size[0] /* gamma */ ,
                                                        handler->coef.size[1] /* j */,
                                                        handler->coef.size[2] /* i */));

        if (((tmp1 + tmp2) > (handler->T_alloc_size + handler->W_alloc_size)) ||
            (handler->scratch == NULL)) {
            handler->T_alloc_size = tmp1;
            handler->W_alloc_size = tmp2;
            if (handler->scratch)
                free(handler->scratch);
            if (posix_memalign(&handler->scratch, 32, sizeof(double) * (tmp1 + tmp2)) != 0)
                abort();
        }
    }

    // WARNING : do not reverse the order in pol otherwise you will have to
    // reverse the order in collocate_dgemm as well.

    grid_fill_pol(true, dh[0][0], roffset[2], lb_cube[2], handler->coef.size[2] - 1, cmax, zetp, &idx3(handler->pol, 1, 0, 0)); /* i indice */
    grid_fill_pol(true, dh[1][1], roffset[1], lb_cube[1], handler->coef.size[1] - 1, cmax, zetp, &idx3(handler->pol, 0, 0, 0)); /* j indice */
    grid_fill_pol(true, dh[2][2], roffset[0], lb_cube[0], handler->coef.size[0] - 1, cmax, zetp, &idx3(handler->pol, 2, 0, 0)); /* k indice */

    /* if (handler->sequential_mode) { */

    extract_cube(lb_cube, cubecenter, npts, grid, lb_grid, &handler->cube);

    collocate_core_rectangular(handler->scratch,
                               // pointer to scratch memory
                               1.0,
                               &handler->pol,
                               &handler->cube,
                               &handler->coef);

    /* the result is Coef_{beta,alpha,gamma} */


/* }  else { */
        /*     compute_blocks(handler, */
        /*                    lb_cube, */
        /*                    cube_size, */
        /*                    cubecenter, */
        /*                    npts, */
        /*                    NULL, */
        /*                    lb_grid, */
        /*                    grid); */
        /* } */
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


    int offset[3];
    int loop_number[3];
    int reminder[3];

    for (int d = 0; d < 3; d++) {
        offset[d] = min(grid->size[d] - position[d], cube->size[d]);
        loop_number[d] = (cube->size[d] - offset[d]) / period[d];
        reminder[d] = min(grid->size[d], cube->size[d] - offset[d] - loop_number[d] * period[d]);
    }

    int z1 = position[0];
    int z_offset = 0;
    for (int z = 0; (z < cube->size[0]); z++, z1++) {

        const int zmin = z1;
        for (int z2 = z;((z1 < grid->size[0]) || (z1 < period[0])) && (z2 < cube->size[0]); z1++, z2++);
        const int zmax = z1 /* + (z1 == (grid->size[0] - 1)) */;

        /* // We have a full plane. */
        if (zmax - zmin) {
            int y1 = position[1];
            int y_offset = 0;
            for (int y = 0; y < cube->size[1]; y1++, y++) {
                const int ymin = y1;

                for (int y2 = y;((y1 < grid->size[1]) ||
                                 (y1 < period[1])) &&
                         (y2 < cube->size[1]);
                     y1++, y2++);

                const int ymax = y1;

                if (ymax - ymin) {

                    if ((zmin > grid->size[0]) || (zmax > grid->size[0]) || (ymin > grid->size[1]) || (ymax > grid->size[1])) {
                        printf("Problem with the subblock boundaries. Some of them are outside the grid\n");
                        printf("Grid size     : %d %d %d\n", grid->size[0], grid->size[1], grid->size[2]);
                        printf("Grid lb_grid  : %d %d %d\n", lb_grid[0], lb_grid[1], lb_grid[2]);
                        printf("zmin-zmax     : %d %d\n", zmin, zmax);
                        printf("ymin-ymax     : %d %d\n", ymin, ymax);
                        printf("Cube position : %d %d %d\n", position[0], position[1], position[2]);
                        abort();
                    }

                    int x1 = position[2];
                    int x_offset = 0;

                    for (int x = 0; x < cube->size[2]; x++, x1++) {

                        const int xmin = x1;
                        for (int x2 = x;
                             ((x1 < grid->size[2]) || (x1 < period[2]))
                                 &&
                                 (x2 < cube->size[2]);
                             x1++, x2++);
                        const int xmax = x1;

                        if (xmax - xmin) {
                            const int lower_corner[3] = {zmin,
                                                         ymin,
                                                         xmin};

                            const int upper_corner[3] = {zmax,
                                                         ymax,
                                                         xmax};

                            int position2[3]= {z_offset, y_offset, x_offset};

                            extract_sub_grid(lower_corner,
                                             upper_corner,
                                             position2, // starting position in the subgrid
                                             grid,
                                             cube);

                        }
                        x_offset += xmax - xmin;
                        x += xmax - xmin - 1;
                        if (x1 == grid->size[2])
                            x1 = -1;
                    }
                    if (y1 == grid->size[1])
                        y1 = -1;

                    y_offset += (ymax - ymin);
                    y += ymax - ymin - 1;
                }
            }
            if (z1 == grid->size[0])
                z1 = -1;
            z_offset += (zmax - zmin);
            z += zmax - zmin - 1;
        }
    }
}
