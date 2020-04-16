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

extern void collocate_core_rectangular(double *scratch,
                                       double prefactor,
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

void grid_integrate(collocation_integration *const handler,
                    const bool use_ortho,
                    const double zetp,
                    const double dh[3][3],
                    const double dh_inv[3][3],
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
    initialize_tensor_3(&handler->pol, 3, handler->coef.size[0], 2 * cmax + 1);
    handler->pol_alloc_size = realloc_tensor(handler->pol.data, handler->pol_alloc_size, handler->pol.alloc_size_,  (void **)&handler->pol.data);

    /* allocate memory for the polynomial and the cube */

    if (handler->sequential_mode) {
        initialize_tensor_3(&handler->cube,
                            cube_size[0],
                            cube_size[1],
                            cube_size[2]);

        handler->cube_alloc_size = realloc_tensor(handler->cube.data,
                                                  handler->cube_alloc_size,
                                                  handler->cube.alloc_size_,
                                                  (void **)&handler->cube.data);

        size_t tmp1 = max(handler->T_alloc_size,
                                 compute_memory_space_tensor_3(handler->coef.size[0] /* alpha */,
                                                               handler->coef.size[1] /* gamma */,
                                                               cube_size[1] /* j */));

        size_t tmp2 = max(handler->W_alloc_size,
                          compute_memory_space_tensor_3(handler->coef.size[1] /* gamma */ ,
                                                        cube_size[1] /* j */,
                                                        cube_size[2] /* i */));

        if (((tmp1 + tmp2) > (handler->T_alloc_size + handler->W_alloc_size)) ||
            (handler->scratch == NULL)) {
            handler->T_alloc_size = tmp1;
            handler->W_alloc_size = tmp2;
            if (handler->scratch)
                free(handler->scratch);
            if (posix_memalign(&handler->scratch, 64, sizeof(double) * (tmp1 + tmp2)) != 0)
                abort();
        }
    }

    /* compute the polynomials */

    // WARNING : do not reverse the order in pol otherwise you will have to
    // reverse the order in collocate_dgemm as well.


    if (use_ortho) {
        grid_fill_pol(true, dh[0][0], roffset[2], lb_cube[2], ub_cube[2], handler->coef.size[2] - 1, cmax, zetp, &idx3(handler->pol, 2, 0, 0)); /* i indice */
        grid_fill_pol(true, dh[1][1], roffset[1], lb_cube[1], ub_cube[1], handler->coef.size[1] - 1, cmax, zetp, &idx3(handler->pol, 1, 0, 0)); /* j indice */
        grid_fill_pol(true, dh[2][2], roffset[0], lb_cube[0], ub_cube[0], handler->coef.size[0] - 1, cmax, zetp, &idx3(handler->pol, 0, 0, 0)); /* k indice */
    } else {
        initialize_tensor_3(&handler->Exp, 3, max(cube_size[0], cube_size[1]), max(cube_size[1], cube_size[2]));
        handler->Exp_alloc_size = realloc_tensor(handler->Exp.data, handler->Exp_alloc_size, handler->Exp.alloc_size_, (void **)&handler->Exp.data);

        double dx[3];
        dx[2] = dh[0][0] * dh[0][0] + dh[0][1] * dh[0][1] + dh[0][2] * dh[0][2];
        dx[1] = dh[1][0] * dh[1][0] + dh[1][1] * dh[1][1] + dh[1][2] * dh[1][2];
        dx[0] = dh[2][0] * dh[2][0] + dh[2][1] * dh[2][1] + dh[2][2] * dh[2][2];

        grid_fill_pol(true, 1.0, roffset[0], lb_cube[0], ub_cube[0], handler->coef.size[0] - 1, cmax, zetp * dx[0], &idx3(handler->pol, 0, 0, 0)); /* k indice */
        grid_fill_pol(true, 1.0, roffset[1], lb_cube[1], ub_cube[1], handler->coef.size[1] - 1, cmax, zetp * dx[1], &idx3(handler->pol, 1, 0, 0)); /* j indice */
        grid_fill_pol(true, 1.0, roffset[2], lb_cube[2], ub_cube[2], handler->coef.size[2] - 1, cmax, zetp * dx[2], &idx3(handler->pol, 2, 0, 0)); /* i indice */

        calculate_non_orthorombic_corrections_tensor(zetp,
                                                     roffset,
                                                     dh,
                                                     lb_cube,
                                                     ub_cube,
                                                     &handler->Exp);

    }

    if (handler->sequential_mode) {
        extract_cube(lb_cube,
                     cubecenter,
                     npts,
                     grid,
                     lb_grid,
                     &handler->cube);

        if (!use_ortho)
            apply_non_orthorombic_corrections(&handler->Exp, &handler->cube);

        collocate_core_rectangular(handler->scratch,
                                   // pointer to scratch memory
                                   1.0,
                                   &handler->cube,
                                   &handler->pol,
                                   &handler->coef);

    } else {
        compute_blocks(handler,
                       lb_cube,
                       cube_size,
                       cubecenter,
                       npts,
                       use_ortho ? NULL : &handler->Exp,
                       lb_grid,
                       grid);
    }
}
