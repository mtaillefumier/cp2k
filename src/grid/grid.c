#include "grid.h"
#include "utils.h"
#include <string.h>

void create_block_grid(struct grid_ *g1, const int *grid_size, const int *block_size)
{
    if (g1 == NULL)
        abort();
    tensor tmp;
    memset(g1, 0, sizeof(struct grid_));

    initialize_tensor_3(&tmp, block_size[0], block_size[1], block_size[2]);
    initialize_tensor_3(&g1->grid, grid_size[0], grid_size[1], grid_size[2]);
    for (int d = 0; d < 3; d++) {
        g1->blockDim[d] = block_size[d];
        g1->gridDim[d] = grid_size[d] / block_size[d] + (grid_size[d] % block_size[d] != 0);
    }

    initialize_tensor_4(&g1->blocks,
                        g1->gridDim[0],
                        g1->gridDim[1],
                        g1->gridDim[2],
                        tmp.alloc_size_);

    posix_memalign(&g1->blocks.data, 32, sizeof(double) * g1->blocks.alloc_size_);
    g1->data = g1->blocks.data;
    memset(g1->data, 0, sizeof(double) * g1->blocks.alloc_size_);
}

/* free allocated memory for a given grid */
void destroy_block_grid(struct grid_ *g1)
{
    if (g1 == NULL)
        return;

    free(g1->data);
}

void decompose_grid_to_block_grid(const tensor *gr, struct grid_ *block_grid)
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

    for (int z = 0; z < block_grid->gridDim[0]; z++) {
        lower_corner[0] = z * block_grid->blockDim[0];
        upper_corner[0] = lower_corner[0] + min(gr->size[0] - lower_corner[0], block_grid->blockDim[0]);
        for (int y = 0; y < block_grid->gridDim[1]; y++) {
            lower_corner[1] = y * block_grid->blockDim[1];
            upper_corner[1] = lower_corner[1] + min(gr->size[1] - lower_corner[1], block_grid->blockDim[1]);
            for (int x = 0; x < block_grid->gridDim[2]; x++) {
                tmp.data = &idx4(block_grid->blocks, z, y, x, 0);
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

void recompose_grid_from_block_grid(const struct grid_ *block_grid, tensor *gr)
{
    tensor tmp;
    initialize_tensor_3(&tmp,
                        block_grid->blockDim[0],
                        block_grid->blockDim[1],
                        block_grid->blockDim[2]);
    int lower_corner[3], upper_corner[3];
    for (int z = 0; z < block_grid->gridDim[0]; z++) {
        lower_corner[0] = z * block_grid->blockDim[0];
        upper_corner[0] = lower_corner[0] + min(gr->size[0] - lower_corner[0], block_grid->blockDim[0]);
        for (int y = 0; y < block_grid->gridDim[1]; y++) {
            lower_corner[1] = y * block_grid->blockDim[1];
            upper_corner[1] = lower_corner[1] + min(gr->size[1] - lower_corner[1], block_grid->blockDim[1]);
            for (int x = 0; x < block_grid->gridDim[2]; x++) {
                tmp.data = &idx4(block_grid->blocks, z, y, x, 0);
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
void add_block_grid_to_grid(const struct grid_ *block_grid, tensor *gr)
{
    tensor tmp;
    initialize_tensor_3(&tmp, block_grid->blockDim[0], block_grid->blockDim[1], block_grid->blockDim[2]);
    int lower_corner[3], upper_corner[3];

    for (int z = 0; z < block_grid->gridDim[0]; z++) {
        lower_corner[0] = z * block_grid->blockDim[0];
        upper_corner[0] = lower_corner[0] + min(gr->size[0] - lower_corner[0], block_grid->blockDim[0]);
        for (int y = 0; y < block_grid->gridDim[1]; y++) {
            lower_corner[1] = y * block_grid->blockDim[1];
            upper_corner[1] = lower_corner[1] + min(gr->size[1] - lower_corner[1], block_grid->blockDim[1]);
            for (int x = 0; x < block_grid->gridDim[2]; x++) {
                tmp.data = &idx4(block_grid->blocks, z, y, x, 0);
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
                            dst[x] += src[x];
                        }
                    }
                }
            }
        }
    }
}

/* void add_block_grid_to_block_grid(const struct grid_ *src, const struct grid_ *dst) */
/* { */
/*     initialize_tensor_3(&tmp, gr->blockDim[0], gr->blockDim[1], gr->blockDim[2]); */
/*     int lower_corner[3], upper_corner[3]; */
/*     for (int z = 0; z < gr->gridDim[0]; z++) { */
/*         lower_corner[0] = z * gr->blockDim[0]; */
/*         upper_corner[0] = lower_corner[0] + min(gr->size[0] - lower_corner[0], gr->blockDim[0]); */
/*         for (int y = 0; y < gr->gridDim[1]; y++) { */
/*             lower_corner[1] = y * gr->blockDim[1]; */
/*             upper_corner[1] = lower_corner[1] + min(gr->size[1] - lower_corner[1], gr->blockDim[0]); */
/*             for (int x = 0; x < gr->gridDim[2]; x++) { */
/*                 tmp->data = &idx3(gr->blocks, z, y, x, 0); */
/*                 lower_corner[2] = y * gr->blockDim[2]; */
/*                 upper_corner[2] = lower_corner[2] + min(gr->size[2] - lower_corner[2], gr->blockDim[2]); */

/*                 const int sizex = upper_corner[2] - lower_corner[2]; */
/*                 const int sizey = upper_corner[1] - lower_corner[1]; */
/*                 const int sizez = upper_corner[0] - lower_corner[0]; */

/*                 for (int z = 0; z < sizez; z++) { */
/*                     for (int y = 0; y < sizey; y++) { */
/*                         double *__restrict__ dst = &idx3(gr[0], lower_corner[0] + z, lower_corner[1] + y, lower_corner[2]); */
/*                         double *__restrict__ src = &idx3(subgrid[0], z, y, 0); */
/*                         for (int x = 0; x < sizex; x++) { */
/*                             dst[x] += src[x]; */
/*                         } */
/*                     } */
/*                 } */
/*             } */
/*         } */
/*     } */
/* } */
