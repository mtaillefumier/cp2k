#ifndef GRID_H
#define GRID_H

#include "tensor_local.h"
struct grid_ {
    // dimension of the block
    int blockDim[3];
    // dimension of the bloc grid
    int gridDim[3];
    void *data;
    tensor blocks;
    tensor grid;
};

void create_block_grid(struct grid_ *g1, const int *grid_size, const int *block_size);
void destroy_block_grid(struct grid_ *g1);
void decompose_grid_to_block_grid(const tensor *gr, struct grid_ *block_grid);
void recompose_grid_from_block_grid(const struct grid_ *block_grid, tensor *gr);
void add_block_grid_to_grid(const struct grid_ *block_grid, tensor *gr);

#endif
