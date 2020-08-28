#ifndef PRIVATE_HEADER_H
#define PRIVATE_HEADER_H

#include "../common/utils.h"
inline void
update_loop_index(const int xmin, const int xmax, const int global_grid_size, int* const x_offset, int* const x,
                  int* const x1)
{
    *x += xmax - xmin - 1;
    *x_offset += (xmax - xmin);

    if (*x1 == global_grid_size) {
        *x1 = -1;
    }
}

inline int
compute_next_boundaries(int* y1, int y, const int global_grid_size, const int cube_size)
{
    *y1 += min(cube_size - y, global_grid_size - *y1);
    return *y1;
}

extern void grid_transform_coef_jik_to_yxz(const double dh[3][3], const tensor* coef_xyz);
extern void grid_transform_coef_xzy_to_ikj(const double dh[3][3], const tensor* coef_xyz);
extern void compute_block_boundaries(const int* blockDim, const int* lb_grid, const int* grid_size,
                                     const int* blocked_grid_size, const int* period, const int* cube_center,
                                     const int* cube_size, const int* lower_boundaries_cube, int* lower_block_corner,
                                     int* upper_block_corner, int* pol_offsets);

extern void grid_fill_pol_dgemm(const bool transpose, const double dr, const double roffset, const int pol_offset,
                                const int xmin, const int xmax, const int lp, const int cmax, const double zetp,
                                double* pol_);

#endif
