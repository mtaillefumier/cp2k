#ifndef UTILS_H_
#define UTILS_H_

// *****************************************************************************
#define min(x, y) ( ((x) < (y)) ? x : y )

// *****************************************************************************
#define max(x, y) ( ((x) > (y)) ? x : y )

// *****************************************************************************
#define mod(a, m)  ( ((a)%(m) + (m)) % (m) )


#include "tensor_local.h"

extern void find_interval(const int start, const int end, const int *non_zero_elements_, int *zmin, int *zmax);

extern int return_length_l(const int l);
extern int return_offset_l(const int l);

extern int return_linear_index_from_exponents(const int alpha, const int beta,
                                              const int gamma);

extern void extract_sub_grid(const int *lower_corner,
                             const int *upper_corner,
                             const int *position,
                             const tensor *grid,
                             const tensor *subgrid);

extern void add_sub_grid(const int *lower_corner,
                         const int *upper_corner,
                         const int *position,
                         const tensor *subgrid,
                         tensor *grid);

extern int compute_cube_properties(const double radius,
                                   const double dh[3][3],
                                   const double dh_inv[3][3],
                                   const double *rp,
                                   double *disr_radius,
                                   double *roffset,
                                   int *cubecenter,
                                   int *lb_cube,
                                   int *ub_cube,
                                   int *cube_size);
extern void  return_cube_position(const int *__restrict__ lb_grid, const int *__restrict__ cube_center, const int *__restrict__ lower_boundaries_cube, const int *__restrict__period, int *__restrict__ const position);
#endif
