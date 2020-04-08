#ifndef UTILS_H_
#define UTILS_H_

#include <stdbool.h>
// *****************************************************************************
#define min(x, y) ( ((x) < (y)) ? x : y )

// *****************************************************************************
#define max(x, y) ( ((x) > (y)) ? x : y )

// *****************************************************************************
#define mod(a, m)  ( ((a)%(m) + (m)) % (m) )


#include "tensor_local.h"

extern void find_interval(const int start, const int end, const int *non_zero_elements_, int *zmin, int *zmax);

extern int multinomial3(const int a, const int b, const int c);

extern int return_exponents(const int index);

extern void apply_non_orthorombic_corrections(const tensor *const Exp,
                                              tensor *const cube);

extern void calculate_non_orthorombic_corrections_tensor(const double mu_mean,
                                                         const double *r_ab,
                                                         const double basis[3][3],
                                                         const int *const size,
                                                         tensor *const Exp);

inline int return_length_l(const int l) {
    static const int length_[] = {1, 4, 7, 11, 16, 22, 29, 37, 46, 56, 67, 79, 92, 106, 121, 137, 154, \
                                  172, 191, 211, 232};
    return length_[l];
}

inline int return_offset_l(const int l) {
    static const int offset_[] = {1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680, \
                                  816, 969, 1140, 1330, 1540, 1771, 2024};
    return offset_[l];
}

inline void update_loop_index(const int xmin, const int xmax, const int local_grid_size, const int global_grid_size, int *const x_offset, int *const x, int *const x1)
{
    *x += xmax - xmin - 1;
    *x_offset += (xmax - xmin);

    if (*x1 == local_grid_size) {
        *x1 = -1;
        /* case we have periodic boundaries conditions and the  grid is divided */
        if (local_grid_size != global_grid_size) {
            *x += global_grid_size;
            *x_offset += global_grid_size;
        }
    }
}

inline void compute_next_boundaries(int y1, int y, const int local_grid_size, const int global_grid_size, const int cube_size, int *const ymin, int *const ymax)
{
    *ymin = y1;

    for (int y2 = y; ((y1 < local_grid_size) ||
                      (y1 < global_grid_size)) &&
             (y2 < cube_size);
         y1++, y2++);

    *ymax = y1;
}


inline int return_linear_index_from_exponents(const int alpha, const int beta,
                                              const int gamma) {
    const int l = alpha + beta + gamma;
    return return_offset_l(l) + (l - alpha) * (l - alpha + 1) / 2 + gamma;
}


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

extern int compute_cube_properties(const bool ortho,
                                   const double radius,
                                   const double dh[3][3],
                                   const double dh_inv[3][3],
                                   const double *rp,
                                   double *disr_radius,
                                   double *roffset,
                                   int *cubecenter,
                                   int *lb_cube,
                                   int *ub_cube,
                                   int *cube_size);
extern void  return_cube_position(const int *__restrict__ grid_size, const int *__restrict__ lb_grid, const int *__restrict__ cube_center, const int *__restrict__ lower_boundaries_cube, const int *__restrict__period, int *__restrict__ const position);
extern void add_sub_grid_with_pcb(const int *period,
                                  const int *lower_corner,
                                  const int *upper_corner,
                                  const int *position,
                                  const tensor *subgrid,
                                  tensor *grid);

#endif
