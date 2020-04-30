#ifndef UTILS_H_
#define UTILS_H_

#include <stdbool.h>
#include "tensor_local.h"

static const int ncoset[] = {1,  // l=0
                             4,  // l=1
                             10, // l=2 ...
                             20, 35, 56, 84, 120, 165, 220, 286, 364,
                             455, 560, 680, 816, 969, 1140, 1330};

/* Table giving the factorial up to 31! */
static const double fac[] = {
        0.10000000000000000000E+01, 0.10000000000000000000E+01, 0.20000000000000000000E+01,
        0.60000000000000000000E+01, 0.24000000000000000000E+02, 0.12000000000000000000E+03,
        0.72000000000000000000E+03, 0.50400000000000000000E+04, 0.40320000000000000000E+05,
        0.36288000000000000000E+06, 0.36288000000000000000E+07, 0.39916800000000000000E+08,
        0.47900160000000000000E+09, 0.62270208000000000000E+10, 0.87178291200000000000E+11,
        0.13076743680000000000E+13, 0.20922789888000000000E+14, 0.35568742809600000000E+15,
        0.64023737057280000000E+16, 0.12164510040883200000E+18, 0.24329020081766400000E+19,
        0.51090942171709440000E+20, 0.11240007277776076800E+22, 0.25852016738884976640E+23,
        0.62044840173323943936E+24, 0.15511210043330985984E+26, 0.40329146112660563558E+27,
        0.10888869450418352161E+29, 0.30488834461171386050E+30, 0.88417619937397019545E+31,
        0.26525285981219105864E+33 };

// *****************************************************************************
#define min(x, y) ( ((x) < (y)) ? x : y )

// *****************************************************************************
#define max(x, y) ( ((x) > (y)) ? x : y )

// *****************************************************************************
#define mod(a, m)  ( ((a)%(m) + (m)) % (m) )


// *****************************************************************************
// Returns zero based indices.
inline static int coset(int lx, int ly, int lz) {
    const int l = lx + ly + lz;
    if (l==0) {
        return 0;
    } else {
        return ncoset[l-1] + ((l-lx) * (l-lx+1)) /2 + lz;
    }
}


#if defined(__MKL) || defined(HAVE_MKL)
#include <mkl.h>
#include <mkl_cblas.h>
#endif

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

typedef struct dgemm_params_ {
    char storage;
    char op1;
    char op2;
#if defined(__LIBXSMM)
    libxsmm_dmmfunction kernel;
    int prefetch;
    int flags;
#endif
    double alpha;
    double beta;
    double *a, *b, *c;
    int m, n, k, lda, ldb, ldc;
} dgemm_params;

extern bool fold_polynomial(double *scratch, tensor *pol, const int axis, const int center, const int cube_size, const int lb_cube, const int lb_grid, const int grid_size, const int period,  int *const pivot);
extern void dgemm_simplified(dgemm_params *const m, const bool use_libxsmm);

extern void find_interval(const int start, const int end, const int *non_zero_elements_, int *zmin, int *zmax);

extern int multinomial3(const int a, const int b, const int c);

extern int return_exponents(const int index);

extern void apply_non_orthorombic_corrections(const bool *__restrict plane,
                                              const tensor *const Exp,
                                              tensor *const cube);

extern void calculate_non_orthorombic_corrections_tensor(const double mu_mean,
                                                         const double *r_ab,
                                                         const double basis[3][3],
                                                         const int *const xmin,
                                                         const int *const xmax,
                                                         bool *plane,
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

inline int compute_next_boundaries(int *y1, int y, const int local_grid_size, const int global_grid_size, const int cube_size)
{
    *y1 += min(cube_size - y, min(local_grid_size, global_grid_size) - *y1);
    return *y1;
}


inline int return_linear_index_from_exponents(const int alpha, const int beta,
                                              const int gamma) {
    const int l = alpha + beta + gamma;
    return return_offset_l(l) + (l - alpha) * (l - alpha + 1) / 2 + gamma;
}


extern void extract_sub_grid(const int *lower_corner,
                             const int *upper_corner,
                             const int *position,
                             const tensor *const grid,
                             tensor *const subgrid);

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
extern void grid_transform_coef_jik_to_yxz(const double dh[3][3],
                                           const tensor *coef_xyz);
extern void grid_transform_coef_xzy_to_ikj(const double dh[3][3],
                                           const tensor *coef_xyz);
extern void batched_dgemm_simplified(dgemm_params *const m,
                                     const int batch_size,
                                     const bool use_libxsmm);

void compute_block_boundaries(const int *blockDim,
                              const int *lb_grid,
                              const int *grid_size,
                              const int *blocked_grid_size,
                              const int *period,
                              const int *cube_center,
                              const int *cube_size,
                              const int *lower_boundaries_cube,
                              int *lower_block_corner,
                              int *upper_block_corner,
                              int *pol_offsets,
                              bool *fold);

extern void grid_fill_pol(const bool transpose,
                          const double dr,
                          const double roffset,
                          const int pol_offset,
                          const int xmin,
                          const int xmax,
                          const int lp,
                          const int cmax,
                          const double zetp,
                          double *pol_);

#endif
