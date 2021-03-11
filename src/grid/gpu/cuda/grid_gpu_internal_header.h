/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#ifndef GRID_GPU_INTERNAL_HEADER_H
#define GRID_GPU_INTERNAL_HEADER_H

#include <algorithm>
#include <assert.h>
#include <cuda.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#if (CUDA_VERSION >= 11000)
#include <cooperative_groups/reduce.h>
#endif

#define GRID_DEVICE __device__
#define GRID_HOST_DEVICE __host__ __device__

extern "C" {
#include "../../common/grid_basis_set.h"
}

#include "../../common/grid_common.h"
#include "../headers/gpu_backend.hpp"


__constant__ orbital coset_inv[1330];
__constant__ double binomial_coef[19][19];


/*******************************************************************************
 * \brief Shared memory representation of a task.
 * \author Ole Schuett
 ******************************************************************************/
struct smem_task {
    bool block_transposed;
    double radius;
    double ra[3];
    double rb[3];
    double rp[3];
    double rab[3];
    double rab2;
    double zeta;
    double zetb;
    double zetp;
    double prefactor;
    double off_diag_twice;
    double dh_max;
    // angular momentum range of actual collocate / integrate operation
    int la_max;
    int lb_max;
    int la_min;
    int lb_min;
    int lp;
    // size of the cab matrix
    int n1;
    int n2;
    // size of entire spherical basis
    int nsgfa;
    int nsgfb;
    // size of spherical set
    int nsgf_seta;
    int nsgf_setb;
    // start of decontracted set, ie. pab and hab
    int first_coseta;
    int first_cosetb;
    // size of decontracted set, ie. pab and hab
    int ncoseta;
    int ncosetb;
    // strides of the sphi transformation matrices
    int maxcoa;
    int maxcob;
    // pointers matrices
    const double *pab_block;
    const double *sphia;
    const double *sphib;
    // integrate
    double *hab_block;
    double *forces_a;
    double *forces_b;
};

    class Matrix {
    public:
        int col_{0};
        int row_{0};
        int ld_{0};
        double *ptr;

        Matrix(double *buffer, const int row__, const int col__) {
            ptr = buffer;
            row_ = row__;
            col_ = col__;
            ld_ = col__;
        }

        Matrix(double *buffer, const int row__, const int col__, const int ld__) {
            ptr = buffer;
            row_ = row__;
            col_ = col__;
            ld_ = ld__;
        }

        ~Matrix() {
        }

        double operator() (int i, int j) {
            if ((i >= row_) || (j >= col_))
                return 0.0;
            return ptr[i * ld_ + j];
        }
        void set_element(int i, int j, double val) {
            if ((i >= row_) || (j >= col_))
                return;
            ptr[i * ld_ + j] = val;
        }
    };


    __device__ __inline__ void dgemm(const char Op1, const char Op2, Matrix &A, Matrix &B, Matrix &C)
    {
        cg::thread_block block = cg::this_thread_block();
        // Block row and column
        int NumblockRow = (A.row_ / 8) + (A.row_ % 8) != 0;
        int NumblockCol = (B.col_ / 8) + (B.col_ % 8) != 0;

        if (Op1 == 'T')
            NumblockRow = (A.col_ / 8) + (A.col_ % 8) != 0;

        if (Op2 == 'T')
            NumblockCol = (B.row_ / 8) + (B.row_ % 8) != 0;

        for (int blockRow = 0; blockRow < NumblockRow; blockRow++) {
            for (int blockCol = 0; blockCol < NumblockCol; blockCol++) {

                // Thread row and column within Csub
                int row = block.thread_rank() / 8;
                int col = block.thread_rank() % 8;
                __shared__ double As[8][8];
                __shared__ double Bs[8][8];
                // Loop over all the sub-matrices of A and B that are
                // required to compute Csub
                // Multiply each pair of sub-matrices together
                // and accumulate the results
                int block_k = (A.col_ / 8) + ((A.col_ % 8) != 0);
                if (Op1 == 'T') {
                    block_k = (A.row_ / 8) + ((A.row_ % 8) != 0);
                }

                double Cvalue = 0.0;
                for (int m = 0; m < block_k; ++m) {

                    // Get sub-matrix Asub of A
                    if (Op1 == 'T')
                        As[row][col] = A(m * 8 + col, blockRow * 8 + row);
                    else
                        As[row][col] = A(blockRow * 8 + row, m * 8 + col);

                    if (Op2 == 'T')
                        Bs[col][row] = B(blockCol * 8 + col, m * 8 + row);
                    else
                        Bs[col][row] = B(m * 8 + row, blockCol * 8 + col);

                    block.sync();


                    // Multiply Asub and Bsub together
                    for (int e = 0; e < 8; ++e)
                        Cvalue += As[row][e] * Bs[col][e];

                    // Synchronize to make sure that the preceding
                    // computation is done before loading two new
                    // sub-matrices of A and B in the next iteration
                    __syncthreads();
                }

                // Write Csub to device memory
                // Each thread writes one element
                C.set_element(blockRow * 8 + row, blockCol * 8 + col, Cvalue);
            }
        }
    }


/*******************************************************************************
 * \brief Atomic add for doubles that also works prior to compute capability 6.
 * \author Ole Schuett
 ******************************************************************************/
__device__ __inline__ void atomicAddDouble(double *address, double val) {
    if (val == 0.0)
        return;

#if __CUDA_ARCH__ >= 600
    atomicAdd(address, val); // part of cuda library
#else
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));

        // Uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
#endif
}

/*******************************************************************************
 * \brief Adds given value to matrix element cab[idx(b)][idx(a)].
 * \author Ole Schuett
 ******************************************************************************/
__device__ static inline void prep_term(const orbital a, const orbital b,
                                        const double value, const int n,
                                        double *cab) {
    atomicAddDouble(&cab[idx(b) * n + idx(a)], value);
}


/*******************************************************************************
 * \brief Sums first within a warp and then issues a single atomicAdd per warp.
 * \author Ole Schuett
 ******************************************************************************/
__device__ __inline__ void coalescedAtomicAdd(double *address, double val) {

    const cg::coalesced_group active = cg::coalesced_threads();

#if (CUDA_VERSION >= 11000)
    // Reduce from Cuda 11+ library is around 30% faster than the solution below.
    const double sum = cg::reduce(active, val, cg::plus<double>());

#else
    // Slow sequential reduction until group size is a power of two.
    double sum1 = 0.0;
    unsigned int group_size = active.size();
    while ((group_size & (group_size - 1)) != 0) {
        sum1 += active.shfl_down(val, group_size - 1);
        group_size--;
    }
    // Fast tree reduction halving group size in each iteration.
    double sum2 = val;
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        sum2 += active.shfl_down(sum2, offset);
    }
    const double sum = sum1 + sum2;
#endif

    // A single atomic add to avoid shared memory bank conflicts.
    if (active.thread_rank() == 0) {
        atomicAddDouble(address, sum);
    }
}

/*******************************************************************************
 * \brief Initializes the device's constant memory.
 * \author Ole Schuett
 ******************************************************************************/
inline static void init_constant_memory() {
    static bool initialized = false;
    if (initialized) {
        return; // constant memory has to be initialized only once
    }

    // Inverse coset mapping
    orbital coset_inv_host[1330];
    for (int lx = 0; lx <= 18; lx++) {
        for (int ly = 0; ly <= 18 - lx; ly++) {
            for (int lz = 0; lz <= 18 - lx - ly; lz++) {
                const int i = coset(lx, ly, lz);
                coset_inv_host[i] = {{lx, ly, lz}};
            }
        }
    }
    cudaError_t error =
        cudaMemcpyToSymbol(coset_inv, &coset_inv_host, sizeof(coset_inv_host), 0,
                           cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);

    // Binomial coefficient
    double binomial_coef_host[19][19] = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 6, 15, 20, 15, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 7, 21, 35, 35, 21, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 8, 28, 56, 70, 56, 28, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 9, 36, 84, 126, 126, 84, 36, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1, 0, 0, 0, 0, 0, 0, 0},
                                         {1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1, 0, 0, 0, 0, 0, 0},
                                         {1, 13, 78, 286, 715, 1287, 1716, 1716,
                                             1287, 715, 286, 78, 13, 1, 0, 0, 0, 0, 0}, {1, 14, 91, 364, 1001,
                                             2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1, 0, 0, 0, 0}, {1,
                                             15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455,
                                             105, 15, 1, 0, 0, 0}, {1, 16, 120, 560, 1820, 4368, 8008, 11440,
                                             12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1, 0, 0}, {1, 17, 136,
                                             680, 2380, 6188, 12376, 19448, 24310, 24310, 19448, 12376, 6188,
                                             2380, 680, 136, 17, 1, 0}, {1, 18, 153, 816, 3060, 8568, 18564,
                                             31824, 43758, 48620, 43758, 31824, 18564, 8568, 3060, 816, 153, 18,
                                             1}};
    error =
        cudaMemcpyToSymbol(binomial_coef, &binomial_coef_host,
                           sizeof(binomial_coef_host), 0, cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);

    initialized = true;
}


/*******************************************************************************
 * \brief Computes the polynomial expansion coefficients:
 *        (x-a)**lxa (x-b)**lxb -> sum_{ls} alpha(ls,lxa,lxb,1)*(x-p)**ls
 * \author Ole Schuett
 ******************************************************************************/
__device__ static void compute_alpha(const device &params,
                                     const smem_task &task,
                                     double *__restrict__ alpha) {
  // strides for accessing alpha
  const int s3 = (task.lp + 1);
  const int s2 = (task.la_max + 1) * s3;
  const int s1 = (task.lb_max + 1) * s2;

  for (int idir = threadIdx.z; idir < 3; idir += blockDim.z) {
    const double drpa = task.rp[idir] - task.ra[idir];
    const double drpb = task.rp[idir] - task.rb[idir];
    for (int la = threadIdx.y; la <= task.la_max; la += blockDim.y) {
      for (int lb = threadIdx.x; lb <= task.lb_max; lb += blockDim.x) {
         const int base = idir * s1 + lb * s2 + la * s3;
        for (int i = 0; i <= task.lp; i++) {
          alpha[base + i] = 0.0;
        }
        double a = 1.0;
        for (int k = 0; k <= la; k++) {
          double b = 1.0;
          const int base = idir * s1 + lb * s2 + la * s3;
          for (int l = 0; l <= lb; l++) {
            alpha[base + la - l + lb - k] +=
                a * b * binomial_coef[la][k] * binomial_coef[lb][l];
            b *= drpb;
          }
          a *= drpa;
        }
      }
    }
  }
  __syncthreads(); // because of concurrent writes to alpha
}

__inline__ __device__ void
return_cube_position(const int3 *__restrict__ const cube_center,
                     const int3 *__restrict__ const lower_boundaries_cube,
                     const int3 period, int3 *const position) {
  position->x = (cube_center->x + lower_boundaries_cube->x) % period.x;
  position->y = (cube_center->y + lower_boundaries_cube->y) % period.y;
  position->z = (cube_center->z + lower_boundaries_cube->z) % period.z;

  if (position->x < 0)
    position->x += period.x;
  if (position->y < 0)
    position->y += period.y;
  if (position->x < 0)
    position->y += period.z;
}

__inline__ __device__ void
convert_to_lattice_coordinates(const double *dh_inv_,
                               const double3 *__restrict__ const rp,
                               double3 *__restrict__ rp_c) {
  rp_c->x = dh_inv_[0] * rp->x + dh_inv_[3] * rp->y + dh_inv_[6] * rp->z;
  rp_c->y = dh_inv_[1] * rp->x + dh_inv_[4] * rp->y + dh_inv_[7] * rp->z;
  rp_c->z = dh_inv_[2] * rp->x + dh_inv_[5] * rp->y + dh_inv_[8] * rp->z;
}

__inline__ __device__ void convert_from_lattice_coordinates_to_cartesian(const double *__restrict__ dh_,
                                                                         const double3 *__restrict__ const rp,
                                                                         double3 *__restrict__ rp_c) {
    rp_c->x = dh_[0] * rp->x + dh_[3] * rp->y + dh_[6] * rp->z;
    rp_c->y = dh_[1] * rp->x + dh_[4] * rp->y + dh_[7] * rp->z;
    rp_c->z = dh_[2] * rp->x + dh_[5] * rp->y + dh_[8] * rp->z;
}


__device__ __inline__ void compute_cube_properties(const double radius,
                                                   const bool orthorhombic_,
                                                   const double *__restrict__ dh_,
                                                   const double *__restrict__ dh_inv_,
                                                   const double3 *__restrict__ rp,
                                                   double3 *__restrict__ roffset,
                                                   int3 *__restrict__ cubecenter,
                                                   int3 *__restrict__ lb_cube,
                                                   int3 *__restrict__ cube_size) {
    int3 ub_cube;

  /* center of the gaussian in the lattice coordinates */
  double3 rp1;

  /* it is in the lattice vector frame */
  convert_to_lattice_coordinates(dh_inv_, rp, &rp1);

  cubecenter->x = floor(rp1.x);
  cubecenter->y = floor(rp1.y);
  cubecenter->z = floor(rp1.z);

  if (orthorhombic_) {
    /* seting up the cube parameters */
    const double3 dr = {.x = dh_[0], .y = dh_[4], .z = dh_[8]};
    const double3 dr_inv = {.x = dh_inv_[0], .y = dh_inv_[4], .z = dh_inv_[8]};
    /* cube center */

    /* lower and upper bounds */

    // Historically, the radius gets discretized.
    const double drmin = min(dr.x, min(dr.y, dr.z));
    const double disr_radius = drmin * max(1.0, ceil(radius / drmin));

    roffset->x = rp->x - cubecenter->x * dr.x;
    roffset->y = rp->y - cubecenter->y * dr.y;
    roffset->z = rp->z - cubecenter->z * dr.z;

    lb_cube->x = ceil(-1e-8 - disr_radius * dr_inv.x);
    lb_cube->y = ceil(-1e-8 - disr_radius * dr_inv.y);
    lb_cube->z = ceil(-1e-8 - disr_radius * dr_inv.z);

    roffset->x *= dr_inv.x;
    roffset->y *= dr_inv.y;
    roffset->z *= dr_inv.z;

    // Symetric interval
    ub_cube.x = 1 - lb_cube->x;
    ub_cube.y = 1 - lb_cube->y;
    ub_cube.z = 1 - lb_cube->z;

  } else {

    lb_cube->x = INT_MAX;
    ub_cube.x = INT_MIN;
    lb_cube->y = INT_MAX;
    ub_cube.y = INT_MIN;
    lb_cube->z = INT_MAX;
    ub_cube.z = INT_MIN;

    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        for (int k = -1; k <= 1; k++) {
          double3 r = make_double3(((double)i) * radius, ((double)j) * radius,
                                   ((double)k) * radius);
          convert_to_lattice_coordinates(dh_inv_, &r, roffset);

          lb_cube->x = min(lb_cube->x, (int)floor(roffset->x));
          ub_cube.x = max(ub_cube.x, (int)ceil(roffset->x));

          lb_cube->y = min(lb_cube->y, (int)floor(roffset->y));
          ub_cube.y = max(ub_cube.y, (int)ceil(roffset->y));

          lb_cube->z = min(lb_cube->z, (int)floor(roffset->z));
          ub_cube.z = max(ub_cube.z, (int)ceil(roffset->z));
        }
      }
    }

    /* compute the offset in lattice coordinates */

    roffset->x = rp1.x - cubecenter->x;
    roffset->y = rp1.y - cubecenter->y;
    roffset->z = rp1.z - cubecenter->z;
  }

  /* compute the cube size ignoring periodicity */
  cube_size->x = ub_cube.x - lb_cube->x + 1;
  cube_size->y = ub_cube.y - lb_cube->y + 1;
  cube_size->z = ub_cube.z - lb_cube->z + 1;
}

__inline__ __device__ void
compute_window_size(const int3 *const grid_size,
                    const int3 *const lower_corner_,
                    const int3 *const period_, /* also full size of the grid */
                    const int border_mask,
                    const int3 *border_width,
                    int3 *const window_size,
                    int3 *const window_shift) {
  window_shift->x = 0;
  window_shift->y = 0;
  window_shift->z = 0;

  window_size->x = grid_size->x;
  window_size->y = grid_size->y;
  window_size->z = grid_size->z;

  if (grid_size->x != period_->x)
    window_size->x--;

  if (grid_size->y != period_->y)
    window_size->y--;

  if (grid_size->z != period_->z)
    window_size->z--;

  if ((grid_size->x != period_->x) || (grid_size->y != period_->y) ||
      (grid_size->z != period_->z)) {
    if (border_mask & (1 << 0))
      window_shift->x += border_width->x;
    if (border_mask & (1 << 1))
      window_size->x -= border_width->x;
    if (border_mask & (1 << 2))
      window_shift->y += border_width->y;
    if (border_mask & (1 << 3))
      window_size->y -= border_width->y;
    if (border_mask & (1 << 4))
      window_shift->z += border_width->z;
    if (border_mask & (1 << 5))
      window_size->z -= border_width->z;
  }
}


/*******************************************************************************
 * \brief Transforms coefficients C_ab into C_xyz.
 * \author Ole Schuett
 ******************************************************************************/
__device__ __inline__ static void cab_to_cxyz(const device &params,
                                              const smem_task &task,
                                              const double *__restrict__ alpha,
                                              const double *__restrict__ cab,
                                              double *__restrict__ cxyz) {

    //   *** initialise the coefficient matrix, we transform the sum
    //
    // sum_{lxa,lya,lza,lxb,lyb,lzb} P_{lxa,lya,lza,lxb,lyb,lzb} *
    //         (x-a_x)**lxa (y-a_y)**lya (z-a_z)**lza (x-b_x)**lxb (y-a_y)**lya
    //         (z-a_z)**lza
    //
    // into
    //
    // sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-p_x)**lxp (y-p_y)**lyp (z-p_z)**lzp
    //
    // where p is center of the product gaussian, and lp = la_max + lb_max
    // (current implementation is l**7)

    // strides for accessing alpha
    const int s3 = (task.lp + 1);
    const int s2 = (task.la_max + 1) * s3;
    const int s1 = (task.lb_max + 1) * s2;

    // TODO: Maybe we can transpose alpha to index it directly with ico and jco.
    for (int lzp = threadIdx.z; lzp <= task.lp; lzp += blockDim.z) {
        for (int lyp = threadIdx.y; lyp <= task.lp - lzp; lyp += blockDim.y) {
            for (int lxp = threadIdx.x; lxp <= task.lp - lzp - lyp;
                 lxp += blockDim.x) {
                double reg = 0.0; // accumulate into a register
                for (int jco = 0; jco < ncoset(task.lb_max); jco++) {
                    const orbital b = coset_inv[jco];
                    for (int ico = 0; ico < ncoset(task.la_max); ico++) {
                        const orbital a = coset_inv[ico];
                        const double p = task.prefactor *
                            alpha[0 * s1 + b.l[0] * s2 + a.l[0] * s3 + lxp] *
                            alpha[1 * s1 + b.l[1] * s2 + a.l[1] * s3 + lyp] *
                            alpha[2 * s1 + b.l[2] * s2 + a.l[2] * s3 + lzp];
                        reg += p * cab[jco * task.n1 + ico]; // collocate
                    }
                }

                cxyz[coset(lxp, lyp, lzp)] = reg; // overwrite - no zeroing needed.
            }
        }
    }
    __syncthreads(); // because of concurrent writes to cxyz / cab
}

/*******************************************************************************
 * \brief Transforms coefficients C_xyz into C_ab.
 * \author Ole Schuett
 ******************************************************************************/
__device__ __inline__ static void cxyz_to_cab(const device &params,
                                              const smem_task &task,
                                              const double *__restrict__ alpha,
                                              const double *__restrict__ cxyz,
                                              double *__restrict__ cab) {

    //   *** initialise the coefficient matrix, we transform the sum
    //
    // sum_{lxa,lya,lza,lxb,lyb,lzb} P_{lxa,lya,lza,lxb,lyb,lzb} *
    //         (x-a_x)**lxa (y-a_y)**lya (z-a_z)**lza (x-b_x)**lxb (y-a_y)**lya
    //         (z-a_z)**lza
    //
    // into
    //
    // sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-p_x)**lxp (y-p_y)**lyp (z-p_z)**lzp
    //
    // where p is center of the product gaussian, and lp = la_max + lb_max
    // (current implementation is l**7)

    // strides for accessing alpha
    const int s3 = (task.lp + 1);
    const int s2 = (task.la_max + 1) * s3;
    const int s1 = (task.lb_max + 1) * s2;

    // TODO: Maybe we can transpose alpha to index it directly with ico and jco.

    // integrate

//    cg::thread_block block = cg::this_thread_block();
    if (threadIdx.z == 0) { // TODO: How bad is this?
        for (int jco = threadIdx.y; jco < ncoset(task.lb_max); jco += blockDim.y) {
            const orbital b = coset_inv[jco];
            for (int ico = threadIdx.x; ico < ncoset(task.la_max);
                 ico += blockDim.x) {
                const orbital a = coset_inv[ico];
                double reg = 0.0; // accumulate into a register
                for (int lzp = 0; lzp <= task.lp; lzp++) {
                    for (int lyp = 0; lyp <= task.lp - lzp; lyp++) {
                        for (int lxp = 0; lxp <= task.lp - lzp - lyp; lxp++) {
                            const double p = task.prefactor *
                                alpha[0 * s1 + b.l[0] * s2 + a.l[0] * s3 + lxp] *
                                alpha[1 * s1 + b.l[1] * s2 + a.l[1] * s3 + lyp] *
                                alpha[2 * s1 + b.l[2] * s2 + a.l[2] * s3 + lzp];

                            reg += p * cxyz[coset(lxp, lyp, lzp)]; // integrate
                        }
                    }
                }
                cab[jco * task.n1 + ico] = reg; // partial loop coverage -> zero it
            }
        }
    }
    __syncthreads(); // because of concurrent writes to cxyz / cab
}

/*******************************************************************************
 * \brief Initializes the cab matrix with zeros.
 * \author Ole Schuett
 ******************************************************************************/
template <typename T> __device__ void zero(T *cab, const int size) {
    cg::thread_block block = cg::this_thread_block();
    for (int i = block.thread_rank(); i < size; i += block.size())
        cab[i] = 0;
}

/*******************************************************************************
 * \brief Copies a task from global to shared memory and does precomputations.
 * \author Ole Schuett
 ******************************************************************************/
__device__ __inline__ void fill_smem_task(const device &dev, const int starting_task_, smem_task &task) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        const auto &glb_task = dev.task_dev_[starting_task_ + blockIdx.x];
        const int iatom = glb_task.iatom;
        const int jatom = glb_task.jatom;
        const int iset = glb_task.iset;
        const int jset = glb_task.jset;
        const int ipgf = glb_task.ipgf;
        const int jpgf = glb_task.jpgf;
        const int ikind = glb_task.ikind;
        const int jkind = glb_task.jkind;
        const auto &ibasis = dev.basis_sets_dev_[ikind];
        const auto &jbasis = dev.basis_sets_dev_[jkind];

        task.zeta = ibasis.zet[iset * ibasis.maxpgf + ipgf];
        task.zetb = jbasis.zet[jset * jbasis.maxpgf + jpgf];
        task.zetp = task.zeta + task.zetb;
        const double f = task.zetb / task.zetp;
        task.rab2 = 0.0;
        for (int i = 0; i < 3; i++) {
            task.rab[i] = glb_task.rab[i];
            task.rab2 += task.rab[i] * task.rab[i];
            task.ra[i] = glb_task.ra[i];
            task.rb[i] = glb_task.rb[i];
            task.rp[i] = glb_task.rp[i];
        }

        // task.border_mask = glb_task.border_mask;
        task.radius = glb_task.radius;
        double radius2 = task.radius * task.radius;
        task.prefactor = exp(-task.zeta * f * task.rab2);
        task.off_diag_twice = (iatom == jatom) ? 1.0 : 2.0;

        // angular momentum range of basis set
        const int la_max_basis = glb_task.lmax[0];
        const int lb_max_basis = glb_task.lmax[1];
        const int la_min_basis = glb_task.lmin[0];
        const int lb_min_basis = glb_task.lmin[1];

        // start of decontracted set, ie. pab and hab
        task.first_coseta = (la_min_basis > 0) ? ncoset(la_min_basis - 1) : 0;
        task.first_cosetb = (lb_min_basis > 0) ? ncoset(lb_min_basis - 1) : 0;

        // size of decontracted set, ie. pab and hab
        task.ncoseta = ncoset(la_max_basis);
        task.ncosetb = ncoset(lb_max_basis);

        // angular momentum range for the actual collocate/integrate opteration.
        task.la_max = la_max_basis + dev.lmax_diff[0];
        task.lb_max = lb_max_basis + dev.lmax_diff[1];
        task.la_min = max(la_min_basis + dev.lmin_diff[0], 0);
        task.lb_min = max(lb_min_basis + dev.lmin_diff[1], 0);
        task.lp = task.la_max + task.lb_max;

        // size of the cab matrix
        task.n1 = ncoset(task.la_max);
        task.n2 = ncoset(task.lb_max);

        // size of entire spherical basis
        task.nsgfa = ibasis.nsgf;
        task.nsgfb = jbasis.nsgf;

        // size of spherical set
        task.nsgf_seta = ibasis.nsgf_set[iset];
        task.nsgf_setb = jbasis.nsgf_set[jset];

        // strides of the sphi transformation matrices
        task.maxcoa = ibasis.maxco;
        task.maxcob = jbasis.maxco;

        // start of spherical set within the basis
        const int sgfa = ibasis.first_sgf[iset] - 1;
        const int sgfb = jbasis.first_sgf[jset] - 1;

    // start of exponent within the cartesian set
        const int o1 = ipgf * task.ncoseta;
        const int o2 = jpgf * task.ncosetb;

        // transformations from contracted spherical to primitiv carthesian basis
        task.sphia = ibasis.sphi + sgfa * glb_task.maxcoa + o1;
        task.sphib = jbasis.sphi + sgfb * glb_task.maxcob + o2;

        // Locate current matrix block within the buffer.
        const int block_num = glb_task.block_num;
        const int block_offset = dev.block_offsets_dev_[block_num];
        task.block_transposed = (iatom > jatom);
        const int subblock_offset = (task.block_transposed)
            ? sgfa * task.nsgfb + sgfb
            : sgfb * task.nsgfa + sgfa;
        task.pab_block = dev.pab_blocks_dev_ + block_offset + subblock_offset;

        if (dev.hab_blocks_dev_ != nullptr) {
            task.hab_block = &dev.hab_blocks_dev_[block_offset + subblock_offset];
            if (dev.forces_dev_ != nullptr) {
                task.forces_a = &dev.forces_dev_[3 * iatom];
                task.forces_b = &dev.forces_dev_[3 * jatom];
            }
        }
    }
    __syncthreads();
}

#endif
