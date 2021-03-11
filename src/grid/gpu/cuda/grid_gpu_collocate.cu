/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#ifdef __GRID_CUDA

#include <algorithm>
#include <assert.h>
#include <cuda.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "grid_gpu_internal_header.h"

#include "../../common/grid_common.h"
#include "../../common/grid_prepare_pab.h"
// #include "grid_gpu_collint.h"


/*******************************************************************************
 * \brief Decontracts the subblock, going from spherical to cartesian harmonics.
 * \author Ole Schuett
 ******************************************************************************/
template <bool IS_FUNC_AB>
__device__ static void block_to_cab(const device &params,
                                    const smem_task &task,
                                    double *__restrict__ cab) {

  // The spherical index runs over angular momentum and then over contractions.
  // The carthesian index runs over exponents and then over angular momentum.

  // Decontract block, apply prepare_pab, and store in cab.
  // This is a double matrix product. Since the pab block can be quite large the
  // two products are fused to conserve shared memory.
  for (int i = threadIdx.x; i < task.nsgf_setb; i += blockDim.x) {
    for (int j = threadIdx.y; j < task.nsgf_seta; j += blockDim.y) {
      double block_val;
      if (task.block_transposed) {
        block_val = task.pab_block[j * task.nsgfb + i] * task.off_diag_twice;
      } else {
        block_val = task.pab_block[i * task.nsgfa + j] * task.off_diag_twice;
      }

      if (IS_FUNC_AB) {
        // fast path for common case
        const int jco_start = task.first_cosetb + threadIdx.z;
        for (int jco = jco_start; jco < task.ncosetb; jco += blockDim.z) {
          const double sphib = task.sphib[i * task.maxcob + jco];
          for (int ico = task.first_coseta; ico < task.ncoseta; ico++) {
            const double sphia = task.sphia[j * task.maxcoa + ico];
            const double pab_val = block_val * sphia * sphib;
            atomicAddDouble(&cab[jco * task.ncoseta + ico], pab_val);
          }
        }
      } else {
        // Since prepare_pab is a register hog we use it only when really needed
        const int jco_start = task.first_cosetb + threadIdx.z;
        for (int jco = jco_start; jco < task.ncosetb; jco += blockDim.z) {
          const orbital b = coset_inv[jco];
          for (int ico = task.first_coseta; ico < task.ncoseta; ico++) {
            const orbital a = coset_inv[ico];
            const double sphia = task.sphia[j * task.maxcoa + idx(a)];
            const double sphib = task.sphib[i * task.maxcob + idx(b)];
            const double pab_val = block_val * sphia * sphib;
            prepare_pab(params.func_, a, b, task.zeta, task.zetb, pab_val,
                        task.n1, cab);
          }
        }
      }
    }
  }
  __syncthreads(); // because of concurrent writes to cab
}

/*******************************************************************************
 * \brief Cuda kernel for collocating all tasks of one grid level.
 * \author Ole Schuett
 ******************************************************************************/
template <bool IS_FUNC_AB>
__global__ static void collocate_kernel(const device dev_, const int first_task) {
    cg::thread_block block = cg::this_thread_block();

// Copy task from global to shared memory and precompute some stuff.
     __shared__ smem_task task;
     fill_smem_task(dev_, first_task, task);

     //  Alloc shared memory.
     extern __shared__ double shared_memory[];
     double *smem_cab = &shared_memory[dev_.smem_cab_offset_];
     double *smem_alpha = &shared_memory[dev_.smem_alpha_offset_];
     double *coefs_ = &shared_memory[dev_.smem_cxyz_offset_];

     zero(smem_cab, task.n1 * task.n2);
     block_to_cab<IS_FUNC_AB>(dev_, task, smem_cab);

     compute_alpha(dev_, task, smem_alpha);
     cab_to_cxyz(dev_, task, smem_alpha, smem_cab, coefs_);

     const bool apply_cutoff = true;
     int3 cube_size, cube_center, lb_cube, window_size, window_shift;
     const bool orthorhombic_ = dev_.orthorhombic_;
     double3 roffset;
     compute_cube_properties(task.radius,
                             orthorhombic_,
                             dev_.dh_,
                             dev_.dh_inv_,
                             (double3 *)dev_.task_dev_[blockIdx.x + first_task].rp,
                             &roffset,
                             &cube_center,
                             &lb_cube,
                             &cube_size);

     compute_window_size((int3*)dev_.grid_local_size_,
                         (int3*)dev_.grid_lower_corner_,
                         (int3*)dev_.grid_full_size_, //also full size of the grid
                         dev_.task_dev_[first_task + blockIdx.x].border_mask,
                         (int3*)dev_.grid_border_width_,
                         &window_size, &window_shift);
     cube_center.z += lb_cube.z;
     cube_center.y += lb_cube.y;
     cube_center.x += lb_cube.x;

     for (int z = threadIdx.z; z < cube_size.z; z += blockDim.z) {
         const double z1 = z + lb_cube.z - roffset.z;
         const int z2 = (z + cube_center.z - dev_.grid_lower_corner_[0] + 32 * dev_.grid_full_size_[0]) %
             dev_.grid_full_size_[0];

         //check if the point is within the window
         if ((z2 < window_shift.z) || (z2 > window_size.z)) {
             continue;
         }

         for (int y = threadIdx.y; y < cube_size.y; y += blockDim.y) {
             double y1 = y + lb_cube.y - roffset.y;
             const int y2 = (y + cube_center.y - dev_.grid_lower_corner_[1] + 32 * dev_.grid_full_size_[1]) %
                 dev_.grid_full_size_[1];

             //check if the point is within the window
             if ((y2 < window_shift.y) || (y2 > window_size.y)) {
                 continue;
             }

             for (int x = threadIdx.x; x < cube_size.x; x += blockDim.x) {
                 const double x1 = (x + lb_cube.x - roffset.x);
                 const int x2 = (x + cube_center.x - dev_.grid_lower_corner_[2] + 32 * dev_.grid_full_size_[2]) %
                     dev_.grid_full_size_[2];

                 //check if the point is within the window
                 if ((x2 < window_shift.x) || (x2 > window_size.x)) {
                     continue;
                 }

                 // compute the coordinates of the point in atomic coordinates
                 double3 r3;
                 if (!orthorhombic_) {
                     r3.x = z1 * dev_.dh_[6] + y1 * dev_.dh_[3] + x1 * dev_.dh_[0];
                     r3.y = z1 * dev_.dh_[7] + y1 * dev_.dh_[4] + x1 * dev_.dh_[1];
                     r3.z = z1 * dev_.dh_[8] + y1 * dev_.dh_[5] + x1 * dev_.dh_[2];
                 } else {
                     r3.x = x1 * dev_.dh_[8];
                     r3.y = y1 * dev_.dh_[4];
                     r3.z = z1 * dev_.dh_[0];
                 }

                 if (apply_cutoff &&
                     ((task.radius * task.radius) < (r3.x * r3.x + r3.y * r3.y + r3.z * r3.z)))
                     continue;

                 double res = 0.0;
                 double dz = 1;


                 switch (task.lp) {
                 case 2:
                     res = coefs_[coset(0, 1, 1)] * r3.y * r3.z +
                         coefs_[coset(1, 0, 1)] * r3.z * r3.x +
                         coefs_[coset(1, 1, 0)] * r3.x * r3.y +
                         coefs_[coset(2, 0, 0)] * r3.x * r3.x +
                         coefs_[coset(0, 2, 0)] * r3.y * r3.y +
                         coefs_[coset(0, 0, 2)] * r3.z * r3.z;
                 case 1:
                     res += coefs_[coset(0, 0, 1)] * r3.z +
                         coefs_[coset(0, 1, 0)] * r3.y +
                         coefs_[coset(1, 0, 0)] * r3.x;
                 case 0:
                     res += coefs_[0];
                     break;
                 default:
                     for (int gamma = 0; gamma <= task.lp; gamma++) {
                         double dy = dz;
                         for (int beta = 0; beta <= (task.lp - gamma); beta++) {
                           double dx = dy;
                           for (int alpha = 0; alpha <= (task.lp - gamma - beta); alpha++) {
                               res += coefs_[coset(alpha, beta, gamma)] * dx;
                               dx *= r3.x;
                           }
                           dy *= r3.y;
                         }
                         dz *= r3.z;
                     }
                     break;
                 }

                 res *= exp(-(r3.x * r3.x + r3.y * r3.y + r3.z * r3.z) * task.zetp);

                 atomicAddDouble(&dev_.grid_dev_[(z2 * dev_.grid_local_size_[1] + y2) * dev_.grid_local_size_[2] + x2],
                                 res);
             }
         }
     }
}

/*******************************************************************************
 * \brief Launches the Cuda kernel that collocates all tasks of one grid level.
 * \author Ole Schuett
 ******************************************************************************/
void device::collocate_one_grid_level(gpu_context &ctx_, const int level) {
    const int ntasks = ctx_.ctx().tasks_per_level(level);
    if (ntasks == 0) {
        return; // Nothing to do.
    }
    this->func_ = ctx_.ctx().func();
    init_constant_memory();
    cudaMemsetAsync(grid_dev_, 0, sizeof(double) * ctx_.ctx().grid(level).size(), stream_);
    // Compute max angular momentum.
    const prepare_ldiffs ldiffs = prepare_get_ldiffs(ctx_.ctx().func());
    const int la_max = ctx_.ctx().lmax_ + ldiffs.la_max_diff;
    const int lb_max = ctx_.ctx().lmax_ + ldiffs.lb_max_diff;
    const int lp_max = la_max + lb_max;
    lmax_diff[0] = ldiffs.la_max_diff;
    lmax_diff[1] = ldiffs.lb_max_diff;
    lmin_diff[0] = ldiffs.la_min_diff;
    lmin_diff[1] = ldiffs.lb_min_diff;

    // Compute required shared memory.
    // TODO: Currently, cab's indicies run over 0...ncoset[lmax],
    //       however only ncoset(lmin)...ncoset(lmax) are actually needed.
    const int cab_len = ncoset(lb_max) * ncoset(la_max);
    const int alpha_len = 3 * (lb_max + 1) * (la_max + 1) * (lp_max + 1);
    const int cxyz_len = ncoset(lp_max);
    const size_t smem_per_block =
        (cab_len + alpha_len + cxyz_len) * sizeof(double);

    if (smem_per_block > 48 * 1024) {
        fprintf(stderr, "ERROR: Not enough shared memory in grid_gpu_collocate.\n");
        fprintf(stderr, "cab_len: %i, ", cab_len);
        fprintf(stderr, "alpha_len: %i, ", alpha_len);
        fprintf(stderr, "cxyz_len: %i, ", cxyz_len);
        fprintf(stderr, "total smem_per_block: %f kb\n\n", smem_per_block / 1024.0);
        abort();
    }

    // kernel parameters
    this->smem_cxyz_offset_ = 0;
    this->smem_alpha_offset_ = cxyz_len;
    this->smem_cab_offset_ = this->smem_alpha_offset_ + alpha_len;

    // Launch !
    const int nblocks = ntasks;
    const dim3 threads_per_block(4, 8, 8);

    if (func_ == GRID_FUNC_AB) {
        collocate_kernel<false><<<nblocks, threads_per_block, smem_per_block,
            stream_>>>(*this, ctx_.ctx().first_task(level));
    } else {
        collocate_kernel<true><<<nblocks, threads_per_block, smem_per_block,
            stream_>>>(*this, ctx_.ctx().first_task(level));
    }

    CHECK(cudaMemcpyAsync(ctx_.ctx().grid(level).at(), grid_dev_, sizeof(double) * ctx_.ctx().grid(level).size(), cudaMemcpyDeviceToHost, stream_));
}
#endif
