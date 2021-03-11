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

extern "C" {
    #include "../../common/grid_common.h"
    #include "../../common/grid_process_vab.h"
}

/*******************************************************************************
 * \brief Contracts the subblock, going from cartesian harmonics to spherical.
 * \author Ole Schuett
 ******************************************************************************/
template <bool COMPUTE_TAU>
__device__ static void store_hab(const device &params,
                                 const smem_task &task,
                                 const double *__restrict__ cab) {
  // The spherical index runs over angular momentum and then over contractions.
  // The carthesian index runs over exponents and then over angular momentum.

  // This is a double matrix product. Since the block can be quite large the
  // two products are fused to conserve shared memory.
  for (int i = threadIdx.z; i < task.nsgf_setb; i += blockDim.z) {
    for (int j = threadIdx.y; j < task.nsgf_seta; j += blockDim.y) {
      const int jco_start = task.first_cosetb + threadIdx.x;
      double res = 0.0;

      for (int jco = jco_start; jco < task.ncosetb; jco += blockDim.x) {
        const orbital b = coset_inv[jco];
        double block_val = 0.0;
        const double sphib = task.sphib[i * task.maxcob + jco];
        for (int ico = task.first_coseta; ico < task.ncoseta; ico++) {
          const orbital a = coset_inv[ico];
          const double hab =
              get_hab(a, b, task.zeta, task.zetb, task.n1, cab, COMPUTE_TAU);
          const double sphia = task.sphia[j * task.maxcoa + ico];
          block_val += hab * sphia * sphib;
        }

        res += block_val;
      }

      if (task.block_transposed) {
          atomicAddDouble(&task.hab_block[j * task.nsgfb + i], res);
      } else {
          atomicAddDouble(&task.hab_block[i * task.nsgfa + j], res);
      }

    }
  }
}
/*******************************************************************************
 * \brief Adds contributions from cab to forces and virial.
 * \author Ole Schuett
 ******************************************************************************/
template <bool COMPUTE_TAU>
__device__ static void store_forces_and_virial(const device &params,
                                               const smem_task &task,
                                               const double *__restrict__ cab) {
    double fa[3] = {0.0, 0.0, 0.0};
    double fb[3] = {0.0, 0.0, 0.0};
    for (int i = threadIdx.z; i < task.nsgf_setb; i += blockDim.z) {
        for (int j = threadIdx.y; j < task.nsgf_seta; j += blockDim.y) {
            double block_val;
            if (task.block_transposed) {
                block_val = task.pab_block[j * task.nsgfb + i] * task.off_diag_twice;
            } else {
                block_val = task.pab_block[i * task.nsgfa + j] * task.off_diag_twice;
            }
            const int jco_start = task.first_cosetb + threadIdx.x;
            for (int jco = jco_start; jco < task.ncosetb; jco += blockDim.x) {
                const double sphib = task.sphib[i * task.maxcob + jco];
                for (int ico = task.first_coseta; ico < task.ncoseta; ico++) {
                    const double sphia = task.sphia[j * task.maxcoa + ico];
                    const double pabval = block_val * sphia * sphib;
                    const orbital b = coset_inv[jco];
                    const orbital a = coset_inv[ico];
                    for (int k = 0; k < 3; k++) {
                        const double force_a = get_force_a(a, b, k, task.zeta, task.zetb,
                                                           task.n1, cab, COMPUTE_TAU);
                        fa[k] += force_a * pabval;
                        const double force_b =
                            get_force_b(a, b, k, task.zeta, task.zetb, task.rab,
                                        task.n1, cab, COMPUTE_TAU);
                        fb[k] += force_b * pabval;
                    }
                    if (params.virial_dev_ != NULL) {
                        for (int k = 0; k < 3; k++) {
                            for (int l = 0; l < 3; l++) {
                                const double virial_a =
                                    get_virial_a(a, b, k, l, task.zeta, task.zetb, task.n1,
                                                 cab, COMPUTE_TAU);
                                const double virial_b =
                                    get_virial_b(a, b, k, l, task.zeta, task.zetb, task.rab,
                                                 task.n1, cab, COMPUTE_TAU);
                                const double virial = pabval * (virial_a + virial_b);
                                coalescedAtomicAdd(&params.virial_dev_[k * 3 + l], virial);
                            }
                        }
                    }
                }
            }
        }
    }
    coalescedAtomicAdd(&task.forces_b[0], fb[0]);
    coalescedAtomicAdd(&task.forces_b[1], fb[1]);
    coalescedAtomicAdd(&task.forces_b[2], fb[2]);
    coalescedAtomicAdd(&task.forces_a[0], fa[0]);
    coalescedAtomicAdd(&task.forces_a[1], fa[1]);
    coalescedAtomicAdd(&task.forces_a[2], fa[2]);
}

/*******************************************************************************
 * \brief Cuda kernel for integrating all tasks of one grid level.
 * \author Ole Schuett
 ******************************************************************************/
template <bool COMPUTE_TAU, bool CALCULATE_FORCES>
__global__ static void integrate_kernel(const device dev_, const int first_task) {

  // Copy task from global to shared memory and precompute some stuff.
  __shared__ smem_task task;
  fill_smem_task(dev_, first_task, task);

  extern __shared__ double shared_memory[];
  double *coef_ = &shared_memory[dev_.smem_cxyz_offset_];

  memset(coef_, 0, ncoset(task.lp) * sizeof(double));

  const bool apply_cutoff = true;
  int3 cube_size, cube_center, lb_cube, window_size, window_shift;
  const int3 grid_lower_corner_ = make_int3(dev_.grid_lower_corner_[2], dev_.grid_lower_corner_[1], dev_.grid_lower_corner_[0]);
  double3 roffset;
  // double disr_radius = 0;
  //const double *__restrict__ coef = coef_gpu_ + coef_offset_gpu_[blockIdx.x];
  //const double zeta = task.zetp;

  const int lmax = task.lp;
  const bool orthorhombic_ = dev_.orthorhombic_;
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
                      (int3*)dev_.grid_full_size_, /* also full size of the grid */
                      dev_.task_dev_[first_task + blockIdx.x].border_mask,
                      (int3*)dev_.grid_border_width_,
                      &window_size, &window_shift);

  cube_center.z += lb_cube.z;
  cube_center.y += lb_cube.y;
  cube_center.x += lb_cube.x;

  double4 coefs__ = make_double4(0.0, 0.0, 0.0, 0.0);

  __syncthreads();
  for (int z = threadIdx.z; z < cube_size.z; z += blockDim.z) {
      const double z1 = z + lb_cube.z - roffset.z;
      const int z2 = (z + cube_center.z - grid_lower_corner_.z + 32 * dev_.grid_full_size_[0]) %
          dev_.grid_full_size_[0];

      /* check if the point is within the window */
      if ((z2 < window_shift.z) || (z2 > window_size.z)) {
          continue;
      }

      for (int y = threadIdx.y; y < cube_size.y; y += blockDim.y) {
          double y1 = y + lb_cube.y - roffset.y;
          const int y2 = (y + cube_center.y - grid_lower_corner_.y + 32 * dev_.grid_full_size_[1]) %
              dev_.grid_full_size_[1];

          /* check if the point is within the window */
          if ((y2 < window_shift.y) || (y2 > window_size.y)) {
              continue;
          }

          for (int x = threadIdx.x; x < cube_size.x; x += blockDim.x) {
              const double x1 = (x + lb_cube.x - roffset.x);
              const int x2 = (x + cube_center.x - grid_lower_corner_.x + 32 * dev_.grid_full_size_[2]) %
                  dev_.grid_full_size_[2];

              /* check if the point is within the window */
              if ((x2 < window_shift.x) || (x2 > window_size.x)) {
                  continue;
              }

              /* compute the coordinates of the point in atomic coordinates */
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

              const double grid_value = dev_.grid_dev_[(z2 * dev_.grid_local_size_[1] + y2) * dev_.grid_local_size_[2] + x2] *
                  exp(-(r3.x * r3.x + r3.y * r3.y + r3.z * r3.z) * task.zetp);
              double dz = 1;

              /* NOTE: the coefficients are stored as lx,lz,ly */
              /* It is suboptimal right now because i do more operations than needed
               * (a lot of coefs_ are zero). Moreover, it is a dgemm underneath and
               * could be treated with tensor cores */
              switch (task.lp) {
              case 1:
                  coefs__.x += grid_value * r3.x;
                  coefs__.y += grid_value * r3.y;
                  coefs__.z += grid_value * r3.z;
              case 0:
                  coefs__.w += grid_value;
                  break;
              default: {

                  // for (int j = 0; j < active.size(); j++) {
                  //     for (int i = (active.thread_rank() + j) % active.size(); i < ncoset(task.lp); i += active.size()) {
                  //         const orbital b = coset_inv[i];
                  //         coef_[i] += grid_value * pow(r3.x, b.l[0]) * pow(r3.y, b.l[1]) * pow(r3.z, b.l[2]);
                  //     }
                  // }
                  for (int gamma = 0; gamma <= lmax; gamma++) {
                      double dy = 1;
                      for (int beta = 0; beta <= (lmax - gamma); beta++) {
                          double dx = dz * dy;
                          for (int alpha = 0; alpha <= (lmax - gamma - beta); alpha++) {
                              coalescedAtomicAdd(&coef_[coset(alpha, beta, gamma)], grid_value * dx);
                              dx *= r3.x;
                          }
                          dy *= r3.y;
                      }
                      dz *= r3.z;
                  }
              }
                  break;
              }
          }
      }
  }

  if (task.lp <= 1)
      coalescedAtomicAdd(&coef_[0], coefs__.w);

  if (task.lp == 1) {
      coalescedAtomicAdd(&coef_[coset(1,0,0)], coefs__.x);
      coalescedAtomicAdd(&coef_[coset(0,1,0)], coefs__.y);
      coalescedAtomicAdd(&coef_[coset(0,0,1)], coefs__.z);
  }

  double *smem_cab = &shared_memory[dev_.smem_cab_offset_];
  double *smem_alpha = &shared_memory[dev_.smem_alpha_offset_];

  zero<double>(smem_cab, task.n1 * task.n2);

  compute_alpha(dev_, task, smem_alpha);

  cxyz_to_cab(dev_, task, smem_alpha, coef_, smem_cab);

  store_hab<COMPUTE_TAU>(dev_, task, smem_cab);

  if (CALCULATE_FORCES) {
      store_forces_and_virial<COMPUTE_TAU>(dev_, task, smem_cab);
  }
}

/*******************************************************************************
 * \brief Launches the Cuda kernel that integrates all tasks of one grid level.
 * \author Ole Schuett
 ******************************************************************************/
void device::integrate_one_grid_level(gpu_context &ctx_, const int level) {
    if (ctx_.ctx().tasks_per_level(level) == 0)
        return;

    cudaMemcpyAsync(grid_dev_,
                    ctx_.grid(level).at(),
                    sizeof(double) * ctx_.ctx().grid(level).size(),
                    cudaMemcpyHostToDevice,
                    stream_);

    init_constant_memory();

  // Compute max angular momentum.
  const bool calculate_forces = (forces_dev_ != NULL);
  const bool calculate_virial = (virial_dev_ != NULL);
  assert(!calculate_virial || calculate_forces);
  const process_ldiffs ldiffs =
      process_get_ldiffs(calculate_forces, calculate_virial, ctx_.ctx().calculate_tau());
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
    fprintf(stderr, "ERROR: Not enough shared memory in grid_gpu_integrate.\n");
    fprintf(stderr, "cab_len: %i, ", cab_len);
    fprintf(stderr, "alpha_len: %i, ", alpha_len);
    fprintf(stderr, "cxyz_len: %i, ", cxyz_len);
    fprintf(stderr, "total smem_per_block: %f kb\n\n", smem_per_block / 1024.0);
    abort();
  }

  // size of the shared memory pull

  this->smem_cxyz_offset_ = 0;
  this->smem_alpha_offset_ = cxyz_len;
  this->smem_cab_offset_ = this->smem_alpha_offset_ + alpha_len;

  // Launch !
  const int nblocks = ctx_.ctx().tasks_per_level(level);
  const dim3 threads_per_block(2, 4, 16);

  if (!ctx_.ctx().calculate_tau() && !calculate_forces) {
      integrate_kernel<false, false><<<nblocks, threads_per_block, smem_per_block,
          stream_>>>(*this, ctx_.ctx().first_task(level));
      return;
  }
  if (ctx_.ctx().calculate_tau() && !calculate_forces) {
      integrate_kernel<true, false><<<nblocks, threads_per_block, smem_per_block, stream_>>>(
          *this, ctx_.ctx().first_task(level));
      return;
  }

  if (!ctx_.ctx().calculate_tau() && calculate_forces) {
      integrate_kernel<false, true><<<nblocks, threads_per_block, smem_per_block,
          stream_>>>(*this, ctx_.ctx().first_task(level));
      return;
  }

  if (ctx_.ctx().calculate_tau() && calculate_forces) {
      integrate_kernel<true, true><<<nblocks, threads_per_block, smem_per_block,
          stream_>>>(*this, ctx_.ctx().first_task(level));
      return;
  }
}

#endif
