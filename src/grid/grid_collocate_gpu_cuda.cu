#include <cuda.h>
#include "collocation_integration.h"
__constant__ __device__ int grid_size_[3];
__constant__ __device__ int period_[3];
__constant__ __device__ double dh_[9];

extern __shared__  double array[];

__device__ void  return_cube_position(const int *grid_size,
                                      const int *lb_grid,
                                      const int *cube_center,
                                      const int *lower_boundaries_cube,
                                      const int *period,
                                      int *const position)
{
    position[0] = (lb_grid[0] + cube_center[0] + lower_boundaries_cube[0] + 32 * period[0]) % period[0];
    position[1] = (lb_grid[1] + cube_center[1] + lower_boundaries_cube[1] + 32 * period[1]) % period[1];
    position[2] = (lb_grid[2] + cube_center[2] + lower_boundaries_cube[2] + 32 * period[2]) % period[2];
}


__global__ void compute_collocation_gpu_(const int *__restrict__ lmax_gpu_,
                                         const double *__restrict__ zeta_gpu,
                                         const int *cube_size_gpu_,
                                         const int *cube_position_,
                                         const double *__restrict__ roffset_gpu_,
                                         const int *__restrict__ coef_offset_gpu_,
                                         const double *__restrict__ coef_gpu_,
                                         double *__restrict__ grid_gpu_)
{
    /* the period is sotred in constant memory */
    /* the displacement vectors as well */

    int lmax = lmax_gpu_[blockIdx.x];

    const int position[3] = {cube_position_[3 * blockIdx.x],
                             cube_position_[3 * blockIdx.x + 1],
                             cube_position_[3 * blockIdx.x + 2]};

    const int cube_size[3] = {cube_size_gpu_[3 * blockIdx.x],
                              cube_size_gpu_[3 * blockIdx.x + 1],
                              cube_size_gpu_[3 * blockIdx.x + 2]};

    const double roffset[3] = {roffset_gpu_[3 * blockIdx.x],
                               roffset_gpu_[3 * blockIdx.x + 1],
                               roffset_gpu_[3 * blockIdx.x + 2]};
    const double *__restrict__ coef = coef_gpu_ + coef_offset_gpu_[blockIdx.x];

    const double zeta = zeta_gpu[blockIdx.x];

    double  *coefs_ = (double *)array;

    int id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x +
        threadIdx.x;

    for (int i = id;
         i < ((lmax + 1) * (lmax + 1) * (lmax + 1));
         i += (blockDim.x * blockDim.y * blockDim.z))
        coefs_[i] = coef[i];
    __syncthreads();

    for (int z = threadIdx.z; z < cube_size[0]; z += blockDim.z) {
        const double z1 = z - (cube_size[0] / 2) - roffset[0];
        const int z2 = (z + position[0] + 32 * period_[0]) % period_[0];
        for (int y = threadIdx.y; y < cube_size[1]; y += blockDim.y) {
            double y1 = y - (cube_size[1] / 2) - roffset[1];
            const int y2 = (y + position[1] + 32 * period_[1]) % period_[1];
            for (int x = threadIdx.x; x < cube_size[2]; x += blockDim.x) {
                const double x1 = x - (cube_size[2] / 2) - roffset[2];
                const int x2 = (x + position[2] + 32 * period_[2]) % period_[2];
                double3 r3;
                r3.x = z1 * dh_[6] + y1 * dh_[3] + x1 * dh_[0];
                r3.y = z1 * dh_[7] + y1 * dh_[4] + x1 * dh_[1];
                r3.z = z1 * dh_[8] + y1 * dh_[5] + x1 * dh_[2];
                double exp_factor = exp(-(r3.x * r3.x + r3.y * r3.y + r3.z * r3.z) * zeta);
                double res = 0.0;
                double dx = 1;

                /* NOTE: the coefficients are stored as lx,lz,ly */

                for (int alpha = 0; alpha <= lmax; alpha++) {
                    double dz = 1;
                    for (int gamma = 0; gamma <= lmax; gamma++) {
                        double dy = dx * dz;
                        for (int beta = 0; beta <= lmax; beta++) {
                            res += coefs_[(alpha * (lmax + 1) + gamma) * (lmax + 1) + beta] * dy;
                            dy *= r3.y;
                        }
                        dz *= r3.z;
                    }
                    dx *= r3.x;
                }

                res *= exp_factor;
                atomicAdd(&grid_gpu_[(z2 * grid_size_[1] + y2) * grid_size_[2] + x2], res);
            }
        }
    }
}

extern "C"  void compute_collocation_gpu(pgf_list_gpu *handler)
{
    if (!handler)
        return;

    cudaStreamSynchronize(handler->stream);
    cudaMemcpyAsync(handler->cube_position_gpu_, handler->cube_position_cpu_, sizeof(int) * 3 * handler->list_length, cudaMemcpyHostToDevice, handler->stream);
    cudaMemcpyAsync(handler->cube_size_gpu_, handler->cube_size_cpu_, sizeof(int) * 3 * handler->list_length, cudaMemcpyHostToDevice, handler->stream);
    cudaMemcpyAsync(handler->roffset_gpu_, handler->roffset_cpu_, sizeof(double) * 3 * handler->list_length, cudaMemcpyHostToDevice, handler->stream);
    cudaMemcpyAsync(handler->zeta_gpu_, handler->zeta_cpu_, sizeof(double) * handler->list_length, cudaMemcpyHostToDevice, handler->stream);
    cudaMemcpyAsync(handler->lmax_gpu_, handler->lmax_cpu_, sizeof(int) * handler->list_length, cudaMemcpyHostToDevice, handler->stream);
    cudaMemcpyAsync(handler->coef_offset_gpu_, handler->coef_offset_cpu_, sizeof(int) * handler->list_length, cudaMemcpyHostToDevice, handler->stream);

    if (handler->durty) {
        cudaFree(handler->coef_gpu_);
        cudaMalloc(&handler->coef_gpu_, sizeof(double) * handler->coef_alloc_size_gpu_);
        handler->durty = false;
    }

    cudaMemcpyAsync(handler->coef_gpu_,
                    handler->coef_cpu_,
                    sizeof(double) * handler->coef_dynamic_alloc_size_gpu_,
                    cudaMemcpyHostToDevice,
                    handler->stream);

    dim3 block, thread;

    block.x = handler->list_length;

    thread.x = 4;
    thread.y = 4;
    thread.z = 4;
    compute_collocation_gpu_<<<block,
        thread,
        (handler->lmax + 1) * (handler->lmax + 1) * (handler->lmax + 1) * sizeof(double), handler->stream>>>(handler->lmax_gpu_,
                                                                                                             handler->zeta_gpu_,
                                                                                                             handler->cube_size_gpu_,
                                                                                                             handler->cube_position_gpu_,
                                                                                                             handler->roffset_gpu_,
                                                                                                             handler->coef_offset_gpu_,
                                                                                                             handler->coef_gpu_,
                                                                                                             handler->data_gpu_);
    reset_list_gpu(handler);
}

extern "C" void initialize_grid_parameters_on_gpu(collocation_integration *handler, const bool grid_resize, const int period[3])
{
    if (!handler->use_gpu)
        return;

    double dh[9];

    dh[0] = handler->dh[0][0];
    dh[1] = handler->dh[0][1];
    dh[2] = handler->dh[0][2];
    dh[3] = handler->dh[1][0];
    dh[4] = handler->dh[1][1];
    dh[5] = handler->dh[1][2];
    dh[6] = handler->dh[2][0];
    dh[7] = handler->dh[2][1];
    dh[8] = handler->dh[2][2];

    cudaMemcpyToSymbol(dh_, dh, sizeof(double) * 9);
    cudaMemcpyToSymbol(grid_size_, handler->grid.size, sizeof(int) * 3);
    cudaMemcpyToSymbol(period_, period, sizeof(int) * 3);

    if (grid_resize || (handler->worker_list->data_gpu_ == NULL)) {
        if (handler->worker_list->data_gpu_)
            cudaFree(handler->worker_list->data_gpu_);
        cudaMalloc(&handler->worker_list->data_gpu_, sizeof(double) * handler->grid.alloc_size_);
    }

    cudaMemset(handler->worker_list->data_gpu_, 0, sizeof(double) * handler->grid.alloc_size_);
    reset_list_gpu(handler->worker_list);
}
