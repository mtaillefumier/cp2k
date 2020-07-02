#ifndef __COLLOCATION_INTEGRATION_H
#define __COLLOCATION_INTEGRATION_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef __COLLOCATE_GPU
#include <cuda.h>
#include <cublas_v2.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor_local.h"
#ifdef __COLLOCATE_GPU
typedef struct pgf_list_gpu_
{
    /* device_id */
    int device_id;
    /* */
    int lmax;

    /* maximum size of the batch */
    int batch_size;

    /* size of the batch */
    int list_length;

    /* number of elements occupied in the buffer */
    size_t coef_dynamic_alloc_size_gpu_;

    /*  total size of the buffer */
    size_t coef_alloc_size_gpu_;

    /* size of the previously allocated coefficent table */
    size_t coef_previous_alloc_size_;

    double* coef_cpu_;
    double* coef_gpu_;

    /* Info about the cubes */
    int* cube_size_cpu_;
    int* cube_position_cpu_;
    int* coef_offset_cpu_;
    double* roffset_cpu_;

    int* cube_size_gpu_;
    int* cube_position_gpu_;
    int* coef_offset_gpu_;
    double* roffset_gpu_;

    /* angular momentum */
    int* lmax_cpu_;
    int* lmax_gpu_;

    double* zeta_cpu_;
    double* zeta_gpu_;

    double* data_gpu_;

    cudaStream_t stream;
    bool job_finished;

    cublasHandle_t blas_handle;

    int3 grid_size, grid_lower_corner_position, period;

    struct pgf_list_gpu_* next;
    /* if true, the grid on the gpu should be reallocated */
    bool durty;
} pgf_list_gpu;
#endif

typedef struct collocation_integration_
{
    /* GPU device id. should replace this with GPU UID */
    int device_id;
    bool use_gpu;

    /* number of gaussians block in each list */
    int number_of_gaussian;

    /* some scratch storage to avoid malloc / free all the time */
    tensor alpha;
    tensor pol;
    tensor coef;

    /* tensors for the grid to collocate or integrate */
    /* original grid */
    tensor grid;

    int period[3];
    int lb_grid[3];

    /* original grid decomposed in block */
    tensor blocked_grid;

    /* do we need to update the grid */
    bool grid_restored;

    /* coordinates of the blocks */
    tensor blocks_coordinates;

    double dh[3][3];
    double dh_inv[3][3];
    double dx[3];

    /* block dimensions */
    int blockDim[4];

    /* Only allocated in sequential mode */
    tensor cube;
    tensor Exp;
    size_t Exp_alloc_size;
    size_t cube_alloc_size;
    size_t coef_alloc_size;
    size_t alpha_alloc_size;
    size_t pol_alloc_size;
    size_t scratch_alloc_size;
    size_t T_alloc_size;
    size_t W_alloc_size;
    int lmax;

    void* scratch;
#ifdef __COLLOCATE_GPU
    pgf_list_gpu* worker_list;
#endif
    int worker_list_size;

    bool durty;
    bool orthogonal[3];
    bool integrate;

    /* bool sequential_mode; */

} collocation_integration;

extern void* collocate_create_handle(const int device_id, const int number_of_gaussian, const bool sequential_mode);
extern void collocate_synchronize(void* gaussian_handler);
extern void collocate_finalize(void* gaussian_handle);
extern void calculate_collocation(void* const in);
extern void initialize_W_and_T(collocation_integration* const handler, const tensor* cube, const tensor* coef);
extern void initialize_basis_vectors(collocation_integration* const handler, const double dh[3][3],
                                     const double dh_inv[3][3]);
extern void initialize_grid(collocation_integration* handler, const bool use_ortho, const bool integrate,
                            const double dh[3][3], const double dh_inv[3][3], const int npts[3], const int lb_grid[3],
                            const int* ngrid, double* grid_);

#ifdef __COLLOCATE_GPU
extern void release_gpu_resources(collocation_integration* handler);
extern void initialize_worker_list_on_gpu(collocation_integration* handler, const int device_id,
                                          const int number_of_gaussian, const int number_of_worker, const bool use_gpu);
extern void initialize_grid_parameters_on_gpu(collocation_integration* handler, const bool grid_resize,
                                              const int period[3]);

extern void reset_list_gpu(pgf_list_gpu* handler);
extern void compute_collocation_gpu(pgf_list_gpu* handler);
extern void add_orbital_to_list(pgf_list_gpu* list, const int lp, const int cube_size[3], const int cube_position[3],
                                const double roffset[3], const double zetp, const tensor* const coef);
#endif

#ifdef __cplusplus
}
#endif
#endif
