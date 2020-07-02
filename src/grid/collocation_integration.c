#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#ifdef __COLLOCATE_GPU
#include <cuda.h>
#include <cublas_v2.h>
#endif

#include "tensor_local.h"
#include "utils.h"
#include "collocation_integration.h"
#include "non_orthorombic_corrections.h"

void apply_reduction_worker_list(collocation_integration* handler);

void*
collocate_create_handle(const int device_id, const int number_of_gaussian, const bool use_gpu)
{
    struct collocation_integration_* handle = NULL;
    handle = (struct collocation_integration_*)malloc(sizeof(struct collocation_integration_));

    if (handle == NULL) {
        abort();
    }
    memset(handle, 0, sizeof(struct collocation_integration_));

    if (device_id >= 0) {
#ifdef __COLLOCATE_GPU
        initialize_worker_list_on_gpu(handle, device_id, number_of_gaussian, 1, use_gpu);
#endif
    }

    handle->alpha.alloc_size_ = 8192;
    handle->coef.alloc_size_  = 1024;
    handle->pol.alloc_size_   = 1024;
    /* it is a cube of size 32 x 32 x 32 */
    handle->cube.alloc_size_ = 32768;

    handle->cube_alloc_size  = realloc_tensor(&handle->cube);
    handle->alpha_alloc_size = realloc_tensor(&handle->alpha);
    handle->coef_alloc_size  = realloc_tensor(&handle->coef);
    handle->pol_alloc_size   = realloc_tensor(&handle->pol);

    handle->scratch            = memalign(64, sizeof(double) * 10240);
    handle->scratch_alloc_size = 10240;
    handle->T_alloc_size       = 8192;
    handle->W_alloc_size       = 2048;
    handle->blockDim[0]        = 5;
    handle->blockDim[1]        = 5;
    handle->blockDim[2]        = 5;

    return (void*)handle;
}

void
collocate_synchronize(void* gaussian_handler)
{
    if (gaussian_handler == NULL) {
        abort();
    }

    struct collocation_integration_* handler = (struct collocation_integration_*)gaussian_handler;

    if (handler->integrate)
        return;

    if (!handler->grid_restored) {
#ifdef __COLLOCATE_GPU
        if (handler->use_gpu) {
            apply_reduction_worker_list(handler);
            handler->grid_restored = true;
            return;
        }
#endif
        if (handler->blocked_grid.blocked_decomposition) {
            add_blocked_tensor_to_tensor(&handler->blocked_grid, &handler->grid);
            memset(handler->blocked_grid.data, 0, sizeof(double) * handler->blocked_grid.alloc_size_);
            handler->grid_restored = true;
        }
    }
}

void
collocate_finalize(void* gaussian_handle)
{
    collocate_synchronize(gaussian_handle);
    struct collocation_integration_* handle = (struct collocation_integration_*)gaussian_handle;

#ifdef __COLLOCATE_GPU
    if (handle->use_gpu && !handle->integrate)
        release_gpu_resources(handle);
#endif

    if (handle->Exp.data)
        free(handle->Exp.data);

    free(handle->scratch);
    free(handle->pol.data);
    free(handle->cube.data);
    free(handle->blocks_coordinates.data);
    handle->alpha.data              = NULL;
    handle->coef.data               = NULL;
    handle->blocks_coordinates.data = NULL;
    free(handle);

    handle = NULL;
}

void
initialize_W_and_T(collocation_integration* const handler, const tensor* cube, const tensor* coef)
{
    size_t tmp1 =
        compute_memory_space_tensor_3(coef->size[0] /* alpha */, coef->size[1] /* gamma */, cube->size[1] /* j */);

    size_t tmp2 =
        compute_memory_space_tensor_3(coef->size[0] /* gamma */, cube->size[1] /* j */, cube->size[2] /* i */);

    const size_t mem_alloc_size_ = max(max(tmp1 + tmp2, cube->alloc_size_), coef->alloc_size_);

    handler->T_alloc_size = tmp1;
    handler->W_alloc_size = tmp2;

    if ((mem_alloc_size_ > handler->scratch_alloc_size) || (handler->scratch == NULL)) {

        handler->scratch_alloc_size = mem_alloc_size_;

        if (handler->scratch)
            free(handler->scratch);
        handler->scratch = memalign(64, sizeof(double) * handler->scratch_alloc_size);
        if (handler->scratch == NULL)
            abort();
    }
}

void
initialize_W_and_T_integrate(collocation_integration* const handler, const int num_block, const tensor* coef,
                             const tensor* block)
{
    /* T */
    size_t tmp1 = compute_memory_space_tensor_4(num_block, block->size[0] /* k */, block->size[1] /* j */,
                                                coef->size[1] /* alpha */);

    /* W */
    size_t tmp2 = compute_memory_space_tensor_4(num_block, block->size[1] /* j */, coef->size[1] /* alpha */,
                                                coef->size[2] /* gamma */);

    const size_t mem_alloc_size_ = tmp1 + tmp2;

    handler->T_alloc_size = tmp1;
    handler->W_alloc_size = tmp2;

    if ((mem_alloc_size_ > handler->scratch_alloc_size) || (handler->scratch == NULL)) {

        handler->scratch_alloc_size = mem_alloc_size_;

        if (handler->scratch)
            free(handler->scratch);
        handler->scratch = memalign(64, sizeof(double) * handler->scratch_alloc_size);
        if (handler->scratch == NULL)
            abort();
    }
}

void
initialize_basis_vectors(collocation_integration* const handler, const double dh[3][3], const double dh_inv[3][3])
{
    handler->dh[0][0] = dh[0][0];
    handler->dh[0][1] = dh[0][1];
    handler->dh[0][2] = dh[0][2];
    handler->dh[1][0] = dh[1][0];
    handler->dh[1][1] = dh[1][1];
    handler->dh[1][2] = dh[1][2];
    handler->dh[2][0] = dh[2][0];
    handler->dh[2][1] = dh[2][1];
    handler->dh[2][2] = dh[2][2];

    handler->dh_inv[0][0] = dh_inv[0][0];
    handler->dh_inv[0][1] = dh_inv[0][1];
    handler->dh_inv[0][2] = dh_inv[0][2];
    handler->dh_inv[1][0] = dh_inv[1][0];
    handler->dh_inv[1][1] = dh_inv[1][1];
    handler->dh_inv[1][2] = dh_inv[1][2];
    handler->dh_inv[2][0] = dh_inv[2][0];
    handler->dh_inv[2][1] = dh_inv[2][1];
    handler->dh_inv[2][2] = dh_inv[2][2];

    /* Only used when we are in the non  orthorombic case */
    handler->dx[2] = handler->dh[0][0] * handler->dh[0][0] + handler->dh[0][1] * handler->dh[0][1] +
                     handler->dh[0][2] * handler->dh[0][2];
    handler->dx[1] = handler->dh[1][0] * handler->dh[1][0] + handler->dh[1][1] * handler->dh[1][1] +
                     handler->dh[1][2] * handler->dh[1][2];
    handler->dx[0] = handler->dh[2][0] * handler->dh[2][0] + handler->dh[2][1] * handler->dh[2][1] +
                     handler->dh[2][2] * handler->dh[2][2];
}

void
initialize_grid(collocation_integration* handler, const bool use_ortho, const bool integrate, const double dh[3][3],
                const double dh_inv[3][3], const int npts[3], const int lb_grid[3], const int* ngrid, double* grid_)
{
    /* the data are durty */
    bool tmpt = handler->grid_restored;

    if (!integrate) {
        tmpt                   = handler->grid_restored;
        handler->grid_restored = false;
        handler->integrate     = false;
    } else {
        handler->use_gpu       = false;
        handler->integrate     = true;
        handler->grid_restored = true;
    }

    // we have a new grid. Note that checking if the results have been stored
    // into the original grid is only valid when I do collocate

    if ((handler->grid.size[0] != ngrid[2]) || (handler->grid.size[1] != ngrid[1]) ||
        (handler->grid.size[2] != ngrid[0])) {

        // Only test here if I do collocate
        if ((handler->grid.data != NULL) && (!tmpt) && (handler->blocked_grid.blocked_decomposition) && (!integrate)) {
            printf("Warning : you forgot to restore the grid.\n");
            printf("You should call collocate_synchronize before switching to a new grid\n");
            abort();
        }

        initialize_basis_vectors(handler, dh, dh_inv);
        verify_orthogonality(dh, handler->orthogonal);

        initialize_tensor_3(&handler->grid, ngrid[2], ngrid[1], ngrid[0]);
        handler->grid.ld_  = ngrid[0];
        handler->grid.data = grid_;

        int grid_reverse[3] = {ngrid[2], ngrid[1], ngrid[0]};

#ifdef __COLLOCATE_GPU
        if (!handler->integrate && handler->use_gpu) {
            grid_reverse[0] = npts[2];
            grid_reverse[1] = npts[1];
            grid_reverse[2] = npts[0];
            initialize_grid_parameters_on_gpu(handler, (handler->grid.old_alloc_size_ < handler->grid.alloc_size_),
                                              grid_reverse);
            reset_list_gpu(handler->worker_list);
            return;
        }
#endif

        /* if (!integrate) { */
        /*     // i shuffle the grid such that we have the grid stored in the yxz */
        /*     // form. The reason for that is linked to the way i access the block */
        /*     // when I collocate. I have three loops and the most outer loop is along y then x then z */
        /*     grid_reverse[0] = ngrid[1]; */
        /*     grid_reverse[1] = ngrid[0]; */
        /*     grid_reverse[2] = ngrid[2]; */
        /* } */

        compute_block_dimensions(grid_reverse, handler->blockDim);
        initialize_tensor_blocked(&handler->blocked_grid, 3, grid_reverse, handler->blockDim);

        /* if (!integrate) { */
        /*     assert(handler->grid.size[0] % handler->blockDim[2] == 0); */
        /*     assert(handler->grid.size[1] % handler->blockDim[0] == 0); */
        /*     assert(handler->grid.size[2] % handler->blockDim[1] == 0); */
        /* } else { */
        assert(handler->grid.size[0] % handler->blockDim[0] == 0);
        assert(handler->grid.size[1] % handler->blockDim[1] == 0);
        assert(handler->grid.size[2] % handler->blockDim[2] == 0);
        /* } */

        realloc_tensor(&handler->blocked_grid);

        if (integrate) {
            decompose_grid_to_blocked_grid(&handler->grid, &handler->blocked_grid);
        } else {
            memset(handler->blocked_grid.data, 0, sizeof(double) * handler->blocked_grid.alloc_size_);
        }
    }
}

#if defined(__COLLOCATE_GPU)

void
reset_list_gpu(pgf_list_gpu* handler)
{
    cudaSetDevice(handler->device_id);
    handler->list_length                  = 0;
    handler->coef_dynamic_alloc_size_gpu_ = 0;
}

void
apply_reduction_worker_list(collocation_integration* handler)
{
    for (int i = 0; i < handler->worker_list_size; i++) {
        if (handler->worker_list->list_length != 0) {
            compute_collocation_gpu(handler->worker_list + i);
            /* actual collocation is done in a stream */
            reset_list_gpu(handler->worker_list + i);
        }
    }
    cudaStreamSynchronize(handler->worker_list[0].stream);

    for (int i = 1; i < handler->worker_list_size; i++) {
        double one = 1.0;
        cudaSetDevice(handler->device_id);
        cudaStreamSynchronize(handler->worker_list[i].stream);
        cublasDaxpy(handler->worker_list[0].blas_handle, handler->grid.alloc_size_, &one,
                    handler->worker_list[i].data_gpu_, 1, handler->worker_list[0].data_gpu_, 1);
        cudaMemset(handler->worker_list[i].data_gpu_, 0, sizeof(double) * handler->grid.alloc_size_);
    }

    double* tmp = malloc(sizeof(double) * handler->grid.alloc_size_);
    cudaMemcpy(tmp, handler->worker_list->data_gpu_, sizeof(double) * handler->grid.alloc_size_,
               cudaMemcpyDeviceToHost);
    cblas_daxpy(handler->grid.alloc_size_, 1.0, tmp, 1, handler->grid.data, 1);
    cudaMemset(handler->worker_list->data_gpu_, 0, sizeof(double) * handler->grid.alloc_size_);
    free(tmp);
}
#endif
